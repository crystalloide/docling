"""
ingest.py — Ingestion de PDFs en production
  - Parsing GPU-accéléré via Docling
  - Gros PDFs (> LARGE_PDF_THRESHOLD pages) : traitement par tranches
    via pymupdf + Docling, pour éviter les std::bad_alloc du modèle de layout
  - Chunking natif Docling HybridChunker :
      * respecte la structure du document (articles, sections, tableaux)
      * calé sur le tokenizer de nomic-embed-text (max 384 tokens/chunk)
      * fusionne automatiquement les petits chunks de même section
  - Embedding batch via Ollama (nomic-embed-text)
  - Indexation idempotente dans Qdrant (reprise sans doublons)

  Dépendances supplémentaires vs version précédente :
    pip install transformers
  (docling et docling-core sont déjà dans requirements.txt)
"""

import os
import uuid
import sys
import requests
import tempfile
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

from tqdm import tqdm
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType
)

from config import (
    PDF_DIR, QDRANT_URL, COLLECTION,
    EMBED_MODEL, EMBED_DIM, CHUNK_SIZE, CHUNK_OVERLAP  # CHUNK_SIZE/OVERLAP non utilisés ici
)

EMBED_BATCH         = 128   # chunks envoyés à Ollama en une seule requête
UPSERT_BATCH        = 100   # points envoyés à Qdrant en une seule requête
LARGE_PDF_THRESHOLD = 100   # pages : au-delà, traitement par tranches
PAGES_PER_CHUNK     = 50    # taille d'une tranche (pages)
MAX_TOKENS_PER_CHUNK = 384  # tokens/chunk — nomic-embed-text est limité à 512,
                             # on garde 128 tokens de marge pour les métadonnées


# Pour vérifier la charge pendant l'ingestion, ouvrez un terminal et tapez :
# Dans un terminal PowerShell :  nvidia-smi -l 1
# Vous devriez voir le pourcentage de "Volatile GPU-Util" grimper bien au-delà des 10-20% habituels.
# Si vous approchez des 22-23 Go de VRAM, baissez layout_batch_size.


# ── Vérification Ollama au démarrage ──────────────────────────────────────
def check_ollama():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("❌ Ollama n'est pas accessible sur http://localhost:11434")
        print("   → Démarrez Ollama (icône systray) ou lancez : ollama serve")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur connexion Ollama : {e}")
        sys.exit(1)

    models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
    if EMBED_MODEL.split(":")[0] not in models:
        print(f"❌ Modèle '{EMBED_MODEL}' absent d'Ollama.")
        print(f"   → Téléchargez-le : ollama pull {EMBED_MODEL}")
        sys.exit(1)

    print(f"[Ollama] ✅ Connecté — modèle embedding : {EMBED_MODEL}")


# ── Initialisation Docling GPU ─────────────────────────────────────────────
def build_docling_converter():
    try:
        import torch
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA non disponible")

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[Docling] GPU : {torch.cuda.get_device_name(0)} ({vram_gb:.1f} Go VRAM)")

        accel = AcceleratorOptions(device=AcceleratorDevice.CUDA)
        pipeline_opts = ThreadedPdfPipelineOptions(
            accelerator_options=accel,
            do_ocr=True,
            do_table_structure=True,
            layout_batch_size=2,
            table_batch_size=1,
            images_scale=1.0,
            num_threads=8,
            generate_page_images=False,
            generate_picture_images=False,
        )
        conv = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
        )
        print("[Docling] ✅ GPU CUDA activé (OCR activé)")
        return conv

    except Exception as e:
        print(f"[Docling] ⚠ Fallback CPU ({e})")
        from docling.document_converter import DocumentConverter
        return DocumentConverter()


# ── Initialisation HybridChunker ──────────────────────────────────────────
def build_chunker():
    """
    HybridChunker calé sur le tokenizer de nomic-embed-text.
    Tente de charger le tokenizer HuggingFace exact ; si indisponible
    (pas d'accès réseau, etc.), utilise le tokenizer par défaut de Docling.

    Le HybridChunker :
      - respecte les frontières naturelles du document (articles, sections)
      - ne coupe jamais un tableau en plein milieu
      - fusionne les petits chunks consécutifs de même section (merge_peers=True)
    """
    from docling.chunking import HybridChunker

    try:
        from transformers import AutoTokenizer
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

        hf_tok = AutoTokenizer.from_pretrained(
            "nomic-ai/nomic-embed-text-v1",
            trust_remote_code=True,
        )
        tokenizer = HuggingFaceTokenizer(
            tokenizer=hf_tok,
            max_tokens=MAX_TOKENS_PER_CHUNK,
        )
        chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
        print(f"[Chunker] ✅ HybridChunker — tokenizer nomic-embed-text, max {MAX_TOKENS_PER_CHUNK} tokens/chunk")

    except Exception as e:
        # Fallback : tokenizer par défaut de Docling (BERT-like, ~même granularité)
        chunker = HybridChunker(merge_peers=True)
        print(f"[Chunker] ⚠ HybridChunker — tokenizer par défaut ({e})")

    return chunker


# ── Sérialisation d'un chunk en texte à embedder ─────────────────────────
def serialize_chunk(chunker, chunk) -> str:
    """
    Utilise contextualize() au lieu de serialize() pour 
    suivre les dernières mises à jour de Docling.
    """
    try:
        # Remplacement de chunker.serialize(chunk=chunk) par :
        return chunker.contextualize(chunk=chunk) 
    except AttributeError:
        # Fallback si votre version de docling est encore ancienne
        return chunker.serialize(chunk=chunk)
    except Exception:
        return chunk.text or ""


# ── Extraction des métadonnées de section ────────────────────────────────
def get_headings(chunk) -> list:
    """Retourne la liste des titres de section du chunk (ex: ['Titre II', 'Article 12'])."""
    try:
        return list(chunk.meta.headings) if chunk.meta and chunk.meta.headings else []
    except Exception:
        return []


# ── Parsing d'une tranche → DoclingDocument ──────────────────────────────
def parse_slice_to_doc(tmp_path: Path, converter):
    """Parse un fichier PDF temporaire et retourne le DoclingDocument natif."""
    result = converter.convert(str(tmp_path))
    return result.document


# ── Parsing + chunking pour les gros PDFs (par tranches) ─────────────────
def parse_and_chunk_large(pdf_path: Path, converter, chunker, page_count: int) -> list:
    """
    Pour les PDFs > LARGE_PDF_THRESHOLD pages :
    découpe en tranches de PAGES_PER_CHUNK pages, parse chaque tranche
    avec Docling, applique HybridChunker sur chaque DoclingDocument,
    retourne la liste consolidée de tous les chunks.
    """
    import fitz

    doc   = fitz.open(str(pdf_path))
    all_chunks = []
    slices     = list(range(0, page_count, PAGES_PER_CHUNK))

    for start in tqdm(slices, desc=f"  Tranches ({page_count} pages)", leave=False):
        end = min(start + PAGES_PER_CHUNK, page_count)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            sub = fitz.open()
            sub.insert_pdf(doc, from_page=start, to_page=end - 1)
            sub.save(str(tmp_path))
            sub.close()

            dl_doc = parse_slice_to_doc(tmp_path, converter)
            slice_chunks = list(chunker.chunk(dl_doc=dl_doc))
            all_chunks.extend(slice_chunks)

        except Exception as e:
            print(f"     ⚠ Erreur tranche pages {start}-{end} : {e}")
        finally:
            tmp_path.unlink(missing_ok=True)

    doc.close()
    return all_chunks


# ── Routeur de parsing + chunking ─────────────────────────────────────────
def parse_and_chunk(pdf_path: Path, converter, chunker) -> list:
    """
    Retourne la liste des chunks Docling pour un PDF.
    Applique le traitement par tranches pour les gros PDFs.
    """
    import fitz
    doc        = fitz.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    if page_count > LARGE_PDF_THRESHOLD:
        n_slices = -(-page_count // PAGES_PER_CHUNK)
        print(f"       ℹ  {page_count} pages → {n_slices} tranches de {PAGES_PER_CHUNK} (Docling GPU + HybridChunker)")
        return parse_and_chunk_large(pdf_path, converter, chunker, page_count)
    else:
        print(f"       ℹ  {page_count} pages → Docling GPU direct + HybridChunker")
        result = converter.convert(str(pdf_path))
        return list(chunker.chunk(dl_doc=result.document))


# ── Initialisation Qdrant ──────────────────────────────────────────────────
def init_qdrant(client: QdrantClient):
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print(f"[Qdrant] Collection '{COLLECTION}' créée ({EMBED_DIM} dims, COSINE)")
    else:
        count = client.get_collection(COLLECTION).points_count
        print(f"[Qdrant] Collection '{COLLECTION}' existante — {count} vecteurs")


# ── Fichiers déjà indexés ──────────────────────────────────────────────────
def get_indexed_sources(client: QdrantClient) -> set:
    sources, offset = set(), None
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=None,
            with_payload=["source"],
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        for point in results:
            sources.add(point.payload.get("source", ""))
        if offset is None:
            break
    return sources


# ── ID déterministe (upsert idempotent) ───────────────────────────────────
def make_point_id(pdf_path: str, chunk_index: int) -> str:
    key = f"{os.path.abspath(pdf_path)}::{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


# ── Ingestion d'un seul PDF ────────────────────────────────────────────────
def process_pdf(pdf_path: Path, converter, chunker, client: QdrantClient) -> int:
    # 1. Parsing + chunking natif Docling
    chunks = parse_and_chunk(pdf_path, converter, chunker)
    if not chunks:
        print(f"  ⚠ Aucun chunk extrait de {pdf_path.name}")
        return 0

    # 2. Sérialisation des chunks en texte (Markdown enrichi avec titres de section)
    texts = [serialize_chunk(chunker, c) for c in chunks]
    # Filtre les chunks vides après sérialisation
    pairs = [(t, c) for t, c in zip(texts, chunks) if t.strip()]
    if not pairs:
        return 0
    texts, chunks = zip(*pairs)

    print(f"       ℹ  {len(chunks)} chunks extraits par HybridChunker")

    # 3. Embedding par batch
    all_points = []
    for batch_start in tqdm(
        range(0, len(texts), EMBED_BATCH),
        desc=f"  Embedding ({len(texts)} chunks)",
        leave=False
    ):
        batch_texts  = texts[batch_start: batch_start + EMBED_BATCH]
        batch_chunks = chunks[batch_start: batch_start + EMBED_BATCH]

        response = ollama.embed(model=EMBED_MODEL, input=list(batch_texts))
        vectors  = response.embeddings

        for j, (text, chunk, vec) in enumerate(zip(batch_texts, batch_chunks, vectors)):
            all_points.append(PointStruct(
                id=make_point_id(str(pdf_path), batch_start + j),
                vector=vec,
                payload={
                    "text":     text,
                    "source":   pdf_path.name,
                    "chunk_id": batch_start + j,
                    "headings": get_headings(chunk),  # ex: ["Titre II", "Article 12"]
                },
            ))

    # 4. Upsert par batch dans Qdrant
    for start in range(0, len(all_points), UPSERT_BATCH):
        client.upsert(
            collection_name=COLLECTION,
            points=all_points[start: start + UPSERT_BATCH],
        )

    return len(all_points)


# ── Point d'entrée ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Production — Ingestion de PDFs")
    print("=" * 60)

    check_ollama()

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"\n❌ Aucun PDF trouvé dans '{PDF_DIR}'")
        print("   Placez vos fichiers .pdf dans ce répertoire et relancez.")
        sys.exit(0)

    converter = build_docling_converter()
    chunker   = build_chunker()
    client    = QdrantClient(url=QDRANT_URL)
    init_qdrant(client)

    already    = get_indexed_sources(client)
    to_process = [p for p in pdf_files if p.name not in already]
    skipped    = len(pdf_files) - len(to_process)

    print(f"\n📂 {len(pdf_files)} PDF(s) trouvé(s)")
    print(f"   ↳ {skipped} déjà indexé(s), {len(to_process)} à traiter\n")

    if not to_process:
        print("✅ Tous les fichiers sont déjà indexés.")
        sys.exit(0)

    total_chunks = 0
    errors = []
    for i, pdf in enumerate(to_process, 1):
        print(f"[{i}/{len(to_process)}] 🔄 {pdf.name}")
        try:
            n = process_pdf(pdf, converter, chunker, client)
            total_chunks += n
            print(f"       ✅ {n} vecteurs indexés")
        except Exception as e:
            errors.append((pdf.name, str(e)))
            print(f"       ❌ Erreur : {e}")

    print("\n" + "=" * 60)
    print("✅ Ingestion terminée")
    print(f"   Fichiers traités : {len(to_process) - len(errors)}/{len(to_process)}")
    print(f"   Vecteurs insérés : {total_chunks}")
    if errors:
        print(f"   Erreurs ({len(errors)}) :")
        for name, err in errors:
            print(f"     • {name} → {err}")
    total = client.get_collection(COLLECTION).points_count
    print(f"   Total en base    : {total} vecteurs")
    print("=" * 60)
