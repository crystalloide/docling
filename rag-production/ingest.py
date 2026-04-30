"""
ingest.py — Ingestion de PDFs en production
  - Parsing GPU-accéléré via Docling
  - Gros PDFs (> LARGE_PDF_THRESHOLD pages) : traitement par tranches
    via pymupdf + Docling, pour éviter les std::bad_alloc du modèle de layout
  - Chunking sémantique via RecursiveCharacterTextSplitter
  - Embedding batch via Ollama (nomic-embed-text)
  - Indexation idempotente dans Qdrant (reprise sans doublons)
"""

import os
import uuid
import sys
import requests
import tempfile
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType
)

from config import (
    PDF_DIR, QDRANT_URL, COLLECTION,
    EMBED_MODEL, EMBED_DIM, CHUNK_SIZE, CHUNK_OVERLAP
)

EMBED_BATCH       = 16    # chunks envoyés à Ollama en une seule requête
UPSERT_BATCH      = 100   # points envoyés à Qdrant en une seule requête
LARGE_PDF_THRESHOLD = 100  # pages : au-delà, traitement par tranches
PAGES_PER_CHUNK   = 50    # taille d'une tranche (pages)


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
            layout_batch_size=4,
            table_batch_size=1,
            images_scale=1.0,
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


# ── Parsing d'une tranche (fichier PDF temporaire) ────────────────────────
def parse_slice(tmp_path: Path, converter) -> str:
    result = converter.convert(str(tmp_path))
    return result.document.export_to_markdown()


# ── Parsing par tranches pour les gros PDFs ───────────────────────────────
def parse_with_docling_chunked(pdf_path: Path, converter, page_count: int) -> str:
    """
    Découpe le PDF en tranches de PAGES_PER_CHUNK pages,
    parse chaque tranche avec Docling, concatène le Markdown.
    Évite les std::bad_alloc du modèle de layout sur les gros documents.
    """
    import fitz  # pymupdf

    doc = fitz.open(str(pdf_path))
    parts = []
    slices = list(range(0, page_count, PAGES_PER_CHUNK))

    for start in tqdm(slices, desc=f"  Tranches ({page_count} pages)", leave=False):
        end = min(start + PAGES_PER_CHUNK, page_count)

        # Écriture d'un PDF temporaire pour la tranche
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            sub = fitz.open()
            sub.insert_pdf(doc, from_page=start, to_page=end - 1)
            sub.save(str(tmp_path))
            sub.close()

            markdown = parse_slice(tmp_path, converter)
            if markdown.strip():
                parts.append(markdown)
        finally:
            tmp_path.unlink(missing_ok=True)  # nettoyage systématique

    doc.close()
    return "\n\n".join(parts)


# ── Routeur de parsing ─────────────────────────────────────────────────────
def parse_pdf(pdf_path: Path, converter) -> str:
    import fitz
    doc = fitz.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    if page_count > LARGE_PDF_THRESHOLD:
        n_slices = -(-page_count // PAGES_PER_CHUNK)  # division arrondie au-dessus
        print(f"       ℹ  {page_count} pages → {n_slices} tranches de {PAGES_PER_CHUNK} (Docling GPU)")
        return parse_with_docling_chunked(pdf_path, converter, page_count)
    else:
        print(f"       ℹ  {page_count} pages → Docling GPU direct")
        return parse_slice(pdf_path, converter)


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
def process_pdf(pdf_path: Path, converter, splitter, client: QdrantClient) -> int:
    # 1. Parsing (direct ou par tranches selon le nombre de pages)
    text = parse_pdf(pdf_path, converter)
    if not text.strip():
        print(f"  ⚠ Aucun texte extrait de {pdf_path.name}")
        return 0

    # 2. Chunking sémantique
    chunks = splitter.split_text(text)
    if not chunks:
        return 0

    # 3. Embedding par batch
    all_points = []
    for batch_start in tqdm(
        range(0, len(chunks), EMBED_BATCH),
        desc=f"  Embedding ({len(chunks)} chunks)",
        leave=False
    ):
        batch = chunks[batch_start: batch_start + EMBED_BATCH]
        response = ollama.embed(model=EMBED_MODEL, input=batch)
        vectors = response.embeddings
        for j, (chunk, vec) in enumerate(zip(batch, vectors)):
            all_points.append(PointStruct(
                id=make_point_id(str(pdf_path), batch_start + j),
                vector=vec,
                payload={
                    "text":     chunk,
                    "source":   pdf_path.name,
                    "chunk_id": batch_start + j,
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    client = QdrantClient(url=QDRANT_URL)
    init_qdrant(client)

    already = get_indexed_sources(client)
    to_process = [p for p in pdf_files if p.name not in already]
    skipped = len(pdf_files) - len(to_process)

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
            n = process_pdf(pdf, converter, splitter, client)
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
