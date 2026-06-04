"""
ingest.py — Ingestion de PDFs avec Tagging Sémantique Multi-Domaine
  - Analyse des premières pages par LLM pour catégoriser (Domaine, Type)
  - Ajout des métadonnées globales à CHAQUE chunk
  - Parsing via Docling et Indexation Qdrant
"""

import os
import uuid
import sys
import requests
import json
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
    EMBED_MODEL, EMBED_DIM, LLM_MODEL
)

EMBED_BATCH         = 128
UPSERT_BATCH        = 100
LARGE_PDF_THRESHOLD = 100
PAGES_PER_CHUNK     = 50
MAX_TOKENS_PER_CHUNK = 384

def check_ollama():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"❌ Erreur connexion Ollama : {e}")
        sys.exit(1)
    print(f"[Ollama] ✅ Connecté")

def build_docling_converter():
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    import torch

    # Configuration avancée du pipeline
    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = True  # Force l'OCR
    pipeline_opts.do_table_structure = True
    
    # Utilisation du GPU si dispo pour accélérer l'OCR
    if torch.cuda.is_available():
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        pipeline_opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
        }
    )

def build_chunker():
    from docling.chunking import HybridChunker
    try:
        from transformers import AutoTokenizer
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        hf_tok = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        tokenizer = HuggingFaceTokenizer(tokenizer=hf_tok, max_tokens=MAX_TOKENS_PER_CHUNK)
        return HybridChunker(tokenizer=tokenizer, merge_peers=True)
    except Exception:
        return HybridChunker(merge_peers=True)

def serialize_chunk(chunker, chunk) -> str:
    try: return chunker.contextualize(chunk=chunk)
    except Exception: return chunk.text or ""

def get_headings(chunk) -> list:
    try: return list(chunk.meta.headings) if chunk.meta and chunk.meta.headings else []
    except Exception: return []

# ── NOUVEAUTÉ : Analyse et Tagging du document par le LLM ──────────────────
def analyze_document_metadata(pdf_path: Path) -> dict:
    """
    Lit le tout début du document pour demander au LLM d'en déduire 
    le domaine, le type de document et un résumé, sans aucune règle préétablie.
    """
    import fitz
    doc = fitz.open(str(pdf_path))
    # Extraction du texte des 3 premières pages pour donner du contexte au LLM
    sample_text = ""
    for i in range(min(3, len(doc))):
        page_text = doc[i].get_text()
        # Sécurité supplémentaire : si une seule page est anormalement gigantesque, 
        # on la coupe pour éviter de saturer le tokenizer ou le LLM
        if len(page_text) > 4000:
            page_text = page_text[:4000]
        sample_text += page_text + "\n"
    doc.close()

    prompt = (
        f"Analyse cet extrait du document '{pdf_path.name}' pour le classifier dans une base de connaissances d'entreprise.\n\n"
        f"Génère un objet JSON STRICT avec ces 3 clés :\n"
        f"1. 'domain' : Le grand domaine fonctionnel ou direction métier concernée (ex: 'Ressources Humaines', 'Juridique', 'Informatique & Infra', 'Finance & Comptabilité', 'R&D', 'Logistique', 'Santé', etc.). Sois macro et standard.\n"
        f"2. 'document_type' : Le format du document (ex: 'Curriculum Vitae', 'Code de lois', 'Facture', 'Rapport Technique', 'Contrat', 'Procédure Interne').\n"
        f"3. 'summary' : Une phrase de résumé global.\n\n"
        f"Extrait :\n{sample_text}\n\n"
        f"Répond UNIQUEMENT avec le bloc JSON."
    )

    try:
        res = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.0})
        content = res["message"]["content"].strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"): content = content[4:]
        meta = json.loads(content.strip())
        print(f"       🏷️  Classification : [{meta.get('domain')}] -> {meta.get('document_type')}")
        return meta
    except Exception as e:
        return {"domain": "Général", "document_type": "Document", "summary": ""}

def parse_and_chunk(pdf_path: Path, converter, chunker) -> list:
    import fitz
    doc = fitz.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    if page_count > LARGE_PDF_THRESHOLD:
        return parse_and_chunk_large(pdf_path, converter, chunker, page_count)
    else:
        result = converter.convert(str(pdf_path))
        return list(chunker.chunk(dl_doc=result.document))

def parse_and_chunk_large(pdf_path: Path, converter, chunker, page_count: int) -> list:
    import fitz
    doc = fitz.open(str(pdf_path))
    all_chunks = []
    slices = list(range(0, page_count, PAGES_PER_CHUNK))
    for start in tqdm(slices, desc="  Tranches", leave=False):
        end = min(start + PAGES_PER_CHUNK, page_count)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            sub = fitz.open()
            sub.insert_pdf(doc, from_page=start, to_page=end - 1)
            sub.save(str(tmp_path))
            sub.close()
            result = converter.convert(str(tmp_path))
            all_chunks.extend(list(chunker.chunk(dl_doc=result.document)))
        finally:
            tmp_path.unlink(missing_ok=True)
    doc.close()
    return all_chunks

def init_qdrant(client: QdrantClient):
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        # Indexation des payloads stratégiques pour accélérer les filtrages dynamiques
        client.create_payload_index(collection_name=COLLECTION, field_name="source", field_schema=PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name=COLLECTION, field_name="domain", field_schema=PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name=COLLECTION, field_name="document_type", field_schema=PayloadSchemaType.KEYWORD)
        print(f"[Qdrant] Collection '{COLLECTION}' créée avec index de filtrage.")

def get_indexed_sources(client: QdrantClient) -> set:
    sources, offset = set(), None
    while True:
        results, offset = client.scroll(collection_name=COLLECTION, with_payload=["source"], with_vectors=False, limit=1000, offset=offset)
        for point in results: sources.add(point.payload.get("source", ""))
        if offset is None: break
    return sources

def make_point_id(pdf_path: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{os.path.abspath(pdf_path)}::{chunk_index}"))

def process_pdf(pdf_path: Path, converter, chunker, client: QdrantClient) -> int:
    # 1. Analyse et génération des métadonnées sémantiques globales du document
    doc_metadata = analyze_document_metadata(pdf_path)

    chunks = parse_and_chunk(pdf_path, converter, chunker)
    if not chunks: return 0

    texts = [serialize_chunk(chunker, c) for c in chunks]
    pairs = [(t, c) for t, c in zip(texts, chunks) if t.strip()]
    if not pairs: return 0
    texts, chunks = zip(*pairs)

    all_points = []
    for batch_start in tqdm(range(0, len(texts), EMBED_BATCH), desc="  Embedding", leave=False):
        batch_texts = texts[batch_start: batch_start + EMBED_BATCH]
        batch_chunks = chunks[batch_start: batch_start + EMBED_BATCH]

        # 🚀 ÉTAPE 1 : On prépare les textes enrichis AVANT de générer l'embedding
        batch_enriched_texts = []
        for text, chunk in zip(batch_texts, batch_chunks):
            # On nettoie un peu le nom du fichier pour enlever l'extension .pdf pour l'embedding
            clean_name = pdf_path.stem.replace("_", " ")
            enriched = (
                f"Document source: {clean_name} | Domaine: {doc_metadata.get('domain')} | "
                f"Type: {doc_metadata.get('document_type')}\n\n{text}"
            )
            batch_enriched_texts.append(enriched)

        # 🚀 ÉTAPE 2 : On vectorise le texte ENRICHI (comme ça, le mot "Raph" ou "Raphaël" fait partie intégrante du vecteur !)
        response = ollama.embed(model=EMBED_MODEL, input=batch_enriched_texts)
        vectors = response.embeddings

        for j, (text, chunk, vec) in enumerate(zip(batch_texts, batch_chunks, vectors)):
            clean_name = pdf_path.name
            # On crée un texte que le LLM ne pourra pas confondre
            meta_header = f"--- DEBUT DU BLOC (SOURCE: {clean_name} | DOMAINE: {doc_metadata.get('domain')}) ---\n"
            enriched_text = f"{meta_header}{text}\n--- FIN DU BLOC ({clean_name}) ---"

            all_points.append(PointStruct(
                id=make_point_id(str(pdf_path), batch_start + j),
                vector=vec, # Ici on garde l'embedding du texte enrichi ou brut
                payload={
                    "text": enriched_text, # 🚀 On envoie le texte balisé au LLM
                    "source": pdf_path.name,
                    "chunk_id": batch_start + j,
                    "headings": get_headings(chunk),
                    "domain": doc_metadata.get("domain"),
                    "document_type": doc_metadata.get("document_type")
                },
            ))

    for start in range(0, len(all_points), UPSERT_BATCH):
        client.upsert(collection_name=COLLECTION, points=all_points[start: start + UPSERT_BATCH])

    return len(all_points)

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Multi-Domaine — Ingestion Automatisée")
    print("=" * 60)
    check_ollama()
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"\n❌ Aucun PDF dans '{PDF_DIR}'")
        sys.exit(0)

    converter = build_docling_converter()
    chunker = build_chunker()
    client = QdrantClient(url=QDRANT_URL)
    init_qdrant(client)

    already = get_indexed_sources(client)
    to_process = [p for p in pdf_files if p.name not in already]
    print(f"\n📂 {len(pdf_files)} PDF(s) trouvé(s) (Déjà indexés: {len(pdf_files) - len(to_process)})")

    total_chunks = 0
    for i, pdf in enumerate(to_process, 1):
        print(f"[{i}/{len(to_process)}] 🔄 {pdf.name}")
        try:
            n = process_pdf(pdf, converter, chunker, client)
            total_chunks += n
            print(f"       ✅ {n} chunks indexés avec succès")
        except Exception as e:
            print(f"       ❌ Erreur : {e}")

    print("\n✅ Ingestion complétée.")