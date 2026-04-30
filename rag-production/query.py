"""
query.py — Boucle RAG interactive
  - Recherche vectorielle dans Qdrant (nomic-embed-text)
  - Génération en streaming avec Ollama (llama3.1:8b ou qwen3:30b)
  - Affiche les sources utilisées pour chaque réponse
"""

import sys
import ollama
from qdrant_client import QdrantClient

from config import (
    QDRANT_URL, COLLECTION,
    EMBED_MODEL, LLM_MODEL, TOP_K
)

# ── Prompt système RAG ─────────────────────────────────────────────────────
# SYSTEM_PROMPT = (
#     "Tu es un assistant expert en analyse de documents. "
#     "Réponds à la question en te basant UNIQUEMENT sur le contexte fourni. "
#     "Si la réponse ne figure pas dans le contexte, dis-le clairement. "
#     "Sois précis et concis."
# )

SYSTEM_PROMPT = (
    "Tu es un assistant expert en analyse documentaire. "
    "Utilise les extraits fournis pour répondre de façon détaillée et structurée. "
    "Tu es également expert en analyse de données. Le contexte contient parfois des tableaux au format Markdown. Analyse attentivement les lignes et colonnes pour répondre. Si les données proviennent d'un tableau, présente ta réponse de manière structurée."
    "Si l'information est partiellement présente, fais une synthèse logique. "
    "Cite toujours le nom du document source dans ta réponse."
)

# ── Initialisation ─────────────────────────────────────────────────────────
def init_client():
    client = QdrantClient(url=QDRANT_URL)
    if not client.collection_exists(COLLECTION):
        print(f"❌ La collection '{COLLECTION}' est introuvable dans Qdrant.")
        print("   Lancez d'abord : python ingest.py")
        sys.exit(1)
    count = client.get_collection(COLLECTION).points_count
    print(f"[Qdrant] Collection '{COLLECTION}' — {count} vecteurs")
    return client


# ── Recherche de chunks pertinents ────────────────────────────────────────
def retrieve(question: str, client: QdrantClient):
    emb = ollama.embed(model=EMBED_MODEL, input=question).embeddings[0]
    response = client.query_points(
        collection_name=COLLECTION,
        query=emb,
        limit=TOP_K,
        with_payload=True,
    )
    return response.points


# ── Génération RAG avec streaming ─────────────────────────────────────────
def generate(question: str, search_results) -> None:
    context_parts = []
    for r in search_results:
        src   = r.payload.get("source", "?")
        chunk = r.payload.get("chunk_id", "?")
        text  = r.payload.get("text", "")
        score = r.score
        context_parts.append(f"[{src} | chunk {chunk} | score {score:.3f}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        f"Contexte extrait des documents :\n\n"
        f"{context}\n\n"
        f"Question : {question}\n\n"
        f"Réponse :"
    )

    # Sources affichées avant la réponse
    print("\n📎 Sources utilisées :")
    for r in search_results:
        print(f"   • {r.payload.get('source')} (chunk {r.payload.get('chunk_id')}"
              f", similarité {r.score:.3f})")
    print()
    print("─" * 60)

    # Streaming de la réponse
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print("\n" + "─" * 60)


# ── Boucle interactive ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Production — Recherche interactive")
    print(f"  LLM      : {LLM_MODEL}")
    print(f"  Embedding: {EMBED_MODEL}  |  Top-K: {TOP_K}")
    print("=" * 60)

    client = init_client()
    print("\n💡 Tapez votre question. Commandes : 'quit' pour quitter, 'stats' pour les infos.\n")

    while True:
        try:
            question = input("❓ Question : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break

        if question.lower() == "stats":
            count = client.get_collection(COLLECTION).points_count
            print(f"   Collection '{COLLECTION}' : {count} vecteurs\n")
            continue

        print("\n🔎 Recherche des passages pertinents...")
        results = retrieve(question, client)

        if not results:
            print("⚠ Aucun document trouvé. Vérifiez que des PDFs ont été ingérés.\n")
            continue

        print(f"\n🤖 Génération (streaming — {LLM_MODEL})...")
        generate(question, results)
        print()
