"""
query.py — Boucle RAG interactive
  - Recherche vectorielle dans Qdrant (nomic-embed-text)
  - Génération en streaming avec Ollama
  - Affiche les sources utilisées pour chaque réponse
  - Filtre par source avec la syntaxe : @"Code de la mutualité.pdf" question
  - Commande 'sources' pour lister les fichiers indexés
"""

import sys
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from collections import Counter

from config import (
    QDRANT_URL, COLLECTION,
    EMBED_MODEL, LLM_MODEL, TOP_K
)

SYSTEM_PROMPT = (
    "Tu es un assistant expert en analyse documentaire. "
    "Utilise les extraits fournis pour répondre de façon détaillée et structurée. "
    "Tu es également expert en analyse de données. Le contexte contient parfois des tableaux "
    "au format Markdown. Analyse attentivement les lignes et colonnes pour répondre. "
    "Si les données proviennent d'un tableau, présente ta réponse de manière structurée. "
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


# ── Liste des fichiers indexés ─────────────────────────────────────────────
def list_sources(client: QdrantClient):
    sources, offset = Counter(), None
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION,
            with_payload=["source"],
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        for p in results:
            sources[p.payload.get("source", "?")] += 1
        if offset is None:
            break
    print(f"\n📚 {len(sources)} fichiers indexés :")
    for i, (src, n) in enumerate(sorted(sources.items()), 1):
        print(f"   {i:>2}. {src}  ({n} chunks)")
    print()


# ── Parsing de la question : détecte @"fichier.pdf" ou @fichier.pdf ────────
def parse_question(raw: str):
    """
    Syntaxes acceptées :
      @"Code de la mutualité.pdf" quelles sont les règles ?
      @Code_civil.pdf question
    Retourne (source_filter, question_nettoyée)
    """
    raw = raw.strip()
    if not raw.startswith("@"):
        return None, raw

    rest = raw[1:].strip()

    # Guillemets : @"nom avec espaces.pdf" question
    if rest.startswith('"'):
        end = rest.find('"', 1)
        if end != -1:
            source = rest[1:end]
            question = rest[end + 1:].strip()
            return source, question

    # Sans guillemets : le nom de fichier se termine au premier espace
    # suivi d'un mot non-.pdf  — on prend tout jusqu'à .pdf
    import re
    m = re.match(r'([^\s].*?\.pdf)\s+(.*)', rest, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1), m.group(2).strip()

    # Fallback : tout est le nom du fichier, question vide
    return rest, ""


# ── Recherche de chunks pertinents ────────────────────────────────────────
def retrieve(question: str, client: QdrantClient, source_filter: str = None):
    emb = ollama.embed(model=EMBED_MODEL, input=question).embeddings[0]

    query_filter = None
    if source_filter:
        query_filter = Filter(
            must=[FieldCondition(
                key="source",
                match=MatchValue(value=source_filter)
            )]
        )

    response = client.query_points(
        collection_name=COLLECTION,
        query=emb,
        limit=TOP_K,
        with_payload=True,
        query_filter=query_filter,
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

    print("\n📎 Sources utilisées :")
    for r in search_results:
        print(f"   • {r.payload.get('source')} "
              f"(chunk {r.payload.get('chunk_id')}, similarité {r.score:.3f})")
    print()
    print("─" * 60)

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
HELP = """
  Commandes disponibles :
    sources          → liste les fichiers indexés avec leur nombre de chunks
    stats            → nombre total de vecteurs en base
    quit / q         → quitter

  Filtre par fichier (évite les confusions entre codes) :
    @"Code de la mutualité.pdf" quelles sont les règles ?
    @Code_civil.pdf  qu'est-ce que la responsabilité civile ?
"""

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Production — Recherche interactive")
    print(f"  LLM      : {LLM_MODEL}")
    print(f"  Embedding: {EMBED_MODEL}  |  Top-K: {TOP_K}")
    print("=" * 60)

    client = init_client()
    print(HELP)

    while True:
        try:
            raw = input("❓ Question : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not raw:
            continue

        if raw.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break

        if raw.lower() == "stats":
            count = client.get_collection(COLLECTION).points_count
            print(f"   Collection '{COLLECTION}' : {count} vecteurs\n")
            continue

        if raw.lower() == "sources":
            list_sources(client)
            continue

        if raw.lower() in ("help", "aide", "?"):
            print(HELP)
            continue

        # Parsing filtre source
        source_filter, question = parse_question(raw)

        if not question:
            print("⚠ Question vide après le filtre source.\n")
            continue

        if source_filter:
            print(f"🔍 Filtre actif : {source_filter}")

        print("\n🔎 Recherche des passages pertinents...")
        results = retrieve(question, client, source_filter=source_filter)

        if not results:
            if source_filter:
                print(f"⚠ Aucun résultat pour '{source_filter}'.")
                print("   → Vérifiez le nom exact avec la commande : sources\n")
            else:
                print("⚠ Aucun document trouvé.\n")
            continue

        print(f"\n🤖 Génération (streaming — {LLM_MODEL})...")
        generate(question, results)
        print()
