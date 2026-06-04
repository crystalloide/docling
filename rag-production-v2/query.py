"""
query.py — Boucle RAG interactive avec LLM Router
  - Détermination automatique des domaines et types cibles par le LLM
  - Filtrage à chaud dans Qdrant pour éviter les noyades de contexte
  - Support de l'ancien filtre forcé syntaxe : @"Fichier.pdf" question
"""

import sys
import json
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from collections import Counter

from config import (
    QDRANT_URL, COLLECTION,
    EMBED_MODEL, LLM_MODEL, TOP_K
)

SYSTEM_PROMPT = (
    "Tu es un assistant expert en analyse documentaire multi-domaine.\n"
    "Chaque extrait de texte fourni est encadré par des balises indiquant explicitement sa SOURCE exacte "
    "(ex: --- DEBUT DU BLOC (SOURCE: nom_du_fichier.pdf) ---).\n"
    "CONSIGNE STRICTE D'ATTRIBUTION :\n"
    "Lorsque tu affirmes qu'une information provient d'un document, tu dois IMPÉRATIVEMENT utiliser uniquement "
    "le nom de fichier indiqué dans la balise 'SOURCE' correspondante à cet extrait.\n"
    "Ne te fie pas aux dates ou aux prénoms cités à l'intérieur du texte pour deviner le nom du fichier. "
    "Si le document trouvé ne mentionne pas explicitement le nom ou le prénom de la personne demandée par l'utilisateur, réponds clairement que tu ne trouves pas de document au nom de cette personne, au lieu d'attribuer les données d'un autre profil.\n"
    "CONSIGNE DE COMPTAGE : Si l'utilisateur te demande de compter les documents ou les CV, base-toi UNIQUEMENT sur le nombre de noms de fichiers différents présents dans les balises SOURCE du contexte fourni, et non sur le nombre de blocs de texte.\n"
    "Sois factuel, précis, et cite tes sources."
)

def init_client():
    client = QdrantClient(url=QDRANT_URL)
    if not client.collection_exists(COLLECTION):
        print(f"❌ La collection '{COLLECTION}' est introuvable.")
        sys.exit(1)
    return client

def list_sources(client: QdrantClient):
    sources, domains = Counter(), Counter()
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION,
            with_payload=["source", "domain"],
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        for p in results:
            sources[p.payload.get("source", "?")] += 1
            domains[p.payload.get("domain", "Inconnu")] += 1
        if offset is None:
            break
            
    print(f"\n📂 Domaines identifiés en base :")
    for dom, n in domains.items():
        print(f"   • {dom:<25} (~{n} chunks)")
        
    print(f"\n📚 Fichiers indexés :")
    for i, (src, n) in enumerate(sorted(sources.items()), 1):
        print(f"   {i:>2}. {src:<50} ({n} chunks)")
    print()

def parse_question(raw: str):
    raw = raw.strip()
    if not raw.startswith("@"):
        return None, raw
    rest = raw[1:].strip()
    if rest.startswith('"'):
        end = rest.find('"', 1)
        if end != -1:
            return rest[1:end], rest[end + 1:].strip()
    import re
    m = re.match(r'([^\s].*?\.pdf)\s+(.*)', rest, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1), m.group(2).strip()
    return rest, ""

# ── NOUVEAUTÉ : Le LLM Router ──────────────────────────────────────────────
def route_query_intent(question: str, client: QdrantClient) -> dict:
    """
    Routeur bi-critère : Filtre dynamiquement par Domaine et/ou par Type de document.
    """
    # 1. Récupération des taxonomies réelles en base
    results, _ = client.scroll(collection_name=COLLECTION, with_payload=["domain", "document_type"], limit=500)
    existing_domains = list(set([p.payload.get("domain") for p in results if p.payload.get("domain")]))
    existing_types = list(set([p.payload.get("document_type") for p in results if p.payload.get("document_type")]))

    # 2. Prompt à double niveau
    prompt = (
        f"Tu es le routeur d'une base de connaissances d'entreprise.\n"
        f"Voici les DOMAINES disponibles : {existing_domains}\n"
        f"Voici les TYPES DE DOCUMENTS disponibles : {existing_types}\n\n"
        f"Analyse la question de l'utilisateur : '{question}'\n\n"
        f"Instructions :\n"
        f"- Si la question parle d'un type de document précis (ex: 'les CV', 'le code', 'un rapport', 'combien de CV'), "
        f"remplis la clé 'target_types' avec les types correspondants.\n"
        f"- Si la question cible un domaine métier (ex: 'la finance', 'le juridique'), "
        f"remplis la clé 'target_domains'.\n"
        f"- Si la question est globale, quantitative, ou nécessite de chercher partout (ex: 'combien de documents', 'liste les fichiers'), "
        f"laisse les listes vides [] pour désactiver le filtrage.\n\n"
        f"Génère un JSON STRICT :\n"
        f"{{\n"
        f"  \"target_domains\": [],\n"
        f"  \"target_types\": []\n"
        f"}}\n"
        f"Répond uniquement avec le JSON."
    )

    try:
        res = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.0})
        content = res["message"]["content"].strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"): content = content[4:]
        return json.loads(content.strip())
    except Exception:
        return {"target_domains": [], "target_types": []}
        
# ── Recherche de chunks filtrés dynamiquement ──────────────────────────────
def retrieve(question: str, client: QdrantClient, source_filter: str = None):
    emb = ollama.embed(model=EMBED_MODEL, input=question).embeddings[0]

    must_conditions = []
    should_conditions = []
    
    # 1. Cas du filtre manuel forcé via @ (Reste un ET strict)
    if source_filter:
        must_conditions.append(FieldCondition(key="source", match=MatchValue(value=source_filter)))
    
    # 2. Cas du filtrage automatique par le LLM Router (On passe en OU / should)
    else:
        intent = route_query_intent(question, client)
        target_domains = intent.get("target_domains", [])
        target_types = intent.get("target_types", [])
        
        if target_domains:
            print(f"🤖 [Router] Ciblage sémantique : Filtre appliqué sur les domaines {target_domains}")
            should_conditions.append(FieldCondition(key="domain", match=MatchAny(any=target_domains)))
            
        if target_types:
            print(f"🤖 [Router] Ciblage structurel : Filtre appliqué sur les types {target_types}")
            should_conditions.append(FieldCondition(key="document_type", match=MatchAny(any=target_types)))

    # Construction du filtre combiné intelligent
    # Si on a du automatique, il suffit de remplir une des conditions (should)
    # Si on a du forcé (@), il doit obligatoirement matcher (must)
    query_filter = Filter(
        must=must_conditions if must_conditions else None,
        should=should_conditions if should_conditions else None
    ) if (must_conditions or should_conditions) else None

    response = client.query_points(
        collection_name=COLLECTION,
        query=emb,
        limit=TOP_K,
        with_payload=True,
        query_filter=query_filter,
    )
    return response.points

def generate(question: str, search_results) -> None:
    context_parts = []
    for r in search_results:
        text = r.payload.get("text", "")
        context_parts.append(text)

    context = "\n\n=========\n\n".join(context_parts)
    prompt = f"Contextes documentaires fournis :\n\n{context}\n\nQuestion : {question}\n\nRéponse :"
    
    print("\n📎 Sources consultées :")
    seen_sources = set()
    for r in search_results:
        src_info = f"{r.payload.get('source')} ({r.payload.get('domain')} -> {r.payload.get('document_type')})"
        if src_info not in seen_sources:
            print(f"   • {src_info}")
            seen_sources.add(src_info)
    print()
    print("─" * 60)

    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        options={
            "temperature": 0.3,
            "num_predict": 2048
        },
        stream=True,
    )
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print("\n" + "─" * 60)

HELP = """
  Commandes disponibles :
    sources          → Liste les fichiers, les domaines détectés et les statistiques de chunks
    stats            → Nombre total de vecteurs en base
    quit / q         → Quitter
"""

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Multi-Domaine Intelligent")
    print(f"  Modèle : {LLM_MODEL}  |  Top-K Contextuel : {TOP_K}")
    print("=" * 60)

    client = init_client()
    print(HELP)

    while True:
        try:
            raw = input("❓ Question : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not raw or raw.lower() in ("quit", "exit", "q"): break
        if raw.lower() == "stats":
            print(f"   Total base : {client.get_collection(COLLECTION).points_count} vecteurs\n")
            continue
        if raw.lower() == "sources":
            list_sources(client)
            continue

        source_filter, question = parse_question(raw)
        if not question: continue

        if source_filter:
            print(f"🔍 Filtre source explicite : {source_filter}")

        print("🔎 Recherche sémantique en cours...")
        results = retrieve(question, client, source_filter=source_filter)

        if not results:
            print("⚠ Aucun document pertinent trouvé pour cette requête.\n")
            continue

        print(f"🤖 Génération de la réponse...")
        generate(question, results)
        print()