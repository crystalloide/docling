# check_index.py
from qdrant_client import QdrantClient
from collections import Counter

client = QdrantClient(url="http://localhost:6333")
COLLECTION = "prod_documents"  # adaptez si besoin

sources, offset = Counter(), None
while True:
    results, offset = client.scroll(
        collection_name=COLLECTION,
        with_payload=["source"],
        with_vectors=False,
        limit=1000,
        offset=offset,
    )
    for point in results:
        sources[point.payload.get("source", "?")] += 1
    if offset is None:
        break

print(f"\n{'Fichier':<60} {'Chunks':>8}")
print("-" * 70)
for src, count in sorted(sources.items()):
    print(f"{src:<60} {count:>8}")
print(f"\nTotal : {sum(sources.values())} vecteurs dans {len(sources)} fichiers")