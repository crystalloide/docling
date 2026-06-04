# clear_qdrant.py
from qdrant_client import QdrantClient
from config import QDRANT_URL, COLLECTION

client = QdrantClient(url=QDRANT_URL)

if client.collection_exists(COLLECTION):
    print(f"⚠️ Suppression de la collection '{COLLECTION}' en cours...")
    client.delete_collection(collection_name=COLLECTION)
    print(f"✅ La collection '{COLLECTION}' a été vidée et supprimée avec succès.")
else:
    print(f"ℹ️ La collection '{COLLECTION}' n'existe pas ou est déjà vide.")