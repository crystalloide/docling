
guide = r"""# Mode Opératoire Complet — Stack RAG Production

> **Environnement** : Windows 11 · 64 Go RAM · 16 cœurs · GPU NVIDIA 24 Go VRAM · Ollama installé · Docker Desktop installé

---

## Architecture

```
PDFs (data/pdfs/)
      |
      v
 [Docling GPU]  -->  Markdown structuré
      |
      v
 [RecursiveCharacterTextSplitter]  -->  chunks (512 tokens, overlap 64)
      |
      v
 [Ollama : nomic-embed-text]  -->  vecteurs 768 dims
      |
      v
 [Qdrant Docker]  -->  collection HNSW/COSINE persistante
      |
      v (recherche Top-K)
 [Ollama : llama3.1:8b]  -->  réponse streaming
```

| Composant | Outil | Rôle |
|---|---|---|
| Parsing PDF | **Docling** (IBM) | Extraction layout-aware, tableaux, OCR GPU |
| Chunking | **RecursiveCharacterTextSplitter** | Découpage sémantique sur séparateurs Markdown |
| Embedding | **nomic-embed-text** via Ollama | Vecteurs 768 dims, multilingue |
| Base vectorielle | **Qdrant** (Docker) | Stockage HNSW, filtrage, scalable en cluster |
| LLM | **llama3.1:8b** via Ollama | 128k contexte, ~169 tok/s sur 24 Go VRAM |
| Config | **.env + config.py** | Centralisation, zéro modification des scripts |

---

## Structure du projet

```
C:\rag-production\
├── docker-compose.yml      <- Qdrant container (pret a l'emploi)
├── .env                    <- Seul fichier de configuration
├── config.py               <- Chargement .env (ne pas modifier)
├── requirements.txt        <- Dependances Python
├── ingest.py               <- Ingestion des PDFs
├── query.py                <- Recherche RAG interactive
└── data\
    ├── pdfs\               <- DEPOSEZ VOS PDFs ICI
    └── qdrant_storage\     <- Donnees Qdrant (cree automatiquement)
```

---

## Etape 1 — Verifier les prerequis

### 1.1 Ollama + GPU

Ouvrez un **terminal PowerShell** :

```powershell
ollama --version
```

Testez que le GPU est utilise :

```powershell
ollama run llama3.1:8b "Reponds juste : GPU OK"
# La premiere fois : telecharge le modele (~4.7 Go)
```

### 1.2 Docker Desktop + GPU NVIDIA

```powershell
docker version

# Tester le passthrough GPU NVIDIA dans Docker
docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi
```

> Si `nvidia-smi` retourne les infos de votre GPU : OK.
> Sinon : Docker Desktop -> Settings -> Resources -> WSL Integration -> activer Ubuntu.

### 1.3 Python 3.11 ou 3.12

```powershell
python --version
# Requis : Python 3.11 ou 3.12
```

---

## Etape 2 — Telecharger les modeles Ollama

```powershell
# Modele LLM principal (environ 4.7 Go)
ollama pull llama3.1:8b

# Modele d'embedding (environ 270 Mo)
ollama pull nomic-embed-text

# Verifier
ollama list
```

> **Alternative LLM** : pour une meilleure qualite de raisonnement (mais plus lent),
> modifiez dans .env : `LLM_MODEL=qwen3:30b` puis `ollama pull qwen3:30b` (~18 Go).

---

## Etape 3 — Creer le projet et l'environnement Python

### 3.1 Creer la structure

```powershell
mkdir C:\rag-production
cd C:\rag-production
mkdir data\pdfs
mkdir data\qdrant_storage
```

Copiez tous les fichiers fournis dans `C:\rag-production\` :
`docker-compose.yml`, `.env`, `config.py`, `requirements.txt`, `ingest.py`, `query.py`

### 3.2 Environnement virtuel Python

```powershell
cd C:\rag-production

python -m venv .venv

.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

> **Si PowerShell refuse l'activation** (erreur de politique d'execution) :
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Relancez ensuite `.venv\Scripts\Activate.ps1`

### 3.3 Installer PyTorch avec support CUDA

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verification :

```powershell
python -c "import torch; print('CUDA disponible :', torch.cuda.is_available())"
# Attendu : CUDA disponible : True
```

> Si False : verifiez que vos drivers NVIDIA sont a jour (>=525).
> Telechargez depuis https://www.nvidia.com/drivers

### 3.4 Installer les dependances du projet

```powershell
pip install -r requirements.txt
```

> **Si Docling echoue** (erreur de compilation C++) :
> Installez Visual Studio Build Tools -> composant "Desktop development with C++"
> https://visualstudio.microsoft.com/visual-cpp-build-tools/
> Puis relancez `pip install -r requirements.txt`

---

## Etape 4 — Demarrer Qdrant

```powershell
cd C:\rag-production

docker compose up -d

docker compose ps
```

Attendu :

```
NAME      STATUS
qdrant    Up X seconds (healthy)
```

Interface Web Qdrant : **http://localhost:6333/dashboard**

Arret propre si necessaire : `docker compose down`

---

## Etape 5 — Ingerer vos PDFs

1. Copiez vos fichiers PDF dans `C:\rag-production\data\pdfs\`
2. Lancez l'ingestion :

```powershell
cd C:\rag-production
.venv\Scripts\Activate.ps1
python ingest.py
```

**Sortie attendue :**

```
============================================================
  RAG Production -- Ingestion de PDFs
============================================================
[Docling] OK GPU CUDA active
[Qdrant] Collection 'prod_documents' creee (768 dims, COSINE)

3 PDF(s) trouves
   2 deja indexes, 1 a traiter

[1/1] rapport_2024.pdf
       OK 87 vecteurs indexes

============================================================
OK Ingestion terminee
   Fichiers traites : 1/1
   Vecteurs inseres : 87
   Total en base    : 87 vecteurs
============================================================
```

> **Reprise automatique** : relancer `python ingest.py` ne retraite pas les fichiers
> deja indexes. Seuls les nouveaux PDFs sont traites.

---

## Etape 6 — Interroger vos documents

```powershell
python query.py
```

**Exemple d'interaction :**

```
============================================================
  RAG Production -- Recherche interactive
  LLM      : llama3.1:8b
  Embedding: nomic-embed-text  |  Top-K: 5
============================================================
[Qdrant] Collection 'prod_documents' -- 87 vecteurs

Tapez votre question. Commandes : 'quit' pour quitter, 'stats' pour les infos.

Question : Quelles sont les conclusions du rapport 2024 ?

Recherche des passages pertinents...

Sources utilisees :
   rapport_2024.pdf (chunk 12, similarite 0.891)
   rapport_2024.pdf (chunk 13, similarite 0.874)
   rapport_2024.pdf (chunk 8,  similarite 0.831)

Generation (streaming -- llama3.1:8b)...
------------------------------------------------------------
D'apres le rapport 2024, les principales conclusions sont...
------------------------------------------------------------

Question : stats
   Collection 'prod_documents' : 87 vecteurs

Question : quit
Au revoir !
```

---

## Etape 7 — Ajout continu de PDFs (usage quotidien)

Deposez les nouveaux fichiers dans `data\pdfs\` et relancez simplement :

```powershell
python ingest.py
```

Les fichiers deja indexes sont detectes et ignores. L'ingestion est incrementale et idempotente.

---

## Parametrage (fichier .env uniquement)

Le fichier `.env` est le **seul point de configuration**. Aucun script a modifier.

| Variable | Defaut | Description |
|---|---|---|
| `LLM_MODEL` | `llama3.1:8b` | Remplacer par `qwen3:30b` pour plus de qualite |
| `EMBED_MODEL` | `nomic-embed-text` | Remplacer par `bge-m3` + `EMBED_DIM=1024` |
| `CHUNK_SIZE` | `512` | Augmenter pour des documents tres structures |
| `CHUNK_OVERLAP` | `64` | Augmenter si les reponses manquent de contexte |
| `TOP_K` | `5` | Augmenter pour plus de contexte (ralentit la generation) |
| `PDF_DIR` | `./data/pdfs` | Changer pour pointer vers un autre repertoire |

> **Attention** : changer `EMBED_MODEL` ou `EMBED_DIM` necessite de vider la collection
> et de reingerer tous les PDFs. Supprimez `data/qdrant_storage/` et relancez `ingest.py`.

---

## Resolution des problemes

| Symptome | Cause probable | Solution |
|---|---|---|
| `CUDA disponible : False` | PyTorch CPU-only | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `Connection refused :6333` | Qdrant non demarre | `docker compose up -d` |
| `model 'llama3.1:8b' not found` | Modele non telecharge | `ollama pull llama3.1:8b` |
| Docling tres lent | GPU non detecte | Verifier etape 3.3 (PyTorch CUDA) |
| Doublons dans Qdrant | Ancien script utilise | Utiliser `ingest.py` fourni (IDs UUID5 deterministes) |
| Set-ExecutionPolicy refuse | Droits insuffisants | Lancer PowerShell en Administrateur |
| Erreur compilation pip | Build Tools manquants | Installer Visual Studio Build Tools C++ |
| `qdrant_storage` vide apres redemarrage | Volume Docker manquant | Verifier que le dossier `data/qdrant_storage/` existe avant `docker compose up` |

---

## Scalabilite vers un cluster Qdrant (optionnel)

Pour passer a un deploiement multi-noeuds sans modifier `ingest.py` ni `query.py`,
remplacez simplement `QDRANT_URL` dans `.env` par l'URL du load balancer du cluster.
Le code Python est identique : Qdrant gere le sharding et la replication de facon transparente.
"""

with open('/home/user/rag-production/RAG_Guide_Complet.md', 'w', encoding='utf-8') as f:
    f.write(guide)
print("Guide OK")
