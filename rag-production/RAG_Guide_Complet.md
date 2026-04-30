# Mode Opératoire Complet — Stack RAG Production

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
 [Ollama : qwen3.5:9b]  -->  réponse streaming
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

### 1.0 Docker Desktop 

Lancer **Docker Desktop** et vérifier que des conteneurs sans lien avec le projet ne sont pas déjà en train de s'exécuter:

### 1.1 Ollama + GPU

Ouvrez un **terminal PowerShell** :

```powershell
ollama --version
```

Testez que le GPU est utilise :

```powershell
# Voir ici les modèles et tailles disponibles : https://ollama.com/library/qwen3.5

#ollama run llama3.1:8b "Reponds juste : GPU OK"
# La premiere fois : telecharge le modele (~4.7 Go)

# On choisit celui-ci pour commencer : 
ollama run qwen3.5:9b "Reponds juste : GPU OK"
# La premiere fois : telecharge le modele (~6,6 Go)

#ollama run qwen3.5:27b "Reponds juste : GPU OK"
# La premiere fois : telecharge le modele (~17 Go)
```

### 1.2 Docker Desktop + GPU NVIDIA

```powershell
docker version

# Tester le passthrough GPU NVIDIA dans Docker
docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi
```

> Si `nvidia-smi` retourne les infos de votre GPU : OK.
> Sinon : Docker Desktop -> Settings -> Resources -> WSL Integration -> activer Ubuntu.

##### Exemple ok :   
```text
Thu Apr 30 08:26:47 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        On  |   00000000:0C:00.0  On |                  N/A |
|  0%   48C    P2            110W /  370W |    1587MiB /  24576MiB |      5%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

### 1.3 Python 3.11 ou 3.12 ou 3.13

```powershell
python --version
# Requis : Python 3.11 ou 3.12 ou 3.13
```

---

## Etape 2 — Telecharger les modeles Ollama

```powershell
# Modele LLM principal (environ 4.7 Go)
#ollama pull llama3.1:8b

# Modele LLM principal (environ 6.6 Go)
ollama pull qwen3.5:9b

# Modele LLM principal (environ 17 Go)
#ollama pull qwen3.5:27b

# Modele d'embedding (environ 270 Mo)
ollama pull nomic-embed-text

# Verifier
ollama list
```

> **Alternative LLM** : pour une meilleure qualite de raisonnement (mais plus lent),
> modifiez dans .env : `LLM_MODEL=qwen3.5:27b` puis `ollama pull qwen3.5:27b` (~17 Go).

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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
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

## Etape 5 — Ingérer vos PDFs

1. Copiez vos fichiers PDF dans `C:\rag-production\data\pdfs\`
2. Lancez l'ingestion :

```powershell
cd C:\rag-production
.venv\Scripts\Activate.ps1
python -u ingest.py
```

**Sortie attendue :**

```text
============================================================
  RAG Production -- Ingestion de PDFs
============================================================
[Ollama] ✅ Connecté — modèle embedding : nomic-embed-text
[Docling] GPU : NVIDIA GeForce RTX 3090 (24.0 Go VRAM)
[Docling] ✅ GPU CUDA activé (OCR activé)
[Qdrant] Collection 'prod_documents' créée (768 dims, COSINE)

📂 8 PDF(s) trouvé(s)
   ↳ 0 déjà indexé(s), 8 à traiter

[1/8] 🔄 Code de déontologie des architectes.pdf
       ℹ  23 pages → Docling GPU direct
Loading weights: 100%|█████████████████████████████████████████████████████████████| 770/770 [00:00<00:00, 1250.46it/s]
       ✅ 59 vecteurs indexés
[2/8] 🔄 Code de l'aviation civile.pdf
       ℹ  35 pages → Docling GPU direct
       ✅ 88 vecteurs indexés
[3/8] 🔄 Code de la famille et de l'aide sociale.pdf
       ℹ  12 pages → Docling GPU direct
       ✅ 24 vecteurs indexés
[4/8] 🔄 Code des instruments monétaires et des médailles.pdf
       ℹ  9 pages → Docling GPU direct
       ✅ 16 vecteurs indexés
[5/8] 🔄 Code des pensions de retraite des marins français du commerce, de pêche ou de plaisance.pdf
       ℹ  18 pages → Docling GPU direct
       ✅ 40 vecteurs indexés
[6/8] 🔄 Code disciplinaire et pénal de la marine marchande.pdf
       ℹ  10 pages → Docling GPU direct
       ✅ 16 vecteurs indexés
[7/8] 🔄 Code du domaine de l'Etat et des collectivités publiques applicable à la collectivité territoriale de Mayotte.pdf
       ℹ  8 pages → Docling GPU direct
       ✅ 14 vecteurs indexés
[8/8] 🔄 Code du domaine public fluvial et de la navigation intérieure.pdf
       ℹ  8 pages → Docling GPU direct
       ✅ 9 vecteurs indexés


============================================================
✅ Ingestion terminée
   Fichiers traités : 8/8
   Vecteurs insérés : 266
   Total en base    : 266 vecteurs
============================================================
```

```powershell
Invoke-RestMethod http://localhost:6333/collections/prod_documents
```

Et lisez directement le résultat affiché — cherchez points_count ou indexed_vectors_count dans l'output.

```text
Ou encore plus simple, directement dans le navigateur : http://localhost:6333/collections/prod_documents 
— le JSON s'affiche proprement et vous voyez tous les champs disponibles d'un coup.
```

```text
les fichiers qdrant sont stockés dans le dossier que vous avez monté comme volume Docker :

C:\rag-production\data\qdrant_storage\
```

```bash
docker logs qdrant
```


### ✅ Option 1 pour suivre l'activité sur la carte GPU : 

##### Pour un affichage continu et lisible :
```bash
powershellnvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu --format=csv -l 1
```

💡 Fonctionne aussi depuis l'intérieur d'un conteneur Docker si vous passez --gpus all.


### ✅ Option 2 — nvitop (recommandé pour le dev)

##### htop pour GPU, très lisible dans le terminal :
```bash
pip install nvitop
nvitop
```




> **Reprise automatique** : relancer `python ingest.py` ne retraite pas les fichiers
> deja indexes. Seuls les nouveaux PDFs sont traites.

---

## Etape 6 — Interroger vos documents

```powershell
cd C:\rag-production
.venv\Scripts\Activate.ps1
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
| `LLM_MODEL` | `llama3.1:8b` | Remplacer par `qwen3.5:27b` pour plus de qualite |
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
