# Mode opératoire complet — Stack RAG Production

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
	  
 [Ollama : qwen3.5:9b]   		-->  réponse streaming
 [Ollama : qwen2.5:7b]  		-->  réponse streaming
 [Ollama : llama3.1:8b]  		-->  réponse streaming
 [Ollama : mistral-nemo:12b]  	-->  réponse streaming
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


##### Clonage du projet : 
```sh
cd ~
sudo rm -Rf docling

git clone https://github.com/crystalloide/docling

cd docling

```
    
### Démarrage


#### Lancer l'environnement qdrant
```bash
docker compose -f docker-compose.yml up -d
```



## Comparatif des Modèles Équivalents à Llama 3.1 8B

Ce document présente une analyse comparative des principaux Large Language Models (LLM) de taille moyenne qui rivalisent avec le Llama 3.1 8B de Meta.

## Tableau récapitulatif :


| Modèle			| Paramètres 	| Fenêtre de contexte 	| Spécialité					|
|---|---|---|-----|
| **Llama 3.1 8B**	| 8B 			| 128k					| Polyvalence & Écosystème 		|
| **Gemma 2 9B** 	| 9B 			| 8k 					| Logique & Nuances 			|
| **Mistral NeMo** 	| 12B 			| 128k					| Français & Multilingue 		|
| **Qwen 2.5 7B** 	| 7B 			| 128k					| Code & Mathématiques 			|
| **Phi-3.5 Mini** 	| 3.8B			| 128k 					| Vitesse & Appareils mobiles	|


---

## Analyse par modèle LLM :

### 1. Llama 3.1 8B (Meta)
Le standard actuel. Il bénéficie de l'écosystème le plus vaste (compatible avec presque tous les outils comme Ollama, LM Studio, etc.). Sa fenêtre de contexte de 128k est un atout majeur pour traiter de longs documents.

### 2. Gemma 2 9B (Google)
Basé sur la technologie des modèles Gemini. Il excelle dans le raisonnement pur et offre une qualité de rédaction souvent jugée supérieure (moins de répétitions, style plus fluide). Son point faible reste sa fenêtre de contexte plus limitée (8k).

### 3. Mistral NeMo 12B (Mistral AI & NVIDIA)
Fruit d'une collaboration entre le leader européen et le géant des puces. Avec ses 12 milliards de paramètres, il est plus "intelligent" sur les nuances linguistiques, particulièrement en français, et gère parfaitement les contextes longs.

### 4. Qwen 2.5 7B (Alibaba)
C'est le modèle qui domine les benchmarks techniques. Si votre besoin concerne la génération de code (Python, C++, etc.) ou la résolution de problèmes mathématiques complexes, c'est l'alternative la plus robuste.

### 5. Phi-3.5 Mini (Microsoft)
La preuve que la taille ne fait pas tout. Ce modèle est extrêmement optimisé. Il est idéal pour être déployé localement sur des machines avec peu de RAM ou pour des applications nécessitant une latence minimale tout en conservant un bon niveau de compréhension.


#### Voir ici les modèles et tailles disponibles : https://ollama.com/library/qwen2.5


### 1.0 Docker Desktop 

Lancer **Docker Desktop** et vérifier que des conteneurs sans lien avec le projet ne sont pas déjà en train de s'exécuter:

### 1.1 Ollama + GPU

Ouvrez un **terminal PowerShell** :

## Étape 1 — Démarrer Ollama

Lance Ollama via l'application Windows (icône dans la barre des tâches) ou via PowerShell :

```powershell
ollama serve
```

Attends les messages de démarrage, notamment la ligne confirmant la détection du GPU.
```powershell
ollama --version
```

```text
# Affichage attendu en retour : 

Warning: could not connect to a running Ollama instance
Warning: client version is 0.21.2
```

##### Testez que le GPU est utilisé :


## Étape 2 — Vérifier le GPU avec `ollama ps`

C'est la méthode la plus directe. Pendant qu'un modèle tourne, exécute dans un **second terminal** :[^2]

```powershell
ollama ps
```

La colonne `PROCESSOR` indique clairement l'utilisation  :[^2]


| Valeur `PROCESSOR` | Signification |
| :-- | :-- |
| `100% GPU` | Modèle entièrement en VRAM |
| `100% CPU` | Pas de GPU utilisé |
| `48% GPU / 52% CPU` | Split VRAM + RAM système |

## Étape 3 — Vérifier avec `nvidia-smi`

Pour confirmer avec ton GPU NVIDIA (RTX avec 24 Go de VRAM), surveille la charge GPU en temps réel pendant une inférence  :[^3]

```powershell
# Affichage en continu toutes les 500ms
nvidia-smi dmon -s u -d 0.5
```

Ou en mode watch :

```powershell
watch -n 1 nvidia-smi
```

La colonne `GPU-Util` doit monter à plusieurs dizaines de % si le GPU est actif.

## Étape 4 — Lire les logs Ollama

Les logs indiquent explicitement si un GPU compatible est détecté. Sur Windows, les logs sont accessibles via :[^4]

```powershell
# Chemin typique des logs Ollama sur Windows
Get-Content "$env:LOCALAPPDATA\Ollama\server.log" -Tail 50
```

Cherche des lignes comme `CUDA detected` ou `no compatible GPUs were discovered`.
### On choisit celui-ci pour commencer : 

```powershell
ollama run mistral-nemo:12b "Reponds juste : 'GPU OK'"
```

### La premiere fois : telecharge le modele (~7,1 Go)

```powershell
ollama run llama3.1:8b "Reponds juste : GPU OK"
```

### La premiere fois : telecharge le modele (~4.7 Go)

### Possibilité non retenue ici mais à tester si vous voulez  : 
ollama run qwen2.5:7b "Reponds juste : GPU OK"
### La premiere fois : telecharge le modele (~4,7 Go)

### Voir ici les modèles et tailles disponibles : https://ollama.com/library/qwen3.5
#ollama run qwen3.5:9b "Reponds juste : GPU OK"
### La premiere fois : telecharge le modele (~6,6 Go)

#ollama run qwen3.5:27b "Reponds juste : GPU OK"
### La premiere fois : telecharge le modele (~17 Go)
```

#### 1.2 Docker Desktop + GPU NVIDIA

```powershell
docker version
```

#### Tester le passthrough GPU NVIDIA dans Docker
```powershell
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
(.venv) PS C:\rag-production> docker ps -a
CONTAINER ID   IMAGE                                COMMAND             CREATED       STATUS                     PORTS                                                             NAMES
a0762db01b6e   qdrant/qdrant:latest                 "./entrypoint.sh"   3 hours ago   Up 3 min                 0.0.0.0:6333-6334->6333-6334/tcp, [::]:6333-6334->6333-6334/tcp   qdrant
```

Interface Web Qdrant : **http://localhost:6333/dashboard**
<img width="2541" height="673" alt="image" src="https://github.com/user-attachments/assets/ee61b0fe-786e-483a-9594-20b6c6c9b967" />

Arret propre si necessaire : `docker compose down`

---

## Etape 5 — Ingérer vos PDFs

1. Copiez vos fichiers PDF dans `C:\rag-production\data\pdfs\`

<img width="871" height="386" alt="image" src="https://github.com/user-attachments/assets/19784bbb-8259-458d-84e7-afc3fff15fd3" />

   
2. Lancez l'ingestion :

```powershell
cd C:\rag-production
.venv\Scripts\Activate.ps1
python -u ingest.py
```

##### Important : ne tenez pas compte du message suivant : (la gestion en interne est faite correctement, il y a redécoupage en chunk)

```powershell
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (12114 > 8192). Running this sequence through the model will result in indexing errors
```

**Sortie attendue :**

```text
============================================================
  RAG Production — Ingestion de PDFs
============================================================
[Ollama] ✅ Connecté — modèle embedding : nomic-embed-text
[Docling] GPU : NVIDIA GeForce RTX 3090 (24.0 Go VRAM)
[Docling] ✅ GPU CUDA activé (OCR activé)
[Chunker] ✅ HybridChunker — tokenizer nomic-embed-text, max 384 tokens/chunk
[Qdrant] Collection 'prod_documents' existante — 16630 vecteurs

📂 12 PDF(s) trouvé(s)
   ↳ 5 déjà indexé(s), 7 à traiter

[1/7] 🔄 Code de la mutualité.pdf
       ℹ  210 pages → 5 tranches de 50 (Docling GPU + HybridChunker)
  Tranches (210 pages):   0%|                                                                                                                                                                                                                             | 0/5 [00:00<?, ?it/s][INFO] 2026-04-30 16:16:40,198 [RapidOCR] base.py:22: Using engine_name: torch
[INFO] 2026-04-30 16:16:40,204 [RapidOCR] device_config.py:64: Using GPU device with ID: 0
[INFO] 2026-04-30 16:16:40,215 [RapidOCR] download_file.py:60: File exists and is valid: C:\rag-production\.venv\Lib\site-packages\rapidocr\models\ch_PP-OCRv4_det_mobile.pth
[INFO] 2026-04-30 16:16:40,216 [RapidOCR] main.py:50: Using C:\rag-production\.venv\Lib\site-packages\rapidocr\models\ch_PP-OCRv4_det_mobile.pth
[INFO] 2026-04-30 16:16:40,721 [RapidOCR] base.py:22: Using engine_name: torch
[INFO] 2026-04-30 16:16:40,721 [RapidOCR] device_config.py:64: Using GPU device with ID: 0
[INFO] 2026-04-30 16:16:40,723 [RapidOCR] download_file.py:60: File exists and is valid: C:\rag-production\.venv\Lib\site-packages\rapidocr\models\ch_ptocr_mobile_v2.0_cls_mobile.pth
[INFO] 2026-04-30 16:16:40,723 [RapidOCR] main.py:50: Using C:\rag-production\.venv\Lib\site-packages\rapidocr\models\ch_ptocr_mobile_v2.0_cls_mobile.pth
[INFO] 2026-04-30 16:16:40,797 [RapidOCR] base.py:22: Using engine_name: torch
[INFO] 2026-04-30 16:16:40,797 [RapidOCR] device_config.py:64: Using GPU device with ID: 0
[INFO] 2026-04-30 16:16:40,816 [RapidOCR] download_file.py:60: File exists and is valid: C:\rag-production\.venv\Lib\site-packages\rapidocr\models\ch_PP-OCRv4_rec_mobile.pth
[INFO] 2026-04-30 16:16:40,816 [RapidOCR] main.py:50: Using C:\rag-production\.venv\Lib\site-packages\rapidocr\models\ch_PP-OCRv4_rec_mobile.pth
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 770/770 [00:00<00:00, 4318.27it/s]
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (93234 > 8192). Running this sequence through the model will result in indexing errors                                     | 408/770 [00:00<00:00, 4077.53it/s]
       ℹ  1461 chunks extraits par HybridChunker
       ✅ 1461 vecteurs indexés
[2/7] 🔄 Code des instruments monétaires et des médailles.pdf
       ℹ  9 pages → Docling GPU direct + HybridChunker
       ℹ  17 chunks extraits par HybridChunker
       ✅ 17 vecteurs indexés
[3/7] 🔄 Code des pensions de retraite des marins français du commerce, de pêche ou de plaisance.pdf
       ℹ  18 pages → Docling GPU direct + HybridChunker
       ℹ  60 chunks extraits par HybridChunker
       ✅ 60 vecteurs indexés
[4/7] 🔄 Code disciplinaire et pénal de la marine marchande.pdf
       ℹ  10 pages → Docling GPU direct + HybridChunker
       ℹ  18 chunks extraits par HybridChunker
       ✅ 18 vecteurs indexés
[5/7] 🔄 Code du domaine de l'Etat et des collectivités publiques applicable à la collectivité territoriale de Mayotte.pdf
       ℹ  8 pages → Docling GPU direct + HybridChunker
       ℹ  19 chunks extraits par HybridChunker
       ✅ 19 vecteurs indexés
[6/7] 🔄 Code du domaine public fluvial et de la navigation intérieure.pdf
       ℹ  8 pages → Docling GPU direct + HybridChunker
       ℹ  15 chunks extraits par HybridChunker
       ✅ 15 vecteurs indexés
[7/7] 🔄 Code du tourisme.pdf
       ℹ  192 pages → 4 tranches de 50 (Docling GPU + HybridChunker)
  Tranches (192 pages):   0%|                                                                                                                                              Tranches (192 pages):  25%|█████████████████████████████████████████████████████                                                                                         Tranches (192 pages):  50%|██████████████████████████████████████████████████████████████████████████████████████████████████████████                                    Tranches (192 pages):  75%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████  Tranches (192 pages): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                                                                                ℹ  1686 chunks extraits par HybridChunker
       ✅ 1686 vecteurs indexés

============================================================
✅ Ingestion terminée
   Fichiers traités : 7/7
   Vecteurs insérés : 3276
   Total en base    : 19906 vecteurs
============================================================
```

##### Pour voir les points insérés dans Qdrant : Dans le navigateur web 

```text
http://localhost:6333/collections/prod_documents
```

```text
les fichiers qdrant sont stockés dans le dossier que vous avez monté comme volume Docker :

C:\rag-production\data\qdrant_storage\
```

```bash
docker logs qdrant
```


### ✅ Option 1 pour suivre l'activité sur la carte GPU : 

##### Pour un affichage continu et lisible : (en powershell)
```bash
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu --format=csv -l 1
```
<img width="1315" height="178" alt="image" src="https://github.com/user-attachments/assets/5c019631-b70a-40e0-90b0-c0a07957877b" />


💡 Fonctionne aussi depuis l'intérieur d'un conteneur Docker si vous passez --gpus all.


### ✅ Option 2 — nvitop (recommandé pour le dev)

##### htop pour GPU, très lisible dans le terminal :
```bash
pip install nvitop
nvitop
```

<img width="2151" height="514" alt="image" src="https://github.com/user-attachments/assets/a4a6dc03-9c17-4a87-b899-53fd11bddea6" />

### ✅ Option 3 pour vérifier la charge pendant l'ingestion :
```bash
# Dans un terminal PowerShell :  
nvidia-smi -l 1
```

##### Vous devriez voir le pourcentage de "Volatile GPU-Util" grimper bien au-delà des 10-20% habituels lors des phases d'extraction et d'embedding.  

##### Si vous augmentez ces valeurs, surveillez bien la colonne Memory-Usage. Si vous approchez des 22-23 Go, baissez un peu le layout_batch_size.



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

#### Vous pouvez utiliser deux syntaxes :

##### Sans filtre (comportement par défaut — tous les fichiers) :
❓ Question : 
```question
règles du Code de la mutualité
```

##### Avec filtre (cherche uniquement dans un fichier précis) :
❓ Question :

Pour afficher la liste complète avec les noms exacts à copier-coller dans vos @"...".
(attention, c'est long !!!) 
```question
 sources
```

```question
 @"Code de la mutualité.pdf" quelles sont les règles en synthèse ?
```

Et la commande sources affiche la liste complète avec les noms exacts à copier-coller dans vos @"...".

---
<img width="621" height="286" alt="image" src="https://github.com/user-attachments/assets/a41f2723-50e2-4837-95d9-c9e1271b66ed" />

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

## Etape 8 — Vérifier l'indexation 

```powershell
cd C:\rag-production
.venv\Scripts\Activate.ps1
python check_index.py
```

##### Exemple en retour (long) : 

```powershell
Fichier                                                        Chunks
----------------------------------------------------------------------
Code civil.pdf                                                   2947
Code de commerce.pdf                                             9536
Code de déontologie des architectes.pdf                            59
Code de l'aviation civile.pdf                                      88
Code de la famille et de l'aide sociale.pdf                        24
Code de la mutualité.pdf                                         1120
Code des instruments monétaires et des médailles.pdf               16
Code des pensions de retraite des marins français du commerce, de pêche ou de plaisance.pdf       40
Code disciplinaire et pénal de la marine marchande.pdf             16
Code du domaine de l'Etat et des collectivités publiques applicable à la collectivité territoriale de Mayotte.pdf       14
Code du domaine public fluvial et de la navigation intérieure.pdf        9
Code du tourisme.pdf                                             1052

Total : 14921 vecteurs dans 12 fichiers
```


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




### Exemple d'intéraction : 
```powershell
(.venv) PS C:\rag-production> python query.py
============================================================
  RAG Production — Recherche interactive
  LLM      : mistral-nemo:12b
  Embedding: nomic-embed-text  |  Top-K: 15
============================================================
[Qdrant] Collection 'prod_documents' — 32840 vecteurs

  Commandes disponibles :
    sources          → liste les fichiers indexés avec leur nombre de chunks
    stats            → nombre total de vecteurs en base
    quit / q         → quitter

  Filtre par fichier (évite les confusions entre codes) :
    @"Code de la mutualité.pdf" quelles sont les règles ?
    @Code_civil.pdf  qu'est-ce que la responsabilité civile ?

❓ Question : qu'est ce qu'est la responsabilité civile ?

🔎 Recherche des passages pertinents...

🤖 Génération (streaming — mistral-nemo:12b)...

📎 Sources utilisées :
   • Code civil.pdf (chunk 1602, similarité 0.851)
   • Code civil.pdf (chunk 1688, similarité 0.848)
   • Code civil.pdf (chunk 2361, similarité 0.825)
   • Code civil.pdf (chunk 2243, similarité 0.816)
   • Code civil.pdf (chunk 2372, similarité 0.798)
   • Code civil.pdf (chunk 2237, similarité 0.778)
   • Code de justice militaire (nouveau).pdf (chunk 395, similarité 0.777)
   • Code de la justice pénale des mineurs.pdf (chunk 810, similarité 0.774)
   • Code civil.pdf (chunk 3366, similarité 0.769)
   • Code de justice militaire (nouveau).pdf (chunk 387, similarité 0.769)
   • Code civil.pdf (chunk 3725, similarité 0.769)
   • Code civil.pdf (chunk 1430, similarité 0.768)
   • Code civil.pdf (chunk 4572, similarité 0.767)
   • Code de commerce.pdf (chunk 2402, similarité 0.766)
   • Code de l'expropriation pour cause d'utilité publique.pdf (chunk 657, similarité 0.766)

────────────────────────────────────────────────────────────
La responsabilité civile est une notion juridique qui définit l'obligation pour un individu ou une entité de réparer le dommage causé à un tiers par sa propre faute. Cette obligation peut être engagée en cas de dommage matériel, moral ou corporel, causé à autrui par le fait du responsable.

La responsabilité civile peut être engagée dans divers domaines tels que la route (responsabilité civile automobile), la vie privée (responsabilité civile du particulier), la vie professionnelle (responsabilité civile des entreprises) etc.

En France, la loi dispose que toute personne qui a causé un dommage à autrui, par sa faute ou celle de ses préposés, est tenue de réparer ce dommage. La réparation peut prendre différentes formes : dommages-intérêts, remise en état, fourniture d'une chose équivalente etc.

Il convient également de noter que la responsabilité civile peut être limitée ou exclue dans certains cas prévus par la loi (exemple : clause limitative de responsabilité).
────────────────────────────────────────────────────────────

❓ Question : sources

📚 25 fichiers indexés :
    1. Code civil.pdf  (5014 chunks)
    2. Code de commerce.pdf  (11359 chunks)
    3. Code de déontologie des architectes.pdf  (93 chunks)
    4. Code de justice militaire (nouveau).pdf  (899 chunks)
    5. Code de l'artisanat.pdf  (1622 chunks)
    6. Code de l'aviation civile.pdf  (130 chunks)
    7. Code de l'expropriation pour cause d'utilité publique.pdf  (830 chunks)
    8. Code de la Légion d'honneur, de la Médaille militaire et de l'ordre national du Mérite.pdf  (412 chunks)
    9. Code de la famille et de l'aide sociale.pdf  (34 chunks)
   10. Code de la justice pénale des mineurs.pdf  (1278 chunks)
   11. Code de la mutualité.pdf  (1461 chunks)
   12. Code de la voirie routière.pdf  (1093 chunks)
   13. Code des communes.pdf  (377 chunks)
   14. Code des douanes.pdf  (1294 chunks)
   15. Code des instruments monétaires et des médailles.pdf  (17 chunks)
   16. Code des pensions civiles et militaires de retraite.pdf  (571 chunks)
   17. Code des pensions de retraite des marins français du commerce, de pêche ou de plaisance.pdf  (60 chunks)
   18. Code des relations entre le public et l'administration.pdf  (1103 chunks)
   19. Code disciplinaire et pénal de la marine marchande.pdf  (18 chunks)
   20. Code du domaine de l'Etat et des collectivités publiques applicable à la collectivité territoriale de Mayotte.pdf  (19 chunks)
   21. Code du domaine de l'Etat.pdf  (1390 chunks)
   22. Code du domaine public fluvial et de la navigation intérieure.pdf  (15 chunks)
   23. Code du tourisme.pdf  (1686 chunks)
   24. Code minier (nouveau).pdf  (1810 chunks)
   25. Code minier.pdf  (255 chunks)

❓ Question : quit
Au revoir !
```

Have Fun !
