# HOMEBARISTA COACH — ULTRAPLAN
## RAG-powered coffee diagnostic & coaching assistant

> Version : 1.0 — Mai 2026
> Contexte : LLM Zoomcamp (DataTalks.Club) + portfolio Data/AI Engineering
> Objectif : projet portfolio démontrant un RAG pipeline complet, évaluable, déployable
> Thématique : café / barista / extraction — 0 concurrent dans la promo 2025

---

## 0. Ce qu'on construit et pourquoi

**HomeBarista Coach** est un coach barista IA qui diagnostique les problèmes d'extraction
et guide l'utilisateur vers un café parfait, quelle que soit sa machine.

L'utilisateur dit :
> "DeLonghi Dinamica, grains éthiopiens light roast, mon espresso est trop acide, 
> extraction 22 secondes."

HomeBarista Coach :
1. **Extrait** les symptômes, la machine et les paramètres de la description
2. **Diagnostique** les causes racines probables via un moteur déterministe
3. **Récupère** les passages les plus pertinents depuis une knowledge base de contenu 
   barista (James Hoffmann, SCA, World Barista Championship, guides d'extraction)
4. **Génère** un plan d'action coaching précis, étape par étape, via LLM
5. **Évalue** la qualité et la sécurité du coaching produit

**Pourquoi YouTube et guides techniques :**
- James Hoffmann (2M+ abonnés) : dizaines d'heures de contenu ultra-technique avec transcripts
- SCA / Barista Hustle : standards professionnels de l'industrie café
- World Barista Championship : talks techniques de niveau expert
- Transcripts via `youtube-transcript-api` : signal riche, vocabulaire précis
- RAG genuinement justifié : corpus de 300+ documents hétérogènes, vocabulaire technique dense,
  recherche sémantique sur des symptômes non structurés

**Pourquoi le diagnostic pattern fonctionne :**
Le projet Plant Disease RAG (LLM Zoomcamp 2025, score 22/25) valide ce pattern :
symptôme décrit en langage naturel → RAG sur corpus technique → diagnostic + plan d'action.
HomeBarista applique exactement ce pattern au domaine café. Aucun concurrent dans la promo 2025.

**Pourquoi ce projet répond aux critères LLM Zoomcamp :**

| Critère Zoomcamp | Comment HomeBarista l'adresse |
|-----------------|-------------------------------|
| Problem description | Diagnostic café depuis description libre |
| RAG flow | Symptômes → ChromaDB → LLM coaching |
| Retrieval evaluation | Precision@k, MRR sur dataset annoté |
| RAG evaluation | Checks déterministes + LLM judge |
| UI | Streamlit avec mode demo |
| Ingestion pipeline | YouTube API + transcripts + embeddings |
| Monitoring (bonus) | Logs JSON + dashboard |
| Containerisation (bonus) | Docker Compose |

---

## 1. Architecture complète

```
┌─────────────────────────────────────────────────────┐
│                  OFFLINE PIPELINE                   │
│  (tourne une fois + scheduled refresh)              │
│                                                     │
│  YouTube Data API v3                                │
│  → métadonnées (titre, description, tags, durée)    │
│         ↓                                           │
│  youtube-transcript-api                             │
│  → transcripts (sous-titres auto ou manuels)        │
│         ↓                                           │
│  ContentClassifier                                  │
│  → domaine : extraction / grind / machine /         │
│    origin / method / troubleshooting                │
│  → méthode : espresso / moka / v60 /                │
│    aeropress / french_press / super_automatic       │
│         ↓                                           │
│  Chunker + Embedder                                 │
│  → chunks de 400 tokens, overlap 80                 │
│  → sentence-transformers/all-MiniLM-L6-v2           │
│         ↓                                           │
│  ChromaDB (persist)                                 │
│  → collection "barista_knowledge"                   │
│  → metadata: source_id, domain, method,             │
│    channel, difficulty, url                         │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                  ONLINE PIPELINE                    │
│  (par requête utilisateur)                          │
│                                                     │
│  User input: machine + beans + problem + goal       │
│         ↓                                           │
│  SymptomExtractor (règles + LLM)                    │
│  → symptoms[], machine_type, method,                │
│    parameters, goal                                 │
│         ↓                                           │
│  DiagnosticPlanner (déterministe)                   │
│  → root_causes[] avec probabilités                  │
│  → intervention_plan[] ordonné                      │
│         ↓                                           │
│  Query Builder → embed query enrichie               │
│         ↓                                           │
│  ChromaDB retrieval (top-k=15)                      │
│  → filtres metadata: method, domain                 │
│         ↓                                           │
│  Re-ranker                                          │
│  → score = sémantique × 0.6 + pertinence × 0.4     │
│         ↓                                           │
│  CoachingEvaluator (déterministe)                   │
│  → vérifie complétude + cohérence + sécurité        │
│         ↓                                           │
│  CoachingGenerator (LLM)                            │
│  → plan coaching personnalisé par étape             │
│  → explication des causes + tests de validation     │
│         ↓                                           │
│  Logger → sessions.jsonl                            │
│         ↓                                           │
│  Streamlit UI                                       │
└─────────────────────────────────────────────────────┘
```

---

## 2. Stack technique

| Brique | Outil | Justification |
|--------|-------|--------------|
| Données YouTube | `google-api-python-client` (YouTube Data API v3) | Officiel, stable, 10k quota/day |
| Transcripts | `youtube-transcript-api` | Très fiable, pas de quota |
| Content classification | Règles keyword + `claude-haiku` backup | Rapide, coût minimal |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Gratuit, local, 384 dims |
| Vector store | ChromaDB (persistent) | Simple, local, suffisant portfolio |
| LLM coach | `anthropic` SDK (claude-haiku-3) | Rapide, cheap, excellent pour formulation |
| Moteur diagnostic | Python pur (dataclasses + règles) | Déterministe, testable |
| Évaluation | Python pur + dataset annoté | Reproductible, CI-compatible |
| UI | Streamlit | Deploy sur Streamlit Cloud en 5 min |
| Tests | pytest | Coverage sur moteur + classifier + evaluator |
| Dépendances | uv + pyproject.toml | Standard moderne |
| Containerisation | Docker Compose (bonus) | +1 Zoomcamp |

**Ce qu'on n'utilise pas :**
- OpenAI embeddings → coût inutile, MiniLM suffit
- Pinecone / Weaviate → overkill pour portfolio
- LangChain / LlamaIndex → trop opaque, pipeline custom plus défendable
- APIs propriétaires café → pas nécessaire, YouTube suffit

---

## 3. Structure du repo

```
homebarista/
├── README.md                        # Portfolio-ready
├── pyproject.toml                   # uv dependencies
├── docker-compose.yml               # (bonus) Zoomcamp
├── .env.example
│
├── homebarista/
│   ├── __init__.py
│   ├── models.py                    # Dataclasses : BrewingContext, DiagnosticResult,
│   │                                #   Intervention, CoachingSession
│   ├── youtube_client.py            # YouTube Data API v3
│   ├── transcript_fetcher.py        # youtube-transcript-api
│   ├── content_classifier.py        # Classification domaine + méthode
│   ├── symptom_extractor.py         # Extraction symptômes depuis texte libre
│   ├── embedder.py                  # sentence-transformers
│   ├── vector_store.py              # ChromaDB wrapper
│   ├── retriever.py                 # Query + re-ranking
│   ├── diagnostic_planner.py        # Moteur déterministe diagnostic
│   ├── coaching_evaluator.py        # Évaluation qualité coaching
│   ├── coaching_generator.py        # LLM (Anthropic)
│   └── pipeline.py                  # Orchestration complète
│
├── ingestion/
│   ├── run_ingestion.py             # Script d'ingestion offline
│   ├── channels.yaml                # Chaînes YouTube + playlists à indexer
│   └── ingestion_report.json        # Rapport du dernier run
│
├── data/
│   ├── chroma_db/                   # ChromaDB persist (gitignored)
│   ├── mock_documents.json          # 40 docs mock pour tests sans API
│   └── eval_dataset.json            # Dataset annoté pour retrieval eval
│
├── evals/
│   ├── run_retrieval_eval.py        # Évalue le retrieval (Precision@k, MRR)
│   ├── run_rag_eval.py              # Évalue la qualité du coaching généré
│   └── results/                     # JSON des résultats
│
├── tests/
│   ├── test_symptom_extractor.py
│   ├── test_diagnostic_planner.py
│   ├── test_coaching_evaluator.py
│   ├── test_retriever.py
│   └── fixtures/
│       └── mock_documents.json
│
├── app/
│   └── streamlit_app.py
│
└── logs/
    └── sessions.jsonl               # (gitignored)
```

---

## 4. Modèle de données (models.py)

```python
from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class BrewingContext:
    """Ce que l'utilisateur nous dit sur sa situation."""
    machine_type: Literal[
        "super_automatic", "semi_automatic", "manual_lever",
        "moka", "v60", "aeropress", "french_press", "nespresso", "unknown"
    ]
    machine_model: Optional[str] = None        # ex: "DeLonghi Dinamica"
    bean_origin: Optional[str] = None          # ex: "Ethiopian Yirgacheffe"
    roast_level: Optional[Literal["light", "medium", "dark", "unknown"]] = None
    bean_freshness_days: Optional[int] = None  # jours depuis torréfaction
    grind_size: Optional[str] = None           # "fine" | "medium" | "coarse" | "setting X"
    dose_grams: Optional[float] = None
    water_temp_celsius: Optional[float] = None
    extraction_time_seconds: Optional[int] = None
    water_ratio: Optional[str] = None          # ex: "1:15" pour V60
    raw_problem: str = ""                      # description libre de l'utilisateur
    goal: Literal["troubleshoot", "optimize", "explore", "learn"] = "troubleshoot"

@dataclass
class RootCause:
    """Une cause racine probable du problème."""
    hypothesis: str                            # ex: "sous-extraction"
    probability: float                         # 0.0 à 1.0
    evidence: str                              # pourquoi on pense ça
    parameter_affected: Optional[str] = None  # ex: "grind_size"

@dataclass
class Intervention:
    """Une action corrective à effectuer."""
    step: int
    action: str                                # ex: "Affiner la mouture d'un cran"
    parameter: Optional[str] = None           # ex: "grind_size"
    direction: Optional[str] = None           # ex: "finer" | "coarser" | "higher"
    magnitude: Optional[str] = None           # ex: "1 cran" | "2°C" | "3 secondes"
    expected_result: str = ""
    validation_test: str = ""                 # comment savoir si ça a marché
    priority: Literal["critical", "high", "medium", "low"] = "high"

@dataclass
class DiagnosticResult:
    """Le résultat du moteur de diagnostic déterministe."""
    symptoms: list[str]                        # extraits du texte libre
    root_causes: list[RootCause]               # triés par probabilité desc
    intervention_plan: list[Intervention]      # plan d'action ordonné
    diagnostic_confidence: float               # 0.0 à 1.0
    method_detected: str                       # méthode de brassage identifiée
    warnings: list[str] = field(default_factory=list)

@dataclass
class CoachingSession:
    """La session de coaching complète."""
    context: BrewingContext
    diagnostic: DiagnosticResult
    coaching_text: str = ""                    # output LLM
    follow_up_questions: list[str] = field(default_factory=list)
    evaluation: dict = field(default_factory=dict)
    retrieval_metadata: dict = field(default_factory=dict)
    session_id: str = ""
```

---

## 5. Corpus à indexer (channels.yaml)

```yaml
# Canaux YouTube barista / café (transcripts disponibles)
channels:
  - id: UCMb0O2CdPBNi-QqPk5T3gsQ
    name: James Hoffmann
    tags: [espresso, filter, grind, origin, equipment, technique]
    priority: critical  # Corpus #1 — ultra-dense en connaissance technique
    max_videos: 80

  - id: UCbS5dZ7rVAGBWMQGbMBZCcg  
    name: Barista Hustle
    tags: [extraction, sca, professional, technique, science]
    priority: critical  # Standards SCA, science de l'extraction
    max_videos: 50

  - id: UCVuB_6YFD7-lrHoqnHCg2Lg
    name: Lance Hedrick
    tags: [espresso, dialing_in, equipment, super_automatic, grind]
    priority: high
    max_videos: 40

  - id: UCn3jcZBFRhAHAC4Xr9sTkQQ
    name: European Coffee Trip
    tags: [origin, roasting, specialty, method]
    priority: medium
    max_videos: 30

  - id: UCo4VPxVEiPSDljgcXIe5RJA
    name: Tim Wendelboe
    tags: [filter, origin, technique, professional]
    priority: medium
    max_videos: 20

# Playlists spécifiques à indexer en priorité
playlists:
  - id: PLynbSsRRLfnoR4CZrRlhvkbY1xPQ5e7Py
    name: "James Hoffmann - How to make better coffee"
    tags: [beginner, technique, troubleshooting]
  
  - id: PLynbSsRRLfnoV1HiYeF0n3CwDo2gm59Gx
    name: "James Hoffmann - Espresso Machines"
    tags: [espresso, machine, equipment, super_automatic]

# Cible : 300-500 documents (vidéos chunkées) pour un portfolio project
```

---

## 6. Classification des domaines (ContentClassifier)

| Domaine | Signaux textuels (keywords) |
|---------|---------------------------|
| `extraction` | extraction, over-extracted, under-extracted, yield, TDS, EY, ratio |
| `grind` | grind, burr, grinder, coarseness, particle size, retention |
| `machine` | machine, boiler, pressure, pump, portafilter, basket, temperature |
| `origin` | Ethiopian, Colombian, Brazilian, washed, natural, honey, terroir |
| `method` | espresso, moka, v60, aeropress, french press, pour over, immersion |
| `troubleshooting` | bitter, sour, acidic, bland, weak, watery, channeling, crema |
| `technique` | tamping, distribution, bloom, pre-infusion, flow rate, ratio |

| Méthode détectée | Signaux |
|-----------------|---------|
| `espresso` | espresso, portafilter, puck, crema, shot, 9 bar |
| `super_automatic` | super automatic, bean to cup, DeLonghi, Jura, Philips, Saeco |
| `moka` | moka, stovetop, bialetti, percolator |
| `v60` | v60, pour over, dripper, filter paper, bloom |
| `aeropress` | aeropress, inverted, pressure, plunger |
| `french_press` | french press, plunger, immersion, 4 minutes |

---

## 7. Matrice de diagnostic (DiagnosticPlanner)

```python
# La bible du diagnostic déterministe

SYMPTOM_MATRIX = {
    "bitter": {
        "root_causes": [
            RootCause("over-extraction", 0.70, "bitter = extraction trop longue ou trop fine"),
            RootCause("grind_too_fine", 0.65, "mouture trop fine = contact trop long"),
            RootCause("water_too_hot", 0.50, "eau trop chaude extrait les composés amers"),
            RootCause("dark_roast_mismatch", 0.40, "robusta ou dark roast nécessite ajustements"),
        ],
        "method_overrides": {
            "moka": [
                RootCause("heat_too_high", 0.80, "chaleur excessive brûle le café dans la moka"),
                RootCause("too_long_on_flame", 0.75, "café resté trop longtemps sur le feu"),
            ],
            "super_automatic": [
                RootCause("grind_too_fine", 0.70, "réglage mouture trop fin sur machine auto"),
                RootCause("temperature_setting_high", 0.55, "température configurée trop haute"),
            ]
        },
        "interventions": [
            Intervention(1, "Alléger la mouture d'un cran", "grind_size", "coarser", "1 cran"),
            Intervention(2, "Réduire la température de 1-2°C", "water_temp", "lower", "1-2°C"),
            Intervention(3, "Raccourcir le temps d'extraction de 3-5 secondes", "time", "shorter"),
        ]
    },
    
    "sour": {
        "root_causes": [
            RootCause("under-extraction", 0.75, "sour = extraction incomplète"),
            RootCause("grind_too_coarse", 0.70, "mouture trop grossière = contact insuffisant"),
            RootCause("water_too_cold", 0.55, "eau trop froide n'extrait pas les sucres"),
            RootCause("too_short_extraction", 0.60, "extraction trop courte"),
            RootCause("light_roast_challenge", 0.45, "light roast naturellement plus acide"),
        ],
        "interventions": [
            Intervention(1, "Affiner la mouture d'un cran", "grind_size", "finer", "1 cran"),
            Intervention(2, "Augmenter la température de 1-2°C", "water_temp", "higher", "1-2°C"),
            Intervention(3, "Allonger l'extraction de 3-5 secondes", "time", "longer"),
        ]
    },
    
    "weak_bland": {
        "root_causes": [
            RootCause("dose_too_low", 0.70, "pas assez de café par rapport à l'eau"),
            RootCause("grind_too_coarse", 0.65, "mouture trop grossière"),
            RootCause("stale_beans", 0.55, "grains rassis = moins d'arômes"),
            RootCause("under-extraction", 0.50, "extraction incomplète"),
        ],
        "interventions": [
            Intervention(1, "Augmenter la dose de 1-2g", "dose", "higher", "1-2g"),
            Intervention(2, "Affiner légèrement la mouture", "grind_size", "finer", "0.5 cran"),
            Intervention(3, "Vérifier la fraîcheur des grains", "freshness", priority="critical"),
        ]
    },
    
    "thin_crema": {
        "root_causes": [
            RootCause("stale_beans", 0.75, "crema fine = CO2 épuisé = grains rassis"),
            RootCause("grind_too_coarse", 0.60, "mouture trop grossière"),
            RootCause("low_pressure", 0.50, "pression insuffisante pour émulsionner"),
            RootCause("super_auto_setting", 0.45, "réglage machine sous-optimal"),
        ],
        "interventions": [
            Intervention(1, "Vérifier la date de torréfaction (< 3 semaines idéal)", "freshness"),
            Intervention(2, "Affiner la mouture d'un cran", "grind_size", "finer"),
            Intervention(3, "Détartrer la machine si nécessaire", "maintenance"),
        ]
    },
    
    "channeling": {
        "root_causes": [
            RootCause("uneven_distribution", 0.75, "mauvaise répartition du café"),
            RootCause("tamping_uneven", 0.65, "tassage irrégulier"),
            RootCause("dose_inconsistent", 0.50, "dose variable"),
        ],
        "interventions": [
            Intervention(1, "Utiliser un outil de distribution (WDT)", "technique"),
            Intervention(2, "Tasser horizontalement avec pression uniforme", "tamping"),
            Intervention(3, "Vérifier la dose avec une balance", "dose"),
        ]
    }
}
```

---

## 8. Plan de développement phase par phase

---

### PHASE 0 — Setup & fondations (Jour 1)

**Prompt Claude Code #0**

```
Create a new Python project called "homebarista" using uv.

pyproject.toml dependencies:
  google-api-python-client, youtube-transcript-api, sentence-transformers,
  chromadb, anthropic, streamlit, pytest, python-dotenv, pydantic, httpx, pyyaml

.env.example:
  YOUTUBE_API_KEY=
  ANTHROPIC_API_KEY=
  CHROMA_PERSIST_DIR=data/chroma_db
  DEMO_MODE=true

Create the full directory structure as specified in the ULTRAPLAN:
  homebarista/, ingestion/, data/, evals/, tests/, app/, logs/

Create homebarista/models.py with the exact dataclasses:
  BrewingContext, RootCause, Intervention, DiagnosticResult, CoachingSession
[paste models above]

Create data/mock_documents.json with 40 mock barista knowledge documents covering:
  - 10 documents on espresso extraction (bitter, sour, channeling, crema)
  - 8 documents on grind and grinders
  - 8 documents on super-automatic machines (DeLonghi, Jura, Philips)
  - 6 documents on moka pot technique
  - 5 documents on filter methods (V60, Aeropress)
  - 3 documents on coffee origins and roast levels
  
Each mock document must have:
  source_id, title, channel, url, domain, method, 
  content (500-800 chars of realistic barista knowledge text),
  difficulty ("beginner"|"intermediate"|"advanced")

Run "uv sync" and verify all imports work.
```

---

### PHASE 1 — SymptomExtractor (Jour 1-2)

**Prompt Claude Code #1**

```
Create homebarista/symptom_extractor.py implementing SymptomExtractor.

Input: raw user text (problem description, free form)
Output: BrewingContext object (populated with extracted info)

Method: extract(raw_text: str) -> BrewingContext

Step 1 — Rule-based extraction:

  MACHINE_KEYWORDS = {
    "super_automatic": ["delonghi dinamica", "jura", "philips", "saeco", 
                        "bean to cup", "super automatic", "automatique"],
    "moka": ["moka", "bialetti", "stovetop", "italienne"],
    "v60": ["v60", "pour over", "hario", "filtre"],
    "aeropress": ["aeropress", "aero press"],
    "french_press": ["french press", "cafetière à piston"],
    "semi_automatic": ["breville", "la marzocco", "rancilio", "gaggia"],
    "nespresso": ["nespresso", "vertuo", "capsule"],
  }

  SYMPTOM_KEYWORDS = {
    "bitter": ["amer", "bitter", "âcre", "brûlé"],
    "sour": ["acide", "sour", "acidité", "piquant", "aigre"],
    "weak_bland": ["fade", "faible", "weak", "bland", "insipide", "eau chaude"],
    "thin_crema": ["crème fine", "thin crema", "pas de crème", "crema absente"],
    "channeling": ["channeling", "irrégulier", "extraction inégale"],
    "bitter_and_sour": ["complexe", "déséquilibré"],
  }

  Extract also:
  - Numbers for extraction_time (regex: \d+ (sec|s|secondes?))
  - Numbers for dose (regex: \d+\.?\d* (g|gr|grammes?))
  - Temperature (regex: \d+ (°C|degrés?|celsius))
  - Grind setting mentions (fin, grossier, coarse, fine + cran, notch, setting)
  - Bean origin keywords (Ethiopian, Colombian, Brazilian, Yirgacheffe, etc.)
  - Roast level (light, medium, dark, clair, foncé, torréfaction)
  - Freshness (days since roast if mentioned)

Step 2 — LLM fallback (only if machine_type = "unknown" after rules):
  Use claude-haiku:
  "Extract from this coffee problem description: machine type, main symptoms, 
   brewing method. Respond in JSON:
   {'machine_type': str, 'symptoms': [str], 'method': str}"

Step 3 — Goal detection:
  "troubleshoot" if symptoms detected
  "optimize" if user says "améliorer", "better", "improve", "optimiser"
  "explore" if user says "essayer", "try", "nouveau", "new", "comprendre"
  "learn" if user says "apprendre", "learn", "débutant", "beginner"

Tests in tests/test_symptom_extractor.py:
  - test_bitter_espresso: "mon espresso delonghi est amer" → 
    machine_type="super_automatic", symptoms=["bitter"]
  - test_sour_moka: "ma moka donne un café très acide" → 
    machine_type="moka", symptoms=["sour"]
  - test_extraction_time: "extraction 22 secondes" → extraction_time_seconds=22
  - test_multilingual: works with French and English input
  - test_unknown_machine: ambiguous input triggers LLM fallback
  - test_full_context: "DeLonghi Dinamica, Yirgacheffe light roast, 
    espresso acide, 20 secondes, mouture 8" → full BrewingContext populated
```

---

### PHASE 2 — YouTube Client + Transcript Fetcher (Jour 2)

**Prompt Claude Code #2**

```
Create homebarista/youtube_client.py and homebarista/transcript_fetcher.py.

=== youtube_client.py ===
Class YouTubeClient using google-api-python-client.

Methods:
1. get_channel_videos(channel_id: str, max_results: int = 50) -> list[dict]
   Returns: videoId, title, description, tags, duration (seconds), channelTitle.
   Handle pagination. Log quota usage.

2. get_playlist_videos(playlist_id: str, max_results: int = 50) -> list[dict]
   Same format. Playlists allow targeting best content.

3. build_document_object(raw: dict, channel_tags: list[str]) -> dict
   Maps API response → document dict ready for ContentClassifier.
   Include: source_id (= videoId), title, channel, url, description, 
   tags (merged: api tags + channel tags), duration_seconds.

=== transcript_fetcher.py ===
Class TranscriptFetcher using youtube-transcript-api.

Methods:
1. fetch_transcript(video_id: str) -> tuple[str, bool]
   Language preference: ["en", "fr"] (English first for barista content).
   Returns (full_transcript_text, is_available).
   No length limit — full transcript needed for chunking.
   
2. fetch_batch(video_ids: list[str]) -> dict[str, str]
   Rate limiting: sleep 0.3s between calls.
   Skip videos without transcripts gracefully.
   Print progress every 10 videos.

Both classes work in DEMO_MODE=true (skip API calls, return mock data).
```

---

### PHASE 3 — ContentClassifier (Jour 2)

**Prompt Claude Code #3**

```
Create homebarista/content_classifier.py implementing ContentClassifier.

Input: document dict (title, description, tags, transcript_text)
Output: (domain: str, method: str, difficulty: str, confidence: float)

Method: classify(doc: dict) -> dict

Step 1 — Domain classification (keyword scoring):
  Score each domain keyword list against:
  title × 4 + description × 2 + tags × 3 + transcript_text[:1000] × 1
  
  Domains: extraction, grind, machine, origin, method, troubleshooting, technique
  
  Top domain = winner. Multiple domains possible (store top 2).
  Confidence = top_score / sum_scores.

Step 2 — Method classification (same approach):
  Methods: espresso, super_automatic, moka, v60, aeropress, french_press, general
  
  "super_automatic" takes priority over "espresso" if both score high
  (super-auto is a subset of espresso machines but needs separate handling).

Step 3 — Difficulty classification (rule-based):
  "beginner" if: keywords like "beginner", "start", "basic", "introduction", 
                 "débutant", "commencer"
  "advanced" if: keywords like "SCA", "TDS", "EY", "refractometer", "barista 
                 championship", "Scott Rao", "professional"
  "intermediate" otherwise

Method: classify_batch(docs: list[dict]) -> list[dict]
  Updates each doc with domain, method, difficulty, classification_confidence.
  Print summary: "Classified X docs: {domain distribution}"

Tests in tests/test_content_classifier.py:
  - test_troubleshooting_bitter: doc about bitter espresso → domain="troubleshooting"
  - test_super_auto_priority: doc mentions both espresso and DeLonghi → method="super_automatic"
  - test_moka_detection: doc about stovetop coffee → method="moka"
  - test_batch_summary: 10 docs, verify summary printed
```

---

### PHASE 4 — Embedder + Vector Store (Jour 2-3)

**Prompt Claude Code #4**

```
Create homebarista/embedder.py and homebarista/vector_store.py.

=== embedder.py ===
Class Embedder using sentence-transformers (all-MiniLM-L6-v2, 384 dims).

Chunking strategy (important — different from video-level):
  For barista knowledge, chunk at PASSAGE level, not video level.
  Chunk size: 400 tokens, overlap: 80 tokens.
  Each chunk = one ChromaDB document.
  
  Why: a 45-min James Hoffmann video contains 10+ distinct topics.
  Chunking allows precise retrieval of the relevant passage
  (e.g., the 2 minutes where he explains why moka coffee gets bitter)
  rather than retrieving the entire video.

Methods:
1. chunk_transcript(transcript: str, source_doc: dict) -> list[dict]
   Split transcript into 400-token chunks with 80-token overlap.
   Each chunk gets: chunk_id (source_id + "_chunk_N"), text, 
   source_id, title, channel, url, domain, method, difficulty.

2. embed_chunk(chunk: dict) -> dict
   text = chunk["text"]
   Returns {"embedding": list[float], "text": str, "metadata": {...}}
   
   Metadata: chunk_id, source_id, title, channel, url, domain, method, difficulty

3. embed_query(query: str) -> list[float]

4. embed_batch(chunks: list[dict]) -> list[dict]
   Batches of 32. Progress bar.

=== vector_store.py ===
Class VectorStore wrapping ChromaDB.
Collection name: "barista_knowledge"

Methods:
1. add_chunks(embedded_chunks: list[dict]) -> None
   Upsert by chunk_id.

2. query(embedding: list[float], n_results: int = 15,
         where: dict = None) -> list[dict]
   Metadata filters: {"method": {"$in": ["espresso", "super_automatic"]}}
   Returns: chunk_id, text, source_id, title, url, distance, metadata

3. get_stats() -> dict
   {total_chunks, domain_distribution, method_distribution, 
    sources_indexed, channels_indexed}

4. delete_collection() -> None
```

---

### PHASE 5 — Ingestion Pipeline (Jour 3)

**Prompt Claude Code #5**

```
Create ingestion/run_ingestion.py — the offline ingestion script.

Steps:
1. Load ingestion/channels.yaml
2. For each channel:
   a. YouTubeClient.get_channel_videos(channel_id, max_results)
   b. YouTubeClient.get_playlist_videos() for priority playlists
   c. TranscriptFetcher.fetch_batch(video_ids)
   d. ContentClassifier.classify_batch(docs)
   e. Filter: keep only docs with transcript available
3. Embedder.chunk_transcript() for each doc → chunks
4. Embedder.embed_batch(all_chunks)
5. VectorStore.add_chunks(all_embedded)
6. Save ingestion_report.json:
   {
     "run_at": ISO timestamp,
     "channels_processed": int,
     "videos_fetched": int,
     "videos_with_transcript": int,
     "total_chunks_indexed": int,
     "domain_distribution": dict,
     "method_distribution": dict,
     "chunks_per_channel": dict,
     "quota_used": int
   }
7. Print report summary.

Flags:
  --demo: uses mock_documents.json (no API calls, CI-safe)
  --dry-run: runs full pipeline but skips ChromaDB write
  --channel CHANNEL_ID: ingest only one channel (useful for testing)

Run: python -m ingestion.run_ingestion [--demo] [--dry-run]
```

---

### PHASE 6 — Retriever (Jour 3)

**Prompt Claude Code #6**

```
Create homebarista/retriever.py implementing Retriever.

Method: retrieve(
    brewing_context: BrewingContext,
    diagnostic: DiagnosticResult,
    n_candidates: int = 15
) -> list[dict]

Steps:
1. Build retrieval query from diagnostic:
   query = f"{brewing_context.machine_type} {' '.join(diagnostic.symptoms)} 
            {diagnostic.root_causes[0].hypothesis} 
            {diagnostic.method_detected} coffee extraction"
   
   Add context enrichment:
   if bean_origin: query += f" {brewing_context.bean_origin}"
   if roast_level: query += f" {brewing_context.roast_level} roast"

2. Build ChromaDB metadata filter:
   allowed_methods = get_allowed_methods(brewing_context.machine_type)
   # super_automatic → ["super_automatic", "espresso", "general"]
   # moka → ["moka", "general"]
   # v60 → ["v60", "filter", "general"]
   where = {"method": {"$in": allowed_methods}}

3. ChromaDB query with metadata filter.

4. Re-rank by composite score:
   semantic_score = 1 - distance
   domain_score = 1.0 if chunk.domain in ["troubleshooting", "technique"] else 0.6
   difficulty_score = match_difficulty(brewing_context, chunk.difficulty)
   
   composite = semantic_score × 0.6 + domain_score × 0.25 + difficulty_score × 0.15

5. Return top-10 chunks sorted by composite score.
   Attach composite_score and retrieval_rank to each chunk.

Helper: get_allowed_methods(machine_type: str) -> list[str]
  Maps machine_type to relevant ChromaDB method tags.

Tests in tests/test_retriever.py:
  - test_moka_filter: moka machine never returns espresso-only chunks
  - test_troubleshooting_priority: bitter symptom returns troubleshooting chunks first
  - test_composite_range: all scores 0-1
  - test_demo_mode: works with mock ChromaDB
```

---

### PHASE 7 — DiagnosticPlanner (Jour 4)

**Prompt Claude Code #7**

```
Create homebarista/diagnostic_planner.py implementing DiagnosticPlanner.

This is the deterministic brain. It never calls the LLM.
It maps symptoms + context → root causes + intervention plan.

Method: diagnose(context: BrewingContext) -> DiagnosticResult

Steps:
1. Normalize symptoms from context.raw_problem 
   (already extracted by SymptomExtractor — use context fields).

2. Look up root causes from SYMPTOM_MATRIX for each detected symptom.
   Apply method_overrides if available for context.machine_type.
   Merge and deduplicate root causes. Adjust probabilities by context:
   - If extraction_time_seconds < 20 and symptom="sour": 
     boost under-extraction probability to 0.90
   - If extraction_time_seconds > 32 and symptom="bitter": 
     boost over-extraction probability to 0.90
   - If bean_freshness_days > 30: boost stale_beans probability
   - If machine_type="moka" and symptom="bitter": 
     boost heat_too_high probability to 0.85

3. Sort root_causes by probability descending.
   Take top 3 causes (avoid overwhelming user).

4. Build intervention_plan from top cause interventions.
   Prioritize "critical" interventions first.
   Max 4 interventions total (keep it actionable).

5. Calculate diagnostic_confidence:
   confidence = max(root_causes[0].probability) × completeness_factor
   completeness_factor = 1.0 if extraction_time known, 0.75 if not.

6. Populate warnings:
   - If no symptoms detected: "Description trop vague — précise tes symptômes"
   - If bean_freshness_days > 45: "⚠️ Grains probablement rassis — essaie d'abord de changer les grains"
   - If machine_type="unknown": "Machine non identifiée — conseils génériques"

SYMPTOM_MATRIX: [paste full matrix from section 7 of ultraplan]

Tests in tests/test_diagnostic_planner.py:
  - test_bitter_espresso: symptoms=["bitter"], method="espresso" 
    → root_causes[0].hypothesis contains "extraction"
  - test_moka_bitter_override: machine_type="moka", symptoms=["bitter"] 
    → root_causes[0].hypothesis == "heat_too_high"
  - test_sour_short_extraction: sour + extraction_time=18s 
    → under-extraction probability >= 0.85
  - test_stale_warning: bean_freshness_days=50 → warnings contains "rassis"
  - test_max_interventions: never returns more than 4 interventions
  - test_unknown_machine_warning: machine_type="unknown" → warning present
```

---

### PHASE 8 — CoachingEvaluator (Jour 4)

**Prompt Claude Code #8**

```
Create homebarista/coaching_evaluator.py implementing CoachingEvaluator.

Evaluates the full coaching session BEFORE LLM generation (structural checks)
and AFTER LLM generation (content checks).

=== Pre-generation checks (on DiagnosticResult) ===
def evaluate_diagnostic(diagnostic: DiagnosticResult, context: BrewingContext) -> dict:

1. symptoms_detected: len(diagnostic.symptoms) >= 1
2. root_causes_present: len(diagnostic.root_causes) >= 1
3. intervention_plan_present: len(diagnostic.intervention_plan) >= 1
4. max_confidence: diagnostic.diagnostic_confidence >= 0.20
   (if < 0.20 = problem too vague to diagnose)
5. method_identified: diagnostic.method_detected != "unknown"
6. no_contradictory_symptoms: not ("bitter" in symptoms and "sour" in symptoms)
   # Nota: can both be present (complex extraction) — just flag, not block
7. retrieval_quality: retrieval_metadata["avg_semantic_score"] >= 0.30

=== Post-generation checks (on coaching_text) ===
def evaluate_coaching(coaching_text: str, diagnostic: DiagnosticResult) -> dict:

1. mentions_root_cause: any(rc.hypothesis in coaching_text.lower() 
                             for rc in diagnostic.root_causes[:2])
2. has_specific_action: coaching_text contains at least one measurement or direction
   (regex: \d+|\bfiner\b|\bcoarser\b|\bplus\b|\bmoins\b|\baffiner\b|\baugmenter\b)
3. has_validation_test: coaching_text contains "tester", "essayer", "noter", 
   "observer", "vérifier", "see if", "check"
4. appropriate_length: 150 <= len(coaching_text.split()) <= 600
5. no_dangerous_phrases: not any phrase from FORBIDDEN_PHRASES list
6. language_coherent: if input is French, output is mostly French; same for English

FORBIDDEN_PHRASES = [
    "ignore si tu as mal", "force through",
    # No physical safety risk in coffee — but flag these:
    "jette la machine", "n'utilise jamais",  # too extreme advice
    "ça ne marchera jamais",  # demotivating absolute
]

Output format (combined):
{
  "pre_checks": {check_name: bool, ...},
  "post_checks": {check_name: bool, ...},
  "overall_score": float,  # ratio checks passing
  "warnings": [str],
  "verdict": "ready" | "review" | "blocked"
  # blocked if: symptoms_detected=False OR root_causes_present=False
}

Tests in tests/test_coaching_evaluator.py:
  - test_perfect_session: all checks pass
  - test_too_vague: raw_problem="mon café n'est pas bon" → symptoms_detected=False → blocked
  - test_coaching_too_short: coaching_text < 150 words → appropriate_length=False
  - test_mentions_cause: verify root cause appears in coaching text
```

---

### PHASE 9 — CoachingGenerator LLM (Jour 5)

**Prompt Claude Code #9**

```
Create homebarista/coaching_generator.py implementing CoachingGenerator.
Model: claude-haiku-3. Max output tokens: 1200.

System prompt (inline, no RAG for the rules themselves):
---
You are HomeBarista Coach, an expert barista coach who helps home coffee 
enthusiasts diagnose and fix their coffee problems.

You write coaching instructions that are:
- Specific and actionable (with exact adjustments: "1 notch finer", "93°C", "28 seconds")
- Grounded in extraction science (explain WHY not just WHAT)
- Adapted to the user's machine and level
- Encouraging but honest (don't over-promise)
- Structured: diagnosis → root cause → action plan → how to validate

You receive:
1. The user's brewing context (machine, beans, problem)
2. The diagnostic result (root causes, intervention plan)
3. Relevant passages retrieved from expert barista content (James Hoffmann, SCA, etc.)

CRITICAL RULES:
- Always explain the science behind the problem in 1-2 sentences
- Always give specific parameters when adjusting (not just "finer" but "try 1 notch finer")
- Always end with a validation test: "If this works, you should notice..."
- Always acknowledge if the problem could have multiple causes
- If beans are likely stale (>3 weeks since roast), mention this FIRST
- Never say coffee is "ruined" or "unfixable" — every problem has a solution
- Match the language of the user's input (French if they wrote in French)

FORBIDDEN:
- Vague advice without measurements
- More than 4-5 action steps (keep it manageable)
- Technical jargon without explanation (TDS, EY — only if user seems advanced)
---

Method: generate_coaching(
    session: CoachingSession,  # contains context + diagnostic + retrieved docs
    style: Literal["detailed", "concise", "technical"]
) -> CoachingSession

Build prompt:
  User context section: machine, beans, problem, measured parameters if known
  Diagnostic section: top root cause + probability, secondary cause
  Retrieved knowledge section: paste top 3 retrieved chunks (most relevant passages)
  Intervention section: ordered action plan from DiagnosticPlanner
  Style instruction: "detailed/concise/technical explanation"

Parse output:
  coaching_text: full coaching response
  follow_up_questions: extract suggested next steps as list
    (regex: "si ... essaie", "next step", "si ça ne marche pas")

Tests in tests/test_coaching_generator.py:
  - Mock Anthropic client
  - Verify system prompt contains "specific" and "validation"
  - Verify output mentions at least one number (specific adjustment)
  - Verify forbidden patterns absent
  - Test both French and English input → output language matches
```

---

### PHASE 10 — Retrieval Evaluation (Jour 5-6)

**Prompt Claude Code #10**

```
Create evals/run_retrieval_eval.py and data/eval_dataset.json.

=== data/eval_dataset.json ===
20 annotated queries covering the main diagnostic scenarios:

[
  {
    "query_id": "q001",
    "problem": "My DeLonghi Dinamica makes bitter espresso",
    "machine_type": "super_automatic",
    "symptoms": ["bitter"],
    "relevant_chunk_ids": ["hoffmann_bitter_espresso_chunk_3", 
                           "barista_hustle_extraction_chunk_7"],
    "non_relevant_chunk_ids": ["hoffmann_origin_chunk_2"]
  },
  {
    "query_id": "q002", 
    "problem": "Ma moka donne un café qui a un goût de brûlé",
    "machine_type": "moka",
    "symptoms": ["bitter"],
    ...
  },
  ...
]

Cover: 
- 5 espresso/super_automatic scenarios (bitter, sour, weak, thin crema, channeling)
- 4 moka scenarios (bitter, sour, weak, burnt)
- 4 filter method scenarios (V60, Aeropress)
- 4 optimization scenarios (not troubleshooting)
- 3 multi-symptom edge cases

=== run_retrieval_eval.py ===
For each query:
1. Run Retriever.retrieve() with the brewing context
2. Compute:
   - Precision@5: % of top-5 chunks in relevant_chunk_ids
   - Precision@10: % of top-10 in relevant_chunk_ids  
   - Recall@10: % of relevant chunks found in top-10
   - MRR: 1/rank of first relevant chunk
   - Hit Rate@5: 1 if any relevant chunk in top-5

Aggregate + save to evals/results/retrieval_eval_{timestamp}.json
Print formatted table.

Thresholds: Precision@5 >= 0.40, Hit Rate@5 >= 0.60, MRR >= 0.45
(Same as FitFlow — standard RAG portfolio targets)

Run: python -m evals.run_retrieval_eval [--demo]
```

---

### PHASE 11 — RAG Evaluation (Jour 6)

**Prompt Claude Code #11**

```
Create evals/run_rag_eval.py.

=== A. Structural evaluation (deterministic) ===
Run full pipeline on all 20 eval queries (demo mode).
For each session, run CoachingEvaluator (pre + post checks).
Aggregate: % sessions "ready", mean overall_score, failed_checks_distribution.

=== B. LLM judge evaluation (optional, --llm-judge flag) ===
For each generated coaching session, ask claude-haiku:
"Rate this barista coaching response on 4 dimensions (1-5):
 1. Specificity: does it give precise adjustments (measurements, directions)?
 2. Science: does it explain WHY the problem occurs?
 3. Actionability: can the user immediately apply the advice?
 4. Completeness: does it cover diagnosis, fix, AND validation?
 Respond in JSON: {specificity, science, actionability, completeness}"

Aggregate mean scores per dimension.

=== Output ===
evals/results/rag_eval_{timestamp}.json:
{
  "structural": {
    "sessions_evaluated": 20,
    "ready_rate": float,
    "mean_score": float,
    "failed_checks": dict
  },
  "llm_judge": {...} or null,
  "verdict": "PASS" | "FAIL",
  "thresholds": {"ready_rate": 0.75, "mean_score": 0.70}
}
```

---

### PHASE 12 — Pipeline + Streamlit (Jour 6-7)

**Prompt Claude Code #12 — Pipeline**

```
Create homebarista/pipeline.py as the single entry point.

async def run_pipeline(
    raw_problem: str,
    coach_style: str = "detailed",  # "detailed" | "concise" | "technical"
    demo_mode: bool = True
) -> dict:

Steps:
1. SymptomExtractor.extract(raw_problem) → BrewingContext
2. DiagnosticPlanner.diagnose(context) → DiagnosticResult
3. If diagnostic.diagnostic_confidence < 0.15:
   raise ValueError("Peux-tu préciser ta machine, tes symptômes et tes paramètres ?")
4. Retriever.retrieve(context, diagnostic) → retrieved_chunks
5. Build CoachingSession(context, diagnostic, retrieval_metadata)
6. CoachingEvaluator.evaluate_diagnostic(diagnostic) → pre_eval
7. If verdict == "blocked": raise ValueError with warnings
8. CoachingGenerator.generate_coaching(session, coach_style) → session
9. CoachingEvaluator.evaluate_coaching(session.coaching_text) → post_eval
10. Log to logs/sessions.jsonl
11. Return {session_dict, evaluation_dict, retrieval_metadata}
```

**Prompt Claude Code #13 — Streamlit UI**

```
Create app/streamlit_app.py for HomeBarista Coach.

Layout:

SIDEBAR:
  - Coach style: Détaillé / Concis / Technique
  - Demo mode toggle (ON by default)
  - "Exemples de problèmes" expander:
    [Cliquer pour remplir] buttons for 5 example problems:
    • "DeLonghi Dinamica, espresso trop amer, extraction 28s"
    • "Moka pot, café brûlé, trop fort"
    • "V60, Ethiopian light roast, trop acide"
    • "Breville barista, crème trop fine"
    • "Aeropress, café fade, mouture grossière"

MAIN:
  - Title + subtitle
  - Large text_area: "Décris ta machine, ton café et ton problème"
    placeholder: "ex: DeLonghi Dinamica, grains éthiopiens, espresso trop acide, 22 secondes"
  - Button: "Diagnostiquer ☕"
  
  After generation:
  - Machine detected badge (colored): 🖤 Super-Automatique / Moka / V60 / ...
  - Confidence score: "Diagnostic confidence: 73%"
  - Evaluation badge: ✅ Coaching prêt / ⚠️ À vérifier / 🚫 Bloqué
  
  Expander "🔍 Diagnostic":
    - Symptoms detected: [list]
    - Root cause #1 (probability %)
    - Root cause #2 (probability %)
    - Intervention plan (numbered list)
  
  Main coaching text (st.markdown, styled)
  
  Expander "📚 Sources utilisées":
    - Top 3 retrieved chunks with: title, channel, url, relevance score
  
  Expander "🔬 Évaluation technique":
    - Checklist of evaluator checks ✅/❌
    - Overall score X/Y

KEY UX: demo mode must work without any API key.
The "Exemples de problèmes" buttons are critical — a recruiter must be able
to test the app in 10 seconds without typing anything.
```

---

## 9. Ordre de développement

```
Jour 1  : Phase 0 (setup) + Phase 1 (SymptomExtractor) + tests
Jour 2  : Phase 2 (YouTube + Transcripts) + Phase 3 (ContentClassifier)
Jour 3  : Phase 4 (Embedder + ChromaDB) + Phase 5 (Ingestion pipeline)
Jour 4  : Phase 6 (Retriever) + Phase 7 (DiagnosticPlanner) + tests
Jour 5  : Phase 8 (CoachingEvaluator) + Phase 9 (CoachingGenerator)
Jour 6  : Phase 10 (Retrieval Eval) + Phase 11 (RAG Eval)
Jour 7  : Phase 12 (Pipeline + Streamlit) + déploiement Streamlit Cloud
Jour 8  : Tests end-to-end + bugfixes + README + demo recording
```

**Règle d'or : ne pas passer à la phase suivante si les tests de la phase en cours échouent.**

---

## 10. Ce que ce projet démontre pour ton repositionnement

| Compétence | Comment HomeBarista la démontre |
|------------|--------------------------------|
| Ingestion pipeline | YouTube API + transcripts + classification + chunking + embedding + vector store |
| RAG pipeline | Symptômes libres → diagnostic → retrieval ciblé → LLM coaching |
| Chunking strategy | Passage-level (pas vidéo-level) — justification et implémentation |
| Retrieval evaluation | Precision@k, MRR, Hit Rate sur dataset annoté 20 queries |
| RAG evaluation | Évaluation structurelle + LLM judge qualité coaching |
| Prompt engineering | System prompt structuré, science-aware, safety guardrails, style variants |
| Moteur déterministe | DiagnosticPlanner — séparation stricte règles / LLM |
| Data quality | ContentClassifier avec confidence, chunking propre, metadata filtres |
| Testing | pytest sur chaque composant, CI-compatible en demo mode |
| Analytics Engineering mindset | Moteur de règles explicite, évaluation reproductible, séparation responsabilités |

**Angle de présentation en entretien :**
> "J'ai construit un système RAG de diagnostic café. Le RAG récupère les passages 
> les plus pertinents depuis un corpus de 300+ chunks de contenu barista expert — 
> James Hoffmann, SCA, World Barista Championship. Mais le diagnostic lui-même est 
> déterministe : un moteur de règles mappe les symptômes aux causes racines avec des 
> probabilités ajustées par le contexte. Le LLM ne décide rien sur le diagnostic — 
> il formule uniquement le coaching. C'est la séparation que j'aurais dans un vrai 
> système de production : logique métier déterministe, LLM pour la formulation."

**Pourquoi aucun concurrent dans Zoomcamp 2025 :**
- 0 projet café / barista dans les 157 projets recensés
- Le pattern "diagnostic RAG" valide par Plant Disease RAG (22/25 pts)
- Corpus riche et spécialisé (James Hoffmann seul = 80h de contenu technique)
- Demo immédiate et mémorable pour un recruteur

---

## 11. Checklist soumission LLM Zoomcamp

- [ ] Problem description dans README (2-3 paragraphes)
- [ ] RAG flow documenté (diagram + explication chunking strategy)
- [ ] Retrieval evaluation : Precision@5, MRR, Hit Rate documentés
- [ ] RAG evaluation : scores structurels + optionnel LLM judge  
- [ ] UI déployée et accessible (Streamlit Cloud)
- [ ] Ingestion pipeline : commande documentée + ingestion_report.json sample
- [ ] Demo mode sans API key fonctionnel + exemples pré-remplis
- [ ] Tests pytest passants en CI (demo mode)
- [ ] Monitoring (bonus) : logs sessions.jsonl
- [ ] Docker Compose (bonus)

---

## 12. README — Message exact

```markdown
# HomeBarista Coach ☕

> AI-powered barista coach that diagnoses your coffee problems and guides 
> you to the perfect cup — whatever your machine.

**Tell HomeBarista your machine, your beans, and what's wrong.**  
HomeBarista diagnoses the root cause, retrieves expert knowledge from a 
curated corpus (James Hoffmann, SCA, World Barista Championship talks), 
and writes a precise, science-backed coaching plan.

"DeLonghi Dinamica, Ethiopian light roast, espresso tastes sour, 22 seconds 
extraction." → Root cause identified, 3 targeted adjustments, validation test.

Works with: super-automatic machines, moka pots, V60, Aeropress, espresso.  
No barista experience needed. No vague advice. Just the fix.

---

Built for LLM Zoomcamp (DataTalks.Club) — demonstrating a production-grade 
RAG pipeline: YouTube ingestion → passage-level chunking → ChromaDB → 
diagnostic retrieval → LLM coaching generation → structured evaluation.

**0 similar projects in the LLM Zoomcamp 2025 cohort.**
```

---

*ULTRAPLAN HomeBarista Coach — Ellie Pascaud — Mai 2026*
*Aucun concurrent direct dans LLM Zoomcamp 2025 (157 projets analysés)*
