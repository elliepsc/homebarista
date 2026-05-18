# HOMEBARISTA COACH — ULTRAPLAN V2
## RAG-powered coffee diagnostic & agentic coaching assistant

> Version : 2.0 — Mai 2026
> Contexte : LLM Zoomcamp (DataTalks.Club) + portfolio Data/AI Engineering
> Objectif : projet portfolio démontrant un RAG pipeline complet + agentic loop, évaluable, déployable
> Thématique : café / barista / extraction

---

## GUIDE D'UTILISATION POUR CLAUDE CODE

Ce fichier est le **single source of truth** du projet. Il est structuré pour être chargé dans Claude Code et exécuté phase par phase.

### Statut des fichiers

| Fichier | Statut | Action Claude Code |
|---------|--------|-------------------|
| `homebarista/__init__.py` | ✅ Implémenté | Ne pas modifier |
| `homebarista/models.py` | ✅ Implémenté | Lire + ne pas modifier (voir Section 4) |
| `homebarista/youtube_client.py` | ❌ À créer | Phase 2 |
| `homebarista/transcript_fetcher.py` | ❌ À créer | Phase 2 |
| `homebarista/content_classifier.py` | ❌ À créer | Phase 3 |
| `homebarista/transcript_preprocessor.py` | ✅ Implémenté | Lire + ne pas modifier |
| `homebarista/embedder.py` | ✅ Implémenté | Lire + ne pas modifier |
| `homebarista/vector_store.py` | ✅ Implémenté | Lire + ne pas modifier |
| `homebarista/symptom_extractor.py` | ✅ Implémenté | Lire + ne pas modifier |
| `homebarista/diagnostic_planner.py` | ✅ Implémenté | Lire + ne pas modifier |
| `homebarista/retriever.py` | ✅ Implémenté | Lire + ne pas modifier |
| `homebarista/coaching_evaluator.py` | ✅ Implémenté | Lire + ne pas modifier |
| `homebarista/agent.py` | ✅ Implémenté | Lire + ne pas modifier |
| `homebarista/pipeline.py` | ✅ Implémenté | Lire + ne pas modifier |
| `ingestion/run_ingestion.py` | ✅ Implémenté | Lire + ne pas modifier |
| `data/mock_documents.json` | ❌ À créer | Phase 0 |
| `ingestion/channels.yaml` | ❌ À créer | Phase 0 |
| `evals/run_retrieval_eval.py` | ❌ À créer | Phase 10 |
| `evals/run_rag_eval.py` | ❌ À créer | Phase 11 |
| `app/streamlit_app.py` | ❌ À créer | Phase 12 |

**Règle absolue** : avant toute phase, lire les fichiers ✅ déjà implémentés pour comprendre les interfaces. Ne jamais les réécrire.

---

## 0. Ce qu'on construit et pourquoi

**HomeBarista Coach** est un coach barista IA qui diagnostique les problèmes d'extraction et guide l'utilisateur vers un café parfait, quelle que soit sa machine.

L'utilisateur dit :
> "DeLonghi Dinamica, Ethiopian light roast, my espresso tastes sour, 22 second extraction."

HomeBarista Coach :
1. **Extrait** les symptômes, machine, paramètres depuis la description libre
2. **Diagnostique** les causes racines via un moteur déterministe (jamais le LLM)
3. **Récupère** les passages pertinents depuis la knowledge base (James Hoffmann, SCA, WBC)
4. **Décide** (via tool-use agent loop) s'il a besoin de plus d'informations ou peut coacher directement
5. **Génère** un plan d'action coaching précis, science-backed
6. **Valide** la qualité du coaching avec des checks déterministes

**Pourquoi "agentic" est justifié ici** (distinction critique en entretien) :
- Le LLM n'exécute pas des étapes dans un ordre fixe — il **choisit** quels tools appeler
- Il peut décider de **demander une clarification** si le contexte est insuffisant
- Il peut **re-retriever** avec une query différente si les premiers résultats sont faibles
- Il peut **re-générer** le coaching si la validation échoue
- C'est un vrai loop raisonner → agir → raisonner, pas un pipeline linéaire déguisé

---

## 1. Architecture complète

```
┌─────────────────────────────────────────────────────────────┐
│                    OFFLINE PIPELINE                         │
│  (une fois + scheduled refresh)                             │
│                                                             │
│  YouTube Data API v3 → métadonnées vidéos                   │
│         ↓                                                   │
│  youtube-transcript-api → transcripts bruts                 │
│         ↓                                                   │
│  ContentClassifier (keywords + claude-haiku backup)         │
│  → domain, method, difficulty, classification_confidence    │
│  [FILTRE: confidence < 0.4 → skip]                          │
│         ↓                                                   │
│  TranscriptPreprocessor                                     │
│  → strip timestamps, remove fillers, repair boundaries      │
│  → is_informative() filter (skip non-coffee content)        │
│         ↓                                                   │
│  Embedder (sentence-aware chunking via preprocessor)        │
│  → chunks 400 tokens, overlap 80, IDs SHA-256 stables       │
│  → sentence-transformers/all-MiniLM-L6-v2 (384 dims)        │
│         ↓                                                   │
│  VectorStore (ChromaDB persistent)                          │
│  → upsert par chunk_id (idempotent)                         │
│  → export snapshot pour déploiement Streamlit Cloud         │
│                                                             │
│  Checkpointing : ingestion/progress.json (par vidéo)        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              ONLINE — AGENTIC LOOP                          │
│  (HomeBaristaAgent avec Claude tool_use)                    │
│                                                             │
│  User input                                                 │
│         ↓                                                   │
│  Claude décide → tool: extract_symptoms                     │
│  → SymptomExtractor (rules + LLM fallback si needed)        │
│  → DiagnosticPlanner (déterministe, jamais LLM)             │
│  → BrewingContext + DiagnosticResult                        │
│         ↓                                                   │
│  [ROUTING sur DiagnosticResult.goal]                        │
│                                                             │
│  goal == "general" ─────────────────────────────────────┐  │
│  (error code / purchase / science / recipe)             │  │
│  Claude décide → tool: answer_general_question          │  │
│  → LLM direct call, pas de RAG                          │  │
│  → scope preamble par question_type                     │  │
│  → status = "coaching" avec LLM knowledge answer        │  │
│                                                         │  │
│  goal in troubleshoot/optimize/learn/explore ◄──────────┘  │
│         ↓                                                   │
│  Claude décide → tool: ask_clarification (si conf < 0.25)   │
│  OU                                                         │
│  Claude décide → tool: retrieve_knowledge                   │
│  → Retriever: embed query → ChromaDB (top-15)               │
│  → Cross-encoder re-rank (ms-marco-MiniLM-L-12-v2)          │
│  → top-10 chunks                                            │
│         ↓                                                   │
│  Claude décide → tool: generate_coaching                    │
│  → CoachingGenerator (focused prompt, pas tool-use)         │
│         ↓                                                   │
│  Claude décide → tool: validate_coaching                    │
│  → CoachingEvaluator (déterministe)                         │
│  → si fail → Claude peut re-générer (1 fois max)            │
│         ↓                                                   │
│  stop_reason == end_turn → coaching final                   │
│         ↓                                                   │
│  Logger → logs/sessions.jsonl                               │
│         ↓                                                   │
│  Streamlit UI (multi-turn via st.session_state)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Stack technique

| Brique | Outil | Justification |
|--------|-------|---------------|
| YouTube metadata | `google-api-python-client` | Officiel, stable, 10k quota/day |
| Transcripts | `youtube-transcript-api` | Fiable, sans quota |
| Transcript cleaning | `nltk` (punkt_tab) | Sentence-aware chunking — évite les chunks coupés en pleine phrase |
| Content classification | Keyword scoring + `claude-haiku-3-5` backup | Rapide, coût minimal |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Gratuit, local, English corpus + English queries |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-12-v2` | Standard industrie, local, documenté BEIR benchmark |
| Vector store | ChromaDB (persistent + snapshot) | Local dev + déployable via snapshot commité |
| LLM agent | `anthropic` SDK — Claude Haiku 3.5 tool_use | Tool-use loop : le LLM choisit les tools |
| Moteur diagnostic | Python pur (dataclasses + règles + machine capability map) | Déterministe, testable, jamais le LLM |
| Évaluation | Python pur + 50 queries synthétiques | Reproductible, CI-compatible |
| UI | Streamlit + `st.session_state` (multi-turn) | Deploy Streamlit Cloud, conversation state |
| Tests | pytest | Coverage sur moteur + classifier + evaluator |
| Dépendances | uv + pyproject.toml | Standard moderne |
| Containerisation | Docker Compose (bonus) | +1 Zoomcamp |

**Décisions clés vs. V1 :**
- Cross-encoder **remplace** le composite score arbitraire (0.6/0.25/0.15)
- `nltk` **ajouté** pour sentence-aware chunking (transcripts sans ponctuation)
- ChromaDB snapshot **résout** l'incompatibilité Streamlit Cloud filesystem éphémère
- Chunk IDs **SHA-256** au lieu de positionnels (eval dataset stable entre ré-ingestions)
- Agent tool-use **remplace** le pipeline linéaire (vrai "agentic")
- `symptoms_detected` **ajouté** à `BrewingContext` (interface entre SymptomExtractor et DiagnosticPlanner)
- `goal="general"` **ajouté** : détecte error codes, purchase advice, coffee science, recipe requests → route vers le 6ème tool sans RAG
- **6ème tool `answer_general_question`** dans l'agent : LLM direct call avec scope preamble par question_type (error_code / purchase_advice / coffee_science / recipe_request / general) — zéro zone morte
- **CoachingEvaluator** étend `non_troubleshoot` à `("learn", "optimize", "explore", "general")` — jamais bloqué pour des questions hors-scope
- **Anthropic client lazy init** dans SymptomExtractor : pas de connexion réseau en demo_mode (évite les crashes sandbox)

---

## 2b. Couverture des intentions utilisateur — zéro zone morte

Chaque type de question utilisateur a **une réponse définie**. Pas de silence, pas de crash.

| Type de question | Exemple | Goal détecté | Flow | Réponse |
|-----------------|---------|--------------|------|---------|
| Problème de goût | "mon espresso est amer" | troubleshoot | extract → retrieve → generate → validate | Coaching RAG-backed |
| Specs sans symptôme | "18g, 93°C, 28s, comment améliorer ?" | optimize | extract → retrieve → generate | Analyse paramétrique |
| Apprendre | "je suis débutant, comment faire" | learn | extract → generate (principes) | Principes fondamentaux |
| Explorer | "je veux essayer un Ethiopian" | explore | extract → retrieve → generate | Guide découverte |
| Trop vague | "mon café est mauvais" | troubleshoot | extract → ask_clarification | Question ciblée |
| **Error code machine** | "mon DeLonghi affiche E5" | **general** | extract → **answer_general_question** | LLM knowledge + disclaimer manuel |
| **Conseil d'achat** | "Breville ou Gaggia Classic ?" | **general** | extract → **answer_general_question** | Critères de sélection, pas d'endorsement |
| **Question scientifique** | "c'est quoi le TDS ?" | **general** | extract → **answer_general_question** | Réponse directe LLM |
| **Demande de recette** | "donne-moi une recette V60" | **general** | extract → **answer_general_question** | Recette baseline + instructions d'ajustement |

**Règle implémentée :** si `goal == "general"`, l'agent appelle `answer_general_question` directement après `extract_symptoms` — sans passer par `retrieve_knowledge` ni `generate_coaching`. Le `CoachingEvaluator` ne bloque jamais les goals non-troubleshoot.

**Limites assumées et documentées (hors scope) :**
- Diagnostic d'erreurs hardware nécessitant un technicien → réponse LLM + "consultez le SAV"
- Recommandations d'achat précises avec prix → réponse LLM avec critères généraux seulement
- Recettes pour machines inconnues → recette générique pour la méthode demandée

---

## 3. Structure du repo

```
homebarista/
├── README.md
├── pyproject.toml
├── docker-compose.yml               (bonus)
├── .env.example
│
├── homebarista/
│   ├── __init__.py
│   ├── models.py                    ← À CRÉER (Phase 0)
│   ├── youtube_client.py            ← À CRÉER (Phase 2)
│   ├── transcript_fetcher.py        ← À CRÉER (Phase 2)
│   ├── content_classifier.py        ← À CRÉER (Phase 3)
│   ├── transcript_preprocessor.py   ✅ IMPLÉMENTÉ
│   ├── embedder.py                  ✅ IMPLÉMENTÉ
│   ├── vector_store.py              ✅ IMPLÉMENTÉ
│   ├── symptom_extractor.py         ✅ IMPLÉMENTÉ
│   ├── diagnostic_planner.py        ✅ IMPLÉMENTÉ
│   ├── retriever.py                 ✅ IMPLÉMENTÉ
│   ├── coaching_evaluator.py        ✅ IMPLÉMENTÉ
│   ├── agent.py                     ✅ IMPLÉMENTÉ
│   └── pipeline.py                  ✅ IMPLÉMENTÉ
│
├── ingestion/
│   ├── run_ingestion.py             ✅ IMPLÉMENTÉ
│   ├── channels.yaml                ← À CRÉER (Phase 0)
│   ├── progress.json                (généré au runtime, gitignored)
│   └── ingestion_report.json        (généré au runtime)
│
├── data/
│   ├── chroma_db/                   (gitignored — généré en local)
│   ├── chroma_snapshot/             ← généré par `--export`, COMMITÉ pour Streamlit Cloud
│   ├── mock_documents.json          ← À CRÉER (Phase 0)
│   └── eval_dataset.json            ← généré par Phase 10 (synthétique)
│
├── evals/
│   ├── run_retrieval_eval.py        ← À CRÉER (Phase 10)
│   ├── run_rag_eval.py              ← À CRÉER (Phase 11)
│   └── results/
│
├── tests/
│   ├── test_symptom_extractor.py    ← À CRÉER (Phase 1)
│   ├── test_diagnostic_planner.py   ← À CRÉER (Phase 8)
│   ├── test_coaching_evaluator.py   ← À CRÉER (Phase 9)
│   ├── test_retriever.py            ← À CRÉER (Phase 7)
│   └── fixtures/
│       └── mock_documents.json      (symlink ou copie)
│
├── app/
│   └── streamlit_app.py             ← À CRÉER (Phase 12)
│
└── logs/
    └── sessions.jsonl               (gitignored)
```

---

## 4. Modèle de données — models.py

**ATTENTION** : `BrewingContext` a un champ `symptoms_detected` qui N'EXISTE PAS dans la V1.
Ce champ est la **clé d'interface** entre `SymptomExtractor` et `DiagnosticPlanner`.
Tous les fichiers ✅ déjà implémentés en dépendent. Ne pas l'omettre.

```python
# homebarista/models.py
from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class BrewingContext:
    """Ce que l'utilisateur nous dit sur sa situation."""
    machine_type: Literal[
        "super_automatic", "semi_automatic", "manual_lever",
        "moka", "v60", "aeropress", "french_press", "nespresso", "unknown"
    ]
    machine_model: Optional[str] = None
    bean_origin: Optional[str] = None
    roast_level: Optional[Literal["light", "medium", "dark", "unknown"]] = None
    bean_freshness_days: Optional[int] = None
    grind_size: Optional[str] = None
    dose_grams: Optional[float] = None
    water_temp_celsius: Optional[float] = None
    extraction_time_seconds: Optional[int] = None
    water_ratio: Optional[str] = None
    raw_problem: str = ""
    # ── NOUVEAU V2 ── 5 goals possibles (pas 4)
    # "troubleshoot" : l'utilisateur a un problème de goût
    # "optimize"     : l'utilisateur a des specs et veut affiner
    # "learn"        : débutant, principes généraux
    # "explore"      : veut essayer quelque chose de nouveau
    # "general"      : hors-scope (error code, achat, science, recette)
    #                  → route vers answer_general_question, pas de RAG
    goal: Literal["troubleshoot", "optimize", "explore", "learn", "general"] = "troubleshoot"

    # ── NOUVEAU V2 ── Interface SymptomExtractor → DiagnosticPlanner
    # Peuplé par SymptomExtractor.extract(), consommé par DiagnosticPlanner.diagnose()
    # Valeurs possibles (12 symptômes) :
    # bitter, sour, weak_bland, thin_crema, channeling, bitter_and_sour,
    # too_strong, astringent, flat_no_aroma, inconsistent,
    # too_slow_extraction, too_fast_extraction
    symptoms_detected: list[str] = field(default_factory=list)


@dataclass
class RootCause:
    """Une cause racine probable du problème."""
    hypothesis: str
    # IMPORTANT : renommé 'heuristic_weight' dans la logique, mais le champ
    # s'appelle 'probability' pour compatibilité dataclass → asdict() → JSON.
    # Ces valeurs sont des POIDS HEURISTIQUES basés sur la littérature
    # d'extraction (Scott Rao, SCA, Barista Hustle) — pas des probabilités
    # empiriquement calibrées. Les présenter comme tels en entretien.
    probability: float
    evidence: str
    parameter_affected: Optional[str] = None


@dataclass
class Intervention:
    """Une action corrective à effectuer."""
    step: int
    action: str
    parameter: Optional[str] = None
    direction: Optional[str] = None
    magnitude: Optional[str] = None
    expected_result: str = ""
    validation_test: str = ""
    priority: Literal["critical", "high", "medium", "low"] = "high"


@dataclass
class DiagnosticResult:
    """Le résultat du moteur de diagnostic déterministe."""
    symptoms: list[str]
    root_causes: list[RootCause]
    intervention_plan: list[Intervention]
    diagnostic_confidence: float
    method_detected: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class CoachingSession:
    """La session de coaching complète."""
    context: BrewingContext
    diagnostic: DiagnosticResult
    coaching_text: str = ""
    follow_up_questions: list[str] = field(default_factory=list)
    evaluation: dict = field(default_factory=dict)
    retrieval_metadata: dict = field(default_factory=dict)
    session_id: str = ""
```

---

## 5. Corpus à indexer — channels.yaml

```yaml
# ingestion/channels.yaml
channels:
  - id: UCMb0O2CdPBNi-QqPk5T3gsQ
    name: James Hoffmann
    tags: [espresso, filter, grind, origin, equipment, technique]
    priority: critical
    max_videos: 80

  - id: UCbS5dZ7rVAGBWMQGbMBZCcg
    name: Barista Hustle
    tags: [extraction, sca, professional, technique, science]
    priority: critical
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

playlists:
  - id: PLynbSsRRLfnoR4CZrRlhvkbY1xPQ5e7Py
    name: "James Hoffmann - How to make better coffee"
    tags: [beginner, technique, troubleshooting]

  - id: PLynbSsRRLfnoV1HiYeF0n3CwDo2gm59Gx
    name: "James Hoffmann - Espresso Machines"
    tags: [espresso, machine, equipment, super_automatic]
```

---

## 6. Plan de développement

### Ordre d'exécution

```
Phase 0  : Setup + models.py + mock data + channels.yaml
Phase 1  : Tests des fichiers déjà implémentés (smoke tests)
Phase 2  : YouTubeClient + TranscriptFetcher
Phase 3  : ContentClassifier
Phase 4  : Ingestion run (--demo) — valider pipeline complet en mock
Phase 5  : Ingestion run (--live) — vraie ingestion YouTube
Phase 6  : Export snapshot ChromaDB
Phase 7  : Tests unitaires (pytest)
Phase 8  : Retrieval Evaluation (50 queries synthétiques)
Phase 9  : RAG Evaluation (structural + LLM judge)
Phase 10 : Streamlit App
Phase 11 : Déploiement Streamlit Cloud
Phase 12 : Tests end-to-end + README + demo recording
```

**Règle d'or** : ne pas passer à la phase suivante si les tests de la phase en cours échouent.

---

### PHASE 0 — Setup & fondations

```
Lire d'abord tous les fichiers ✅ implémentés pour comprendre les interfaces.

1. Créer pyproject.toml avec ces dépendances exactes :
   [project]
   name = "homebarista"
   version = "0.1.0"
   requires-python = ">=3.11"
   dependencies = [
     "google-api-python-client>=2.0",
     "youtube-transcript-api>=0.6",
     "sentence-transformers>=2.7",   # inclut CrossEncoder
     "chromadb>=0.5",
     "anthropic>=0.25",
     "streamlit>=1.35",
     "pytest>=8.0",
     "python-dotenv>=1.0",
     "pydantic>=2.0",
     "httpx>=0.27",
     "pyyaml>=6.0",
     "nltk>=3.8",                    # NOUVEAU V2 — sentence tokenization
   ]

2. Créer .env.example :
   YOUTUBE_API_KEY=
   ANTHROPIC_API_KEY=
   CHROMA_PERSIST_DIR=data/chroma_db
   DEMO_MODE=true

3. ✅ homebarista/models.py DÉJÀ CRÉÉ — ne pas recréer.
   Lire le fichier existant pour vérifier les champs.
   ATTENTION : le champ `goal` accepte maintenant 5 valeurs :
   "troubleshoot" | "optimize" | "explore" | "learn" | "general"

4. Créer ingestion/channels.yaml avec le contenu de la Section 5.

5. Créer data/mock_documents.json avec 40 documents mock :
   - 10 docs sur espresso extraction (bitter, sour, channeling, crema)
   - 8 docs sur grind and grinders
   - 8 docs sur super-automatic machines (DeLonghi, Jura, Philips)
   - 6 docs sur moka pot technique
   - 5 docs sur filter methods (V60, Aeropress)
   - 3 docs sur coffee origins and roast levels

   Format exact de chaque document :
   {
     "source_id": "mock_001",
     "title": "Why Espresso Tastes Bitter",
     "channel": "James Hoffmann",
     "url": "https://youtube.com/watch?v=mock001",
     "domain": "troubleshooting",
     "method": "espresso",
     "difficulty": "beginner",
     "classification_confidence": 0.85,
     "content": "500-800 chars of realistic barista knowledge text in English..."
   }

   IMPORTANT : le champ s'appelle "content" dans mock_documents.json
   mais "transcript_text" dans les vrais documents YouTube.
   Le pipeline.py et run_ingestion.py gèrent les deux :
   transcript = doc.get("transcript_text", doc.get("content", ""))

6. Créer tous les __init__.py vides (sauf homebarista/__init__.py déjà créé).

7. ✅ homebarista/__init__.py DÉJÀ CRÉÉ.

8. Exécuter : uv sync
   Vérifier : python -c "from homebarista.models import BrewingContext; print('OK')"
   Vérifier : python -c "from homebarista.transcript_preprocessor import TranscriptPreprocessor; print('OK')"
   Vérifier : python -c "from homebarista.diagnostic_planner import DiagnosticPlanner; print('OK')"
   Vérifier : python -c "from homebarista.agent import HomeBaristaAgent; print('OK')"
```

---

### PHASE 1 — Tests smoke des fichiers implémentés

```
Lire les fichiers suivants (ne pas modifier) :
  - homebarista/symptom_extractor.py
  - homebarista/diagnostic_planner.py
  - homebarista/coaching_evaluator.py
  - homebarista/transcript_preprocessor.py
  - homebarista/embedder.py

Exécuter les smoke tests intégrés :
  python homebarista/transcript_preprocessor.py
  python homebarista/symptom_extractor.py    (demo_mode=True dans __main__)
  python homebarista/diagnostic_planner.py
  python homebarista/coaching_evaluator.py

Vérifier que chaque smoke test s'exécute sans erreur.

Si une erreur "ModuleNotFoundError" survient :
  - Vérifier que models.py a le champ symptoms_detected
  - Vérifier que uv sync a été exécuté
  - Vérifier que tous les __init__.py existent

Créer tests/test_smoke.py :

  import pytest
  from homebarista.models import BrewingContext
  from homebarista.symptom_extractor import SymptomExtractor
  from homebarista.diagnostic_planner import DiagnosticPlanner
  from homebarista.coaching_evaluator import CoachingEvaluator
  from homebarista.transcript_preprocessor import TranscriptPreprocessor

  def test_bitter_espresso_end_to_end():
      extractor = SymptomExtractor(demo_mode=True)
      planner = DiagnosticPlanner()
      evaluator = CoachingEvaluator()

      ctx = extractor.extract("My DeLonghi Dinamica makes bitter espresso, 28 second extraction")
      assert ctx.machine_type == "super_automatic"
      assert "bitter" in ctx.symptoms_detected
      assert ctx.extraction_time_seconds == 28

      diag = planner.diagnose(ctx)
      assert diag.diagnostic_confidence > 0.0
      assert len(diag.root_causes) > 0
      assert len(diag.intervention_plan) > 0
      # Vérifier que tamping n'est pas dans le plan (impossible sur super-auto)
      for iv in diag.intervention_plan:
          assert iv.parameter not in ("tamping", "distribution"), \
              f"Impossible intervention for super_automatic: {iv.action}"

  def test_moka_bitter_override():
      extractor = SymptomExtractor(demo_mode=True)
      planner = DiagnosticPlanner()
      ctx = extractor.extract("My moka pot makes burnt bitter coffee")
      diag = planner.diagnose(ctx)
      assert diag.root_causes[0].hypothesis == "heat_too_high"

  def test_nespresso_no_impossible_interventions():
      extractor = SymptomExtractor(demo_mode=True)
      planner = DiagnosticPlanner()
      ctx = extractor.extract("My Nespresso makes bitter coffee")
      diag = planner.diagnose(ctx)
      for iv in diag.intervention_plan:
          assert iv.parameter not in ("tamping", "distribution", "pressure"), \
              f"Impossible for Nespresso: {iv.action}"

  def test_symptom_extraction_12_types():
      extractor = SymptomExtractor(demo_mode=True)
      cases = [
          ("coffee too strong overwhelming", "too_strong"),
          ("mouth feels dry astringent", "astringent"),
          ("no aroma flat smell", "flat_no_aroma"),
          ("inconsistent results every time", "inconsistent"),
          ("no flow blocked choked", "too_slow_extraction"),
          ("too fast gushing watery shot", "too_fast_extraction"),
      ]
      for text, expected_symptom in cases:
          ctx = extractor.extract(text)
          assert expected_symptom in ctx.symptoms_detected, \
              f"Expected '{expected_symptom}' in symptoms for: '{text}'"

  def test_preprocessor_removes_timestamps():
      preprocessor = TranscriptPreprocessor()
      raw = "[00:03:45] so the grind size matters [01:22:00] especially for espresso"
      cleaned = preprocessor.clean(raw)
      assert "[" not in cleaned
      assert "grind size matters" in cleaned

  def test_chunk_ids_are_stable():
      from homebarista.embedder import Embedder
      embedder = Embedder()
      doc = {"source_id": "test_001", "title": "T", "channel": "C",
             "url": "u", "domain": "d", "method": "m", "difficulty": "b"}
      transcript = "The grind size is the most important variable in espresso. " * 20
      chunks1 = embedder.chunk_transcript(transcript, doc)
      chunks2 = embedder.chunk_transcript(transcript, doc)
      assert [c["chunk_id"] for c in chunks1] == [c["chunk_id"] for c in chunks2], \
          "Chunk IDs must be stable across identical inputs"

  Exécuter : pytest tests/test_smoke.py -v
  Tous les tests doivent passer avant de continuer.
```

---

### PHASE 2 — YouTubeClient + TranscriptFetcher

```
Lire d'abord ingestion/run_ingestion.py (✅ implémenté) pour comprendre
comment YouTubeClient et TranscriptFetcher sont utilisés.

Créer homebarista/youtube_client.py :

  class YouTubeClient:
    def __init__(self):
      self.service = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
      self.quota_used = 0

    def get_channel_videos(self, channel_id: str, max_results: int = 50) -> tuple[list[dict], int]:
      """
      Returns (videos, quota_used).
      Each video: {videoId, title, description, tags, duration_seconds, channelTitle}
      Handles pagination. Logs quota.
      Quota cost: ~100 units per search page + 1 unit per video (batched).
      """

    def get_playlist_videos(self, playlist_id: str, max_results: int = 50) -> tuple[list[dict], int]:
      """Same format as get_channel_videos."""

    def build_document_object(self, raw: dict, channel_tags: list[str]) -> dict:
      """
      Maps API response → document dict.
      Returns: source_id (=videoId), title, channel, url,
               description, tags (merged api + channel), duration_seconds
      """

  DEMO_MODE support : si DEMO_MODE=true, retourner [] sans appel API.

Créer homebarista/transcript_fetcher.py :

  class TranscriptFetcher:
    def fetch_transcript(self, video_id: str) -> tuple[str, bool]:
      """
      Returns (transcript_text, is_available).
      Language preference: ["en"] — corpus anglais uniquement.
      Retourne le transcript complet non tronqué.
      """

    def fetch_batch(self, video_ids: list[str]) -> dict[str, str]:
      """
      Rate limiting: sleep 0.3s entre chaque appel.
      Skip silencieusement les vidéos sans transcript.
      Progress toutes les 10 vidéos.
      """

  DEMO_MODE : retourner {} sans appel API.

Tests dans tests/test_youtube.py :
  - test_demo_mode_returns_empty: DEMO_MODE=true → [] sans erreur
  - test_build_document_object: structure de retour correcte
  - test_transcript_fetch_missing: vidéo sans transcript → (None, False)
```

---

### PHASE 3 — ContentClassifier

```
Créer homebarista/content_classifier.py.

INTERFACE CRITIQUE : classify() doit retourner classification_confidence.
run_ingestion.py filtre les docs avec confidence < 0.4.

  class ContentClassifier:
    def classify(self, doc: dict) -> dict:
      """
      Input : document dict (title, description, tags, transcript_text ou content)
      Output : {domain, method, difficulty, classification_confidence}

      IMPORTANT : classification_confidence est utilisé par run_ingestion.py
      pour filtrer le contenu non-café. Doit être dans [0.0, 1.0].
      confidence = top_domain_score / sum_all_scores.
      """

    def classify_batch(self, docs: list[dict]) -> list[dict]:
      """Update chaque doc avec domain, method, difficulty, classification_confidence."""

Domaines et signaux :
  extraction   : extraction, over-extracted, under-extracted, yield, TDS, EY, ratio
  grind        : grind, burr, grinder, coarseness, particle size
  machine      : machine, boiler, pressure, pump, portafilter, basket
  origin       : Ethiopian, Colombian, Brazilian, washed, natural, terroir
  method       : espresso, moka, v60, aeropress, french press, pour over
  troubleshoot : bitter, sour, acidic, bland, weak, channeling, crema
  technique    : tamping, distribution, bloom, pre-infusion, flow rate

Méthodes (même approche) :
  espresso, super_automatic, moka, v60, aeropress, french_press, general

Scoring : title × 4 + description × 2 + tags × 3 + transcript[:1000] × 1
super_automatic prioritaire sur espresso si les deux scorent haut.

Difficulté :
  beginner  : beginner, start, basic, introduction
  advanced  : SCA, TDS, EY, refractometer, barista championship, Scott Rao
  intermediate : sinon

Backup LLM (si confidence < 0.3) :
  claude-haiku-3-5 pour classifier domain + method.

Tests dans tests/test_content_classifier.py :
  - test_troubleshooting_bitter: doc sur espresso amer → domain="troubleshooting"
  - test_super_auto_priority: doc mentionne espresso ET DeLonghi → method="super_automatic"
  - test_low_confidence_flagged: doc hors-sujet → confidence < 0.4
  - test_confidence_range: toujours dans [0.0, 1.0]
```

---

### PHASE 4 — Ingestion en demo mode (validation pipeline)

```
À ce stade, tous les composants sont disponibles.
Valider le pipeline complet en mode demo avant la vraie ingestion.

Exécuter :
  python -m ingestion.run_ingestion --demo

Vérifier dans le output :
  ✓ "Loaded 40 mock documents"
  ✓ "[XX chunks] ..." pour chaque doc
  ✓ "Embedded XX/XX chunks — done."
  ✓ "Upserted XX chunks into 'barista_knowledge'"
  ✓ ingestion_report.json créé

Vérifier ingestion_report.json :
  - total_chunks_generated > 0
  - domain_distribution non vide
  - method_distribution non vide

Vérifier VectorStore :
  python -m homebarista.vector_store --stats

Si erreur "ModuleNotFoundError: nltk" :
  python -c "import nltk; nltk.download('punkt_tab')"

Si erreur "collection already exists" : normal, upsert est idempotent.
```

---

### PHASE 5 — Ingestion réelle YouTube

```
Prérequis : YOUTUBE_API_KEY valide dans .env

Démarrer par un seul canal pour tester :
  python -m ingestion.run_ingestion --channel UCMb0O2CdPBNi-QqPk5T3gsQ

Observer :
  - Quota utilisé par canal
  - Taux de transcripts disponibles (target > 70%)
  - Taux de classification_confidence >= 0.4 (target > 80%)
  - Chunks générés par vidéo (target 5-15 chunks)

Si quota YouTube épuisé (10k units/day) :
  Le checkpointing sauvegarde la progression dans ingestion/progress.json
  Relancer le lendemain : python -m ingestion.run_ingestion
  La reprise est automatique depuis la dernière vidéo traitée.

Ingestion complète (tous les canaux) :
  python -m ingestion.run_ingestion

Target final : 300-500 documents, 3000-8000 chunks.

Vérifier :
  python -m homebarista.vector_store --stats
```

---

### PHASE 6 — Export snapshot ChromaDB pour déploiement

```
CRITIQUE pour Streamlit Cloud.
Streamlit Cloud a un filesystem éphémère — ChromaDB local serait perdu
à chaque redémarrage. La solution : commiter un snapshot.

Après ingestion complète :
  python -m homebarista.vector_store --export

Cela crée data/chroma_snapshot/ (copie de data/chroma_db/).

Vérifier la taille :
  du -sh data/chroma_snapshot/

Si < 100MB : commiter directement dans git.
  Retirer data/chroma_snapshot/ du .gitignore
  git add data/chroma_snapshot/
  git commit -m "Add ChromaDB snapshot for Streamlit Cloud deployment"

Si > 100MB : utiliser Chroma Cloud (tier gratuit).
  Ajouter CHROMA_MODE=cloud dans .env
  Adapter VectorStore._init_client() pour Chroma Cloud.
  (VectorStore supporte déjà les deux modes via la variable CHROMA_PERSIST_DIR)

Déploiement sur Streamlit Cloud :
  Au démarrage, VectorStore._init_client() détecte que chroma_db/ n'existe pas
  mais que chroma_snapshot/ existe → restore automatique.
  L'app démarre avec la knowledge base complète sans ré-ingestion.
```

---

### PHASE 7 — Tests unitaires complets

```
Créer tests/test_symptom_extractor.py :

  def test_full_context():
      extractor = SymptomExtractor(demo_mode=True)
      ctx = extractor.extract(
          "DeLonghi Dinamica, Ethiopian light roast, sour espresso, 20 seconds, 18g dose"
      )
      assert ctx.machine_type == "super_automatic"
      assert "sour" in ctx.symptoms_detected
      assert ctx.extraction_time_seconds == 20
      assert ctx.dose_grams == 18.0
      assert ctx.roast_level == "light"
      assert "Ethiopian" in (ctx.bean_origin or "")

  def test_multilingual_french():
      extractor = SymptomExtractor(demo_mode=True)
      ctx = extractor.extract("ma moka donne un café très acide et amer")
      assert ctx.machine_type == "moka"
      assert "sour" in ctx.symptoms_detected

  def test_goal_detection_learn():
      extractor = SymptomExtractor(demo_mode=True)
      ctx = extractor.extract("I want to learn how to make better espresso from scratch")
      assert ctx.goal == "learn"

  def test_all_12_symptoms_recognised():
      extractor = SymptomExtractor(demo_mode=True)
      symptom_inputs = {
          "bitter": "coffee tastes bitter and harsh",
          "sour": "coffee tastes sour and acidic",
          "weak_bland": "coffee is weak and watery",
          "thin_crema": "no crema at all",
          "channeling": "channeling in portafilter",
          "too_strong": "too strong and overwhelming",
          "astringent": "mouth feels dry and astringent",
          "flat_no_aroma": "no aroma flat smell",
          "inconsistent": "inconsistent results every time",
          "too_slow_extraction": "no flow blocked choked drips",
          "too_fast_extraction": "too fast gushing watery",
      }
      for symptom, text in symptom_inputs.items():
          ctx = extractor.extract(text)
          assert symptom in ctx.symptoms_detected, f"Failed for: {symptom}"

Créer tests/test_diagnostic_planner.py :

  def test_max_4_interventions():
      planner = DiagnosticPlanner()
      ctx = BrewingContext(machine_type="semi_automatic",
                           symptoms_detected=["bitter", "thin_crema", "weak_bland"],
                           raw_problem="...", goal="troubleshoot")
      diag = planner.diagnose(ctx)
      assert len(diag.intervention_plan) <= 4

  def test_learn_mode_returns_result():
      planner = DiagnosticPlanner()
      ctx = BrewingContext(machine_type="v60", symptoms_detected=[],
                           raw_problem="want to learn", goal="learn")
      diag = planner.diagnose(ctx)
      assert len(diag.intervention_plan) > 0
      assert "Learning mode" in diag.warnings[0]

  def test_bitter_sour_cooccurrence():
      planner = DiagnosticPlanner()
      ctx = BrewingContext(machine_type="semi_automatic",
                           symptoms_detected=["bitter", "sour"],
                           raw_problem="both bitter and sour", goal="troubleshoot")
      diag = planner.diagnose(ctx)
      assert "bitter_and_sour" in diag.symptoms or "mixed" in diag.root_causes[0].hypothesis

Créer tests/test_coaching_evaluator.py :

  def test_good_coaching_passes():
      evaluator = CoachingEvaluator()
      good_text = (
          "Your espresso tastes bitter because of over-extraction. "
          "This happens when water dissolves too many bitter compounds. "
          "Fix: go 1 notch coarser on grind. Lower temperature by 2°C. "
          "You should notice the bitterness reduce within 2-3 shots."
      )
      result = evaluator.evaluate_coaching(good_text)
      assert result["verdict"] in ("pass", "warn")

  def test_vague_coaching_fails():
      evaluator = CoachingEvaluator()
      vague = "Try adjusting your coffee. It might get better."
      result = evaluator.evaluate_coaching(vague)
      assert result["verdict"] == "fail"
      assert not result["checks"]["specific_enough"]

Exécuter : pytest tests/ -v
```

---

### PHASE 8 — Retrieval Evaluation (50 queries synthétiques)

```
DÉCISION CRITIQUE V2 : 20 queries hardcodées avec chunk IDs positionnels = non défendable.
Solution : générer 50 queries synthétiques depuis le corpus indexé.
Les IDs de chunks sont maintenant SHA-256 stables → dataset persistant entre ré-ingestions.

Créer evals/run_retrieval_eval.py :

  """
  Deux modes :
  --generate : génère le dataset depuis le corpus (à faire une fois après ingestion)
  --eval     : évalue le retrieval sur le dataset existant
  --demo     : évalue sur mock ChromaDB (CI-safe)
  """

  GENERATE MODE :
    1. Charger 50 chunks aléatoires depuis ChromaDB
    2. Pour chaque chunk, appeler claude-haiku-3-5 :
       "Given this barista knowledge passage:
        '{chunk.text[:400]}'
        Write a realistic home user question that this passage directly answers.
        The question should sound like a real user describing their coffee problem.
        Respond with just the question, nothing else."
    3. Sauvegarder data/eval_dataset.json :
       [{
         "query_id": "q001",
         "synthetic_query": "...",
         "relevant_chunk_ids": [chunk_id],  # SHA-256 stable
         "machine_type": chunk.metadata.method,
         "domain": chunk.metadata.domain
       }]
    Couvrir : troubleshooting (30), optimization (10), method-specific (10)

  EVAL MODE :
    Pour chaque query dans eval_dataset.json :
    1. Créer un BrewingContext minimal avec synthetic_query comme raw_problem
    2. Exécuter SymptomExtractor + DiagnosticPlanner → DiagnosticResult
    3. Exécuter Retriever.retrieve() → chunks
    4. Calculer :
       - Hit Rate@5   : 1 si relevant_chunk_id dans top-5
       - Precision@5  : % relevant dans top-5
       - Precision@10 : % relevant dans top-10
       - MRR          : 1/rank du premier chunk pertinent
    5. Agréger + afficher tableau + sauvegarder evals/results/retrieval_{timestamp}.json

  SEUILS :
    Hit Rate@5  >= 0.55  (cible atteignable avec cross-encoder)
    Precision@5 >= 0.35
    MRR         >= 0.45

  Si les seuils ne sont pas atteints :
    - Vérifier la qualité de l'ingestion (ingestion_report.json)
    - Augmenter n_candidates dans Retriever (15 → 20)
    - Vérifier que cross-encoder est bien chargé (pas de fallback silencieux)

  Exécuter :
    python -m evals.run_retrieval_eval --generate   # une fois
    python -m evals.run_retrieval_eval --eval
    python -m evals.run_retrieval_eval --demo       # CI sans API
```

---

### PHASE 9 — RAG Evaluation

```
Créer evals/run_rag_eval.py :

  PARTIE A — Structural evaluation (déterministe, toutes les 50 queries) :
    Pour chaque query du eval_dataset.json :
    1. Exécuter pipeline.run_pipeline(query, demo_mode=False, use_agent=False)
       (mode linéaire pour la reproductibilité de l'éval)
    2. Récupérer evaluation.post dans le résultat
    3. Agréger :
       - % sessions avec verdict "pass"
       - Mean overall_score
       - Distribution des failed_checks

  PARTIE B — LLM Judge (optionnel, flag --llm-judge) :
    Utiliser claude-haiku-3-5 comme judge.
    BIAIS CONNU : même famille de modèles que le generator.
    Documenter ce biais dans les résultats.
    Pour chaque coaching généré :
    "Rate this barista coaching on 4 criteria (1-5 each):
     1. Specificity: precise measurements and adjustments?
     2. Science: explains WHY not just WHAT?
     3. Actionability: immediately applicable?
     4. Completeness: diagnosis + fix + validation test?
     Respond in JSON: {specificity, science, actionability, completeness, comment}"
    Mean scores par dimension.

  OUTPUT : evals/results/rag_eval_{timestamp}.json :
    {
      "structural": {
        "sessions_evaluated": 50,
        "pass_rate": float,
        "mean_score": float,
        "failed_checks_distribution": dict
      },
      "llm_judge": {...} or null,
      "bias_note": "LLM judge uses claude-haiku-3-5, same family as generator. Scores may be inflated.",
      "verdict": "PASS" | "FAIL",
      "thresholds": {"pass_rate": 0.70, "mean_score": 0.65}
    }

  Seuils :
    pass_rate  >= 0.70
    mean_score >= 0.65

  Exécuter :
    python -m evals.run_rag_eval
    python -m evals.run_rag_eval --llm-judge  (optionnel)
```

---

### PHASE 10 — Streamlit App (multi-turn)

```
Créer app/streamlit_app.py.

ARCHITECTURE UI :
  Utiliser st.session_state pour gérer l'état de la conversation.
  L'app supporte 3 tours minimum :
  1. Description initiale → coaching
  2. "Did that help?" → ajustement si non
  3. Clarification si l'agent l'a demandée

STRUCTURE :

SIDEBAR :
  - Coach style : Detailed / Concise / Technical (st.radio)
  - Mode : Demo (no API needed) / Live (requires API key)
  - "Example problems" expander avec 5 boutons [Click to fill] :
    • "DeLonghi Dinamica, Ethiopian light roast, espresso too sour, 20 seconds"
    • "My moka pot coffee tastes burnt and bitter"
    • "V60 pour over, too weak and bland, 15g coffee"
    • "Breville espresso, very thin crema, fresh beans"
    • "I want to learn how to make better Aeropress coffee"
  - "🔬 Technical details" expander (pour recruteurs) :
    Architecture overview, chunk count, model info

MAIN :
  st.title("HomeBarista Coach ☕")
  st.caption("Describe your machine, your beans, and what tastes wrong.")

  # Zone de conversation (multi-turn)
  if "messages" not in st.session_state:
      st.session_state.messages = []
  if "last_result" not in st.session_state:
      st.session_state.last_result = None

  # Afficher historique conversation
  for msg in st.session_state.messages:
      with st.chat_message(msg["role"]):
          st.markdown(msg["content"])

  # Input utilisateur
  user_input = st.chat_input("Describe your coffee problem...")

  if user_input:
      # Ajouter au chat
      st.session_state.messages.append({"role": "user", "content": user_input})

      # Appeler le pipeline (async via asyncio.run)
      import asyncio
      from homebarista.pipeline import run_pipeline
      result = asyncio.run(run_pipeline(
          user_input,
          coach_style=st.session_state.get("style", "detailed"),
          demo_mode=st.session_state.get("demo_mode", True),
          use_agent=not st.session_state.get("demo_mode", True),
      ))
      st.session_state.last_result = result

      # Gérer les cas
      if result["status"] == "clarification_needed":
          response = f"Before I can diagnose, I need to know: **{result['clarification_question']}**"
      elif result["status"] == "coaching":
          response = result["coaching_text"]
      else:
          response = f"⚠️ {result.get('clarification_question', 'An error occurred.')}"

      st.session_state.messages.append({"role": "assistant", "content": response})
      st.rerun()

  # Résultats techniques (si disponibles)
  if st.session_state.last_result and st.session_state.last_result["status"] == "coaching":
      result = st.session_state.last_result
      ctx = result.get("context", {})
      diag = result.get("diagnostic", {})
      eval_data = result.get("evaluation", {})

      col1, col2, col3 = st.columns(3)
      with col1:
          machine = ctx.get("machine_type", "unknown").replace("_", " ").title()
          st.metric("Machine detected", machine)
      with col2:
          conf = diag.get("diagnostic_confidence", 0)
          st.metric("Diagnostic confidence", f"{conf:.0%}")
      with col3:
          verdict = eval_data.get("overall_verdict", "—")
          icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(verdict, "—")
          st.metric("Quality check", f"{icon} {verdict}")

      with st.expander("🔍 Diagnostic details"):
          st.write("**Symptoms detected:**", ", ".join(diag.get("symptoms", [])))
          for i, rc in enumerate(diag.get("root_causes", [])[:2]):
              st.write(f"**Root cause #{i+1}:** {rc.get('hypothesis')} "
                       f"(weight: {rc.get('probability', 0):.0%})")
          st.write("**Intervention plan:**")
          for iv in diag.get("intervention_plan", []):
              st.write(f"  {iv.get('step')}. {iv.get('action')}")
          if diag.get("warnings"):
              for w in diag["warnings"]:
                  st.warning(w)

      chunks = result.get("retrieved_chunks", [])
      if chunks:
          with st.expander("📚 Knowledge sources used"):
              for c in chunks[:3]:
                  score = c.get("cross_encoder_score") or c.get("semantic_score", 0)
                  st.write(f"**[{c.get('channel')}]** {c.get('title')} "
                           f"— relevance: {score:.2f}")
                  if c.get("url"):
                      st.write(f"  🔗 [{c['url']}]({c['url']})")

      post_checks = eval_data.get("post", {}).get("checks", {})
      if post_checks:
          with st.expander("🔬 Quality evaluation"):
              for check, passed in post_checks.items():
                  icon = "✅" if passed else "❌"
                  st.write(f"{icon} {check.replace('_', ' ').title()}")

KEY UX REQUIREMENT :
  Demo mode doit fonctionner SANS aucune API key.
  Les 5 boutons "Example problems" doivent remplir le chat input et lancer l'analyse.
  Un recruteur doit pouvoir tester l'app en 10 secondes.
```

---

### PHASE 11 — Déploiement Streamlit Cloud

```
Prérequis : chroma_snapshot/ commité dans git (Phase 6).

1. requirements.txt (pour Streamlit Cloud — en plus de pyproject.toml) :
   sentence-transformers==2.7.0
   chromadb==0.5.0
   anthropic==0.25.0
   streamlit==1.35.0
   nltk==3.8.1
   google-api-python-client==2.120.0
   youtube-transcript-api==0.6.2
   pyyaml==6.0.1

2. .streamlit/config.toml :
   [server]
   maxUploadSize = 200

3. Sur Streamlit Cloud :
   - Main file : app/streamlit_app.py
   - Secrets : ANTHROPIC_API_KEY, YOUTUBE_API_KEY
   - Demo mode par défaut (DEMO_MODE=true dans secrets)

4. Test de démarrage :
   - L'app doit démarrer sans erreur
   - VectorStore doit logger "restoring from snapshot" au démarrage
   - Demo mode doit fonctionner sans API key

5. Post-déploiement : vérifier les 5 example problems en mode demo.
```

---

### PHASE 12 — Tests end-to-end + README + Demo

```
Tests end-to-end :
  python -m homebarista.pipeline "My DeLonghi Dinamica makes bitter espresso"
  python -m homebarista.pipeline "my moka tastes burnt"
  python -m homebarista.pipeline "V60 Ethiopian light roast too sour"
  python -m homebarista.pipeline "I want to learn how to make espresso"

Vérifier pour chaque run :
  - status == "coaching"
  - coaching_text > 100 mots
  - evaluation.post.verdict in ("pass", "warn")
  - sessions.jsonl mis à jour

README.md (portfolio-ready) :
  # HomeBarista Coach ☕
  > AI-powered barista coach that diagnoses coffee problems and delivers
  > science-backed coaching — whatever your machine.

  ## What it does
  Describe your machine, your beans, and what tastes wrong.
  HomeBarista diagnoses root causes and retrieves expert knowledge from
  300+ chunks of barista content (James Hoffmann, SCA, World Barista Championship).

  ## Architecture
  - **Agentic loop** : Claude tool_use decides whether to ask clarifications,
    re-retrieve with better queries, or regenerate coaching if quality checks fail.
  - **Deterministic diagnostic engine** : rule-based SYMPTOM_MATRIX with machine
    capability filter — LLM never touches the diagnostic logic.
  - **Cross-encoder re-ranking** : ms-marco-MiniLM-L-12-v2 for precise retrieval
    (vs. arbitrary composite scores).
  - **Stable eval dataset** : SHA-256 chunk IDs ensure retrieval metrics
    remain valid across re-ingestions.

  ## Retrieval evaluation results
  [paste from evals/results/retrieval_*.json]

  ## RAG evaluation results
  [paste from evals/results/rag_eval_*.json]

  ## Run locally
  cp .env.example .env  # add your keys
  uv sync
  python -m ingestion.run_ingestion --demo    # mock data
  streamlit run app/streamlit_app.py

  ⚠️ For educational/portfolio use only. YouTube transcripts used under fair use.
```

---

## 7. Interfaces entre composants — contrats critiques

Cette section documente les interfaces exactes entre composants.
**À lire avant d'implémenter ou modifier n'importe quel fichier.**

### BrewingContext → DiagnosticPlanner

`DiagnosticPlanner.diagnose(context)` lit `context.symptoms_detected` (list[str]).
Ce champ est peuplé par `SymptomExtractor.extract()`.
Si `symptoms_detected` est vide et `goal == "troubleshoot"`, le planner retourne confidence=0.

### ContentClassifier → run_ingestion

`ContentClassifier.classify(doc)` DOIT retourner un dict contenant `classification_confidence: float`.
`run_ingestion.py` filtre avec `if classification.get("classification_confidence", 0) < 0.4: skip`.
Sans ce champ, tous les documents passeront le filtre (comportement incorrect).

### Embedder → chunk IDs

Les `chunk_id` sont générés par `Embedder._make_chunk_id(source_id, text)`.
Format : `{source_id}_{sha256(text[:200])[:8]}`.
Les `relevant_chunk_ids` dans `eval_dataset.json` DOIVENT utiliser ce même format.
Ne jamais utiliser des IDs positionnels (`source_id_chunk_3`).

### Retriever ← Agent

`Retriever.retrieve()` accepte `query_override: Optional[str]`.
Quand l'agent appelle `retrieve_knowledge` avec un `query` custom,
il passe ce string via `query_override`.
Si `None`, le retriever construit la query depuis le diagnostic (comportement par défaut).

### VectorStore → Streamlit Cloud

`VectorStore._init_client()` suit cette logique :
1. Si `demo_mode=True` → in-memory ChromaDB (pas de filesystem)
2. Si `persist_dir` existe → load depuis persist_dir
3. Si `persist_dir` absent mais `snapshot_path` existe → restore depuis snapshot
4. Sinon → créer persist_dir vide

`snapshot_path` = `data/chroma_snapshot/` (commité dans git).
Sur Streamlit Cloud, `data/chroma_db/` n'existe jamais → toujours restore depuis snapshot.

### Pipeline → Agent mode vs. Linear mode

`run_pipeline(use_agent=True)` → `HomeBaristaAgent.run()` → tool-use loop
`run_pipeline(use_agent=False)` ou `demo_mode=True` → `_run_linear()` → ordre fixe

Les deux modes retournent le même format de dict.
Les tests utilisent `use_agent=False` pour la reproductibilité (pas de variabilité LLM).

### CoachingEvaluator ← Machine capability

`CoachingEvaluator.evaluate_diagnostic()` appelle `_check_interventions_feasible()`.
Cette méthode utilise `MACHINE_ADJUSTABLE` et `PARAM_TO_CAPABILITY` depuis `diagnostic_planner.py`.
Import : `from homebarista.diagnostic_planner import MACHINE_ADJUSTABLE, PARAM_TO_CAPABILITY`.
**Ces deux dicts sont la source de vérité pour les capabilities machine.**

---

## 8. Checklist soumission LLM Zoomcamp

- [ ] Problem description dans README (2-3 paragraphes)
- [ ] RAG flow documenté (architecture diagram + explication chunking strategy)
- [ ] Retrieval evaluation : Hit Rate@5, Precision@5, MRR documentés
- [ ] RAG evaluation : pass_rate structural + optionnel LLM judge
- [ ] UI déployée et accessible (Streamlit Cloud URL)
- [ ] Ingestion pipeline : commande documentée + ingestion_report.json sample
- [ ] Demo mode sans API key fonctionnel + 5 examples pré-remplis
- [ ] Tests pytest passants en CI (demo mode)
- [ ] Monitoring (bonus) : logs sessions.jsonl
- [ ] Docker Compose (bonus)
- [ ] Snapshot ChromaDB commité (déploiement fonctionnel)
- [ ] BIAS_NOTE dans rag_eval_results (LLM judge same-family bias documenté)

---

## 9. Ce que ce projet démontre pour le repositionnement

| Compétence | Comment HomeBarista la démontre |
|---|---|
| Agentic architecture | Tool-use loop : le LLM décide le flow, pas le code |
| RAG pipeline | Symptômes → diagnostic → retrieval ciblé → coaching |
| Ingestion robuste | Checkpointing par vidéo, quality filter, transcript cleaning |
| Chunking avancé | Sentence-aware (nltk), hash IDs stables, is_informative filter |
| Re-ranking | Cross-encoder ms-marco (vs. composite score arbitraire) |
| Retrieval evaluation | 50 queries synthétiques (vs. 20 hardcodées), Hash IDs stables |
| Honest evaluation | Biais LLM judge documenté, heuristic_weight vs. probability |
| Machine reasoning | Capability map : jamais de conseil physiquement impossible |
| Prompt engineering | System prompt structuré, tool definitions, style variants |
| Déploiement réel | Snapshot strategy pour Streamlit Cloud filesystem éphémère |
| Testing | pytest coverage sur tous les composants, CI demo-safe |

**Angle entretien** :
> "The LLM doesn't follow a fixed pipeline — it orchestrates.
> It decides whether to ask for clarification, whether to retry retrieval
> with a better query, and whether to regenerate coaching if quality checks fail.
> That's the agentic loop. The diagnostic engine stays deterministic —
> LLM is only responsible for communication, never for clinical logic."

---

*ULTRAPLAN V2 — HomeBarista Coach — Ellie Pascaud — Mai 2026*
*Intègre tous les fixes critiques post-audit senior + confrontation experts*
