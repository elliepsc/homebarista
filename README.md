# HomeBarista

HomeBarista is a Python project for coffee troubleshooting, ingestion, retrieval, and coaching workflows.

## Requirements

- Python 3.11.9
- uv
- API keys for optional online ingestion and LLM features

## Setup on Linux / WSL

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

Create a Linux virtual environment and install the locked dependencies, including development tools.
If the repository lives under `/mnt/c` in WSL, use copy mode to avoid hardlink issues between
the WSL environment and the Windows filesystem:

```bash
export UV_LINK_MODE=copy
uv venv .venv
source .venv/bin/activate
uv sync --extra dev --active
```

Activate the virtual environment when you want to run commands directly:

```bash
source .venv/bin/activate
```

Copy the example environment file and fill in local values:

```bash
cp .env.example .env
```

Validate the dependency installation:

```bash
python -c "import anthropic, chromadb, sentence_transformers, pytest; print('deps ok')"
```

If `sentence_transformers` fails with a `sympy` import error, reinstall `sympy` inside
the active environment:

```bash
export UV_LINK_MODE=copy
uv pip install --force-reinstall sympy==1.14.0 mpmath
python -c "from sympy import S; print('sympy ok')"
```

## Test

Run the test suite:

```bash
pytest
```

## Demo Ingestion

Run the mock ingestion pipeline without API calls:

```bash
python -m ingestion.run_ingestion --demo
```

Expected validation markers:

- `Loaded 40 mock documents`
- `Embedded 40 chunks`
- `Upserted 40 chunks`
- `Report saved     : ingestion/ingestion_report.json`

Then inspect the report:

```bash
cat ingestion/ingestion_report.json
```

The report should include:

```json
"mode": "demo",
"videos_fetched": 40,
"total_chunks_generated": 40,
"total_chunks_indexed": 40,
"quota_used": 0
```

## Environment

The expected environment variables are documented in `.env.example`:

- `YOUTUBE_API_KEY`
- `ANTHROPIC_API_KEY`
- `CHROMA_PERSIST_DIR`
- `DEMO_MODE`

## Data

Demo ingestion uses an in-memory Chroma store, so `data/chroma_db/` is not expected
to appear after `python -m ingestion.run_ingestion --demo`.

Persistent local Chroma data for non-demo runs should live under `data/chroma_db/`
and is ignored by Git.
