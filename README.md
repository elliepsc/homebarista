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
```

Create a Linux virtual environment and install the locked dependencies, including development tools:

```bash
uv venv .venv-linux
source .venv-linux/bin/activate
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

## Test

Run the test suite:

```bash
uv run pytest
```

## Environment

The expected environment variables are documented in `.env.example`:

- `YOUTUBE_API_KEY`
- `ANTHROPIC_API_KEY`
- `CHROMA_PERSIST_DIR`
- `DEMO_MODE`

## Data

Local generated Chroma data should live under `data/chroma_db/` and is ignored by Git.
