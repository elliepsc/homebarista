# HomeBarista Coach — Streamlit app image
# Build: docker compose build   |   Run: docker compose up → http://localhost:8501
FROM python:3.11-slim

WORKDIR /app

# Dependencies first (cached layer as long as the lockfile doesn't change)
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

# Download NLTK data + sentence-transformers models at BUILD time,
# otherwise the first request pays a multi-minute cold start.
RUN uv run python -c "import nltk; nltk.download('punkt_tab')" && \
    uv run python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')"

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
