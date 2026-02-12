FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/research-backend

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e .

COPY configs ./configs
COPY scripts ./scripts
COPY pathways ./pathways

CMD ["python", "scripts/run_pipeline.py", "--config", "configs/default.yaml", "--root", ".", "--sample-sheet", "data/interim/rna_sample_sheet_mapped.tsv"]
