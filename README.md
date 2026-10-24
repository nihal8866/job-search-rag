# Job Search RAG

A learning project to build a retrieval-augmented job search API from the supplied LF Jobs dataset.

## Status

Dataset exploration and initial HTML cleaning are complete in `notebooks/01_data_exploration.ipynb`. The pipeline and API are not implemented yet.

## Planned pipeline

1. Read the Excel dataset and audit missing values, duplicates, and metadata.
2. Clean HTML job descriptions and create chunks with job identifiers and metadata.
3. Generate embeddings and store them in a vector database.
4. Retrieve candidates with vector similarity and BM25 keyword search.
5. Merge candidate rankings using Reciprocal Rank Fusion.
6. Rerank candidates with a cross-encoder and consolidate results by job.
7. Generate a grounded LLM response using the retrieved jobs.
8. Expose the pipeline through `POST /api/query` using FastAPI.

Models, libraries, chunk sizes, and retrieval settings will be selected and documented during implementation.

## Project layout

```text
data/raw/          Original Excel dataset
docs/              Assignment, plan, decision log, and report draft
notebooks/         Step-by-step exploration and experiments
src/job_rag/       Reusable pipeline and API code
scripts/           Data ingestion and other entry points
tests/             Pipeline and API verification
```

## Dataset

`data/raw/LF Jobs.xlsx` contains 1,000 job listings, 145 companies, seven categories, and nine columns. Job descriptions contain HTML. The dataset is supplied with the assignment; no redistribution license has been established.

## Setup

The exploration environment uses Python 3.10.11. From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` records the current Windows development environment, including notebook tools. Installation from this snapshot in a fresh environment has not yet been verified.

Open `notebooks/01_data_exploration.ipynb` in VS Code with the Python and Jupyter extensions, select the project `.venv` kernel, and run cells in order. The notebook reads the raw workbook and regenerates `data/processed/jobs_cleaned.jsonl`, which is intentionally ignored by Git. It preserves the original HTML and date strings alongside cleaned descriptions.

Copy `.env.example` to `.env` when configuring model providers. Keep credentials in `.env`, which Git ignores.

## Documentation

- [Project plan](docs/project_plan.md): requirements and milestones.
- [Decision log](docs/decisions.md): choices, reasoning, and alternatives.
- [Report draft](docs/report.md): architecture, usage, evaluation, assumptions, and limitations. Transfer the completed report to Google Docs as requested by the assignment.
- [Assignment PDF](docs/GenAI_Takeaway_Assignment.pdf).

## Evaluation plan

Compare vector-only search, hybrid search, and hybrid search with reranking on the same labeled queries. Check retrieval relevance, job deduplication, grounded responses, and latency. Metrics and results are pending.
