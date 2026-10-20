# Job Search RAG

A learning project to build a retrieval-augmented job search API from the supplied LF Jobs dataset.

## Status

Repository structure and documentation scaffold are in place. The pipeline and API are not implemented yet.

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

Python is the planned runtime. Dependencies and executable setup instructions will be added when implementation begins. `requirements.txt` is currently a placeholder.

Copy `.env.example` to `.env` when configuring model providers. Keep credentials in `.env`, which Git ignores.

## Documentation

- [Project plan](docs/project_plan.md): requirements and milestones.
- [Decision log](docs/decisions.md): choices, reasoning, and alternatives.
- [Report draft](docs/report.md): architecture, usage, evaluation, assumptions, and limitations. Transfer the completed report to Google Docs as requested by the assignment.
- [Assignment PDF](docs/GenAI_Takeaway_Assignment.pdf).

## Evaluation plan

Compare vector-only search, hybrid search, and hybrid search with reranking on the same labeled queries. Check retrieval relevance, job deduplication, grounded responses, and latency. Metrics and results are pending.
