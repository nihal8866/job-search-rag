# Job Search RAG: Report Draft

Status: outline only. Complete this document with verified implementation details and results, then transfer it to Google Docs.

## 1. Problem and dataset

Build an API that retrieves relevant job listings and produces concise answers grounded in the supplied dataset. The workbook has 1,000 listings and nine columns, including HTML job descriptions and structured metadata.

## 2. Architecture

Planned ingestion: Excel -> audit -> HTML cleaning -> metadata-preserving chunks -> embeddings -> vector database and keyword index.

Planned query path: query -> vector and BM25 retrieval -> Reciprocal Rank Fusion -> cross-encoder reranking -> distinct jobs -> LLM response.

Describe the implemented components and their engineering tradeoffs here after validation.

## 3. Setup and installation

Pending: supported Python version, dependency installation, environment configuration, model downloads, ingestion command, and API launch command.

## 4. Example usage

Planned endpoint: `POST /api/query`.

Pending: validated request schema and actual requests with observed responses. Do not present illustrative results as measured output.

## 5. Evaluation and interpretation

Pending: labeled query set, relevance criteria, metrics, comparison of vector-only/hybrid/reranked search, latency, grounding checks, and failure examples.

## 6. Assumptions

- The supplied workbook is the source of job information.
- Publication dates do not establish that a listing remains open.
- The diabetes classification sentence appears unrelated to the job RAG requirements.

Add implementation assumptions as they arise.

## 7. Drawbacks and future enhancements

Pending: measured failure modes and limitations, then specific improvements supported by those observations.

## 8. Reproducibility and submission

Pending: final clean-run verification, GitHub repository access, Google Docs report link, and submission checklist.
