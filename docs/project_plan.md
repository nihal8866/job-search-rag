# Completion Checklist

## Implemented

- [x] Repository structure and GitHub connection.
- [x] Data audit: missing values, duplicates, types, coverage, and dates.
- [x] HTML cleaning with original-data preservation and JSONL export.
- [x] Multilingual token analysis and overlapping chunks with source metadata.
- [x] GPU embeddings with shape, finite-value, and normalization checks.
- [x] Reload verification of values, missingness, and row alignment.
- [x] Persistent Chroma storage and vector-search baseline.
- [x] BM25 retrieval and Reciprocal Rank Fusion.
- [x] Multilingual cross-encoder reranking with pair-length checks.
- [x] Distinct-job presentation and exploratory evidence inspection.

## Review and reproducibility

- [x] Separate notebook responsibilities and remove temporary diagnostic cells.
- [x] Organize direct dependencies, notebook tools, and the environment snapshot.
- [x] Execute all four revised notebooks from fresh kernels.
- [ ] Verify installation in a newly created environment.

## Remaining implementation

- [x] Centralize existing retrieval settings in src/job_rag/config.py.
- [x] Extract keyword tokenization and RRF into retrieval.py with utility checks.
- [x] Add RetrievalIndex to load matching Chroma/chunk IDs and run BM25 search.
- [x] Add query embeddings, vector retrieval, and fused candidate search in reusable modules.
- [x] Connect reusable cross-encoder reranking and distinct-job presentation.
- [x] Add reproducible CLI ingestion for cleaning, chunking, embeddings, and indexing.
- [x] Add editable package installation and a JSON search command.
- [x] Add configurable retrieval settings and optional literal location filtering.
- [ ] Integrate an LLM with grounded responses and source references.
- [x] Prepare the provider-independent prompt, generator protocol, and citation checks.
- [ ] Implement FastAPI POST /api/query with validation and error handling.
- [ ] Evaluate vector-only, hybrid, and reranked retrieval on labeled queries.
- [ ] Check latency, grounding, no-match behavior, and pipeline/API failures.
- [ ] Add verified setup and API request/response examples.
- [ ] Complete the report, limitations, and future enhancements.
- [ ] Prepare the Google Docs report and verify reviewer access.
- [ ] Verify the complete codebase and submit the GitHub repository link.

## Scope

### Budget

The project must have no paid API, subscription, or hosting dependency. Use local
models and a local API server on existing hardware. There must be no automatic
fallback to a billable cloud provider. Electricity and internet remain ordinary
local operating costs rather than project service charges.

The selected direction for answer generation is Ollama with a quantized local
Qwen3 4B instruct model. Installation and real generation checks are pending.

The assignment's title and requirements specify job-data RAG. An isolated sentence
about diabetes classification appears unrelated and is treated as a copy-paste error.

## Verification record - October 28, 2026

All four revised notebooks completed in separate fresh kernels using the existing
project environment and cached models. The run regenerated cleaned data, chunks,
and GPU embeddings; verified artifact reloads and Chroma storage; and completed
BM25, fusion, and reranking. Chroma contained 3,618 records, the embedding array
was (3618, 384), and the example query produced 34 fused candidates.

`pip check` passed. Direct dependency pins match the installed environment, and
the requirements files passed an installed-environment dry run. Notebook schemas,
Python syntax, and documentation file links were checked. A fresh installation
into a new environment and broader relevance evaluation remain pending.

## Additional implementation verification

The package installs in editable mode and provides `job-rag` and `python -m job_rag`.
Full ingestion was run in a separate temporary project root against the original
workbook. It produced 1,000 jobs, 3,618 chunks, 384-dimensional vectors, and a maximum
input length of 438. Chunk IDs, text, contextual inputs, and vectors matched the
notebook artifacts. The installed search CLI returned valid JSON and the expected
first two jobs from the rebuilt database.

The grounded-answer interface is tested with fake generator outputs only.
Provider selection and live LLM verification remain required.
