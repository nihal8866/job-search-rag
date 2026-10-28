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
- [x] Integrate a local LLM with source references and structured responses.
- [x] Prepare the provider-independent prompt, generator protocol, and citation checks.
- [x] Implement FastAPI POST /api/query with validation and error handling.
- [x] Compare vector-only, hybrid, and reranked retrieval on eight labeled known-item cases.
- [x] Record initial latency, three fixed-evidence grounding cases, and API failure checks.
- [ ] Expand to independently reviewed relevance labels, harder queries, and repeated benchmarks.
- [ ] Correct observed answer-grounding/citation failures before final submission.
- [x] Add local setup instructions and API request examples.
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
Qwen3 4B instruct model. Installation and a full command-line generation check
have completed. Grounding evaluation remains pending.

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

The local Ollama adapter is connected. A full CPU retrieval/reranking plus local
generation check returned valid JSON and known citations in approximately 97
seconds on the development machine. This is one observed run, not a benchmark.
Initial output overstated some requirements; the prompt was tightened and
factual grounding remains an explicit evaluation task.

## API verification - October 28, 2026

The 87-test suite passed, including 10 API contract checks. A temporary Uvicorn
server bound to 127.0.0.1 was tested with real cached retrieval models and local
Ollama. GET /health succeeded; POST /api/query returned expected jobs LF0265 and
LF0247, valid JSON, and matching known citations. The query took approximately
55 seconds after startup. An unknown-location request returned empty results;
a blank query returned 422. The temporary server was stopped after testing.
This observed timing is not a latency benchmark. The generated text still
grouped Kafka and MQ as databases instead of messaging frameworks, so semantic
grounding remains unresolved and must be evaluated before submission.

## Evaluation checkpoint

Eight source-derived known-item cases completed: vector/hybrid target Hit@1
8/8, reranked 7/8; all target Hit@5 values 8/8. One displaced target was behind
another relevant internship, so these are not exhaustive relevance judgments.
Three fixed-evidence generation cases and prompt trials are recorded in
evaluation/. The shorter prompt trial worsened grounding and was discarded.
The longer tested revision remains selected and still has documented failures.
The 93-test suite passes; this does not establish live-model semantic accuracy.
See docs/evaluation.md for source review and remaining validation.
