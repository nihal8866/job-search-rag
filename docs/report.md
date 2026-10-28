# Job Search RAG - Technical Report (Draft)

Status: working draft with data preparation, embeddings, reusable retrieval modules, local LLM generation, a local FastAPI interface, and exploratory results. Broader evaluation and final reproducibility details remain pending. A Google Docs version is required for final submission.

## 1. Problem and dataset

Build an API that retrieves relevant job listings and produces concise answers grounded in the supplied dataset. The workbook has 1,000 listings and nine columns, including HTML job descriptions and structured metadata.

## 2. Architecture

Implemented ingestion: Excel -> audit -> HTML cleaning -> metadata-preserving chunks -> embeddings -> vector database and keyword index.

Retrieval path: query -> vector and BM25 retrieval -> Reciprocal Rank Fusion -> cross-encoder reranking -> distinct jobs -> LLM response.

### Implemented data preparation and embedding stage

BeautifulSoup extracts plain text while retaining original HTML. All 1,000 cleaned descriptions are non-empty. Eleven descriptions contain characters in the checked CJK range; this is not full language detection. The initial model is intfloat/multilingual-e5-small, with multilingual support and a 512-token input limit (https://huggingface.co/intfloat/multilingual-e5-small).

Token windows target 384 description tokens with 48-token overlap. The resulting 3,618 chunks retain job metadata and identifiers. Embedding text includes the passage prefix, title, company, category, level, and chunk text. Long location lists initially caused 35 oversized inputs. Excluding locations from embedding text, while retaining them in metadata, reduced the maximum full input to 438 tokens. The reusable pipeline now applies an explicit location-text filter before retrieval.

GPU encoding produced a normalized array of shape (3618, 384). Validation checks passed for finite values, unit norms, and saved-artifact reload consistency. Chunk records and embeddings are associated by row order. Revised reload checks compare missing-value masks as well as normalized values. These checks establish numerical and persistence properties, not relevance or grounded-answer quality. Execution verification is tracked in the completion checklist.

### Implemented retrieval experiments

Chroma stores 3,618 vectors with chunk text and metadata using cosine distance. A direct record check verified text, metadata, and vector consistency for one record. Notebook 03 provides the vector-only baseline; notebook 04 independently reloads chunks and reconnects to the populated collection.

BM25 searches metadata and chunk text using a shared query/document tokenizer. The tokenizer normalizes case, preserves C++ and C#, and separates common CJK/kana characters as a simple fallback. It does not translate, stem words, or perform linguistic CJK word segmentation. Full locations are searchable keywords. The reusable pipeline also restricts both retrievers to IDs whose recorded locations contain the explicit filter text, ignoring case. This does not resolve aliases or geographic equivalence.

Up to 20 candidates from each retriever are combined using equal-weight Reciprocal Rank Fusion with k=60. The example query produced 34 unique chunks. BAAI/bge-reranker-v2-m3 scores query/contextual-chunk pairs on GPU in batches of four. All example pairs fit the configured 512-token pair limit, with a maximum of 435. Final job presentation keeps each job's best-scoring candidate chunk.

## 3. Setup and installation

The retrieval stages are also implemented as reusable modules under src/job_rag.
JobSearch composes hybrid candidates, pair-length-checked reranking, and distinct
job results while retaining evidence text and chunk identifiers. Fifty-eight
automated utility/orchestration checks pass; a real GPU run reproduced the
notebook example's five distinct jobs in the same order. Data preparation is also available through the installed ingestion command.

The development environment is Windows with Python 3.10.11 and an NVIDIA GPU. The README documents dependency installation and notebook execution order. Model files require a download on first use. The installed job-rag command supports ingestion, JSON search, and local answers. FastAPI provides the HTTP interface.

## 4. Example usage

Implemented endpoint: `POST /api/query`.

Requests include query, optional top_k (default 2, range 1-5), and optional literal location filter. Responses contain a generated answer, cited job IDs, retrieved evidence and scores, and the normalized request fields. Empty retrieval skips generation. Validation failures return 422, busy/local pipeline failures return 503, and unexpected errors return sanitized 500 responses. One shared service is loaded during startup with CPU retrieval and local Ollama generation. Setup and PowerShell request examples are in docs/api_usage.md. This is a local review configuration; broader API and grounding evaluation remain pending.

## 5. Evaluation and interpretation

An initial known-item comparison used eight assistant-curated source-derived
queries, covering six categories and a Chinese query. Vector and hybrid search
found all eight target jobs at rank 1. Reranking found seven at rank 1 and one
at rank 2; every method found all eight targets within five jobs. MRR@5 was
1.0000, 1.0000, and 0.9375, respectively. The alternate first result in the last
case is another relevant meal-plan marketing internship. These are target-discovery
metrics with incomplete relevance judgments, not overall accuracy or exhaustive
recall. No general improvement from the optional enhancements was demonstrated.

Mean retrieval-stage times in one CUDA run were 0.122 seconds (vector), 0.173
(hybrid), and 6.476 (reranked), excluding model startup and generation. These
are not CPU API latency estimates. Three fixed-evidence local-generation cases
tested partial matches, missing salary, and current availability. The initial
run exposed technical-grouping/qualifier errors and an answer rejected for
missing inline citations. Initial and revised outputs are retained separately.
See docs/evaluation.md and evaluation/ for methodology, results, provenance, and
remaining independent review. Broader judged relevance and repeated performance
measurements remain pending.

Prompt trials did not eliminate grounding problems. The selected longer revision
still produced unsupported qualifier wording and invalid citations in the fixed
cases; a shorter trial restored availability citations but introduced unsupported
API/direct-match claims and was discarded. All trial outputs are retained. The
93 passing automated tests check implementation contracts, not model truthfulness.
Correcting these observed failures remains necessary before submission.

For the exploratory query "Python backend developer working with APIs and databases", fusion ranked LF0247 first, while reranking placed LF0265 first. LF0247's inspected chunk supports backend work, Python as an accepted language, and database technologies. LF0265 explicitly requires Python but its inspected chunk does not establish API or database work. The results illustrate partial matches and the need to evaluate against explicit relevance criteria. They do not demonstrate overall improvement from hybrid search or reranking.

## 6. Assumptions

- The supplied workbook is the source of job information.
- Publication dates do not establish that a listing remains open.
- The diabetes classification sentence appears unrelated to the job RAG requirements.


## 7. Drawbacks and future enhancements

Current limitations include historical listings, location filtering without alias or geographic resolution, basic keyword tokenization, sentence boundaries within token windows, and job ranking from a single candidate chunk. The example also shows that a reranked result can omit requested skills. Future changes will be assessed using labeled queries rather than a single example.

## 8. Reproducibility and submission

On October 28, 2026, all four revised notebooks completed in separate fresh kernels in the existing project environment using cached model files. Dependency compatibility checks passed. A real loopback API check passed health, generated-query, empty-filter-result, and blank-query validation checks. One query took about 55 seconds after startup. Fresh-environment installation, broader failure/grounding evaluation, final report publication, and reviewer-access checks remain pending.

## Command-line and grounded-answer preparation

The package installs in editable mode and exposes job-rag ingest and job-rag search.
A full isolated ingestion reproduced the notebook's 1,000 jobs, 3,618 chunks, and
384-dimensional vectors; chunk IDs, text, contextual inputs, and vector values
were compared against the notebook artifacts. The installed search command
returned valid JSON with expected job IDs and source evidence from that rebuilt
database. Optional missing values are serialized as JSON null.

A provider-independent answer service and prompt are prepared. Answers must refer
to known retrieved job IDs, and declared citations must match inline references.
Empty retrieval bypasses generation. Unit checks use fake generator outputs;
the service also connects to a downloaded Qwen3 4B instruct model through local
Ollama. The ask command returns a generated answer plus source excerpts and
citations without a paid API dependency. The adapter restricts requests to
loopback, disables proxies/redirects, and has no cloud fallback.

A full retrieval, reranking, and local generation run took approximately 97
seconds with CPU retrieval on the development machine. It returned valid JSON
and known citations, but the answer overstated some requirements. The prompt
was subsequently tightened. Citation consistency does not establish factual
grounding; systematic live-model evaluation remains pending. The suite now
contains 87 passing tests, including 10 API contract tests with injected services.
The real API response still grouped Kafka and MQ as databases, rather than
messaging frameworks. This is a documented generation error to cover in the
grounding evaluation.
