# Engineering decision log

Implementation choices, reasons, and current limitations for the notebook pipeline.

## 2026-10-20: Project scope

- Decision: build job retrieval with RAG, including both optional enhancements.
- Reason: the assignment explicitly requests job search, and both hybrid search and reranking were selected for this project.
- Scope: Python, FastAPI, vector retrieval plus BM25, Reciprocal Rank Fusion, a cross-encoder reranker, and an LLM response stage.
- Current status: reusable retrieval, local LLM generation, and the local FastAPI interface are implemented; broader evaluation remains pending.

## 2026-10-20: Repository organization

- Decision: separate original data, experiments, reusable code, scripts, tests, and documentation.
- Reason: preserve original inputs and keep experiments separate from the eventual API implementation.
- Documentation: maintain Markdown during development and transfer the final report to Google Docs for submission.

## 2026-10-24: Initial data audit and cleaning

- Environment: Python 3.10.11 in the project virtual environment; installed versions are recorded in requirements.txt. The installed environment passes pip check.
- Findings: 1,000 records, nine original columns, no fully duplicated rows, and no repeated job IDs. Nine locations and 880 tags are missing. Retain all records without inventing missing metadata.
- Dates: all publication dates parse as UTC; they span February 1, 2023, to August 1, 2025. They do not confirm that jobs remain open. Original date strings are retained in the export.
- Cleaning: use BeautifulSoup with html.parser and get_text(separator="\n", strip=True). Store the result in Clean Description, preserving original HTML. All 1,000 cleaned descriptions are non-empty.
- Language: 11 descriptions contain characters in the checked CJK ideograph range. This is not full language detection. Whitespace word counts undercount Chinese content, so token limits and multilingual coverage must guide model and chunking decisions.
- Persistence: export jobs to data/processed/jobs_cleaned.jsonl with Unicode preserved. The generated file is ignored by Git and can be recreated by the notebook.
- Verification: reloaded values match after replacing missing values with empty strings in temporary comparison copies. The revised verification also compares missing-value masks, so an empty string cannot conceal a missing-value change.
- Output: cleaned JSONL retains source fields and extracted descriptions.

## 2026-10-27: Chunking and embeddings

- Model: intfloat/multilingual-e5-small, selected as an initial multilingual retrieval baseline. Its documented maximum input is 512 tokens. Model card: https://huggingface.co/intfloat/multilingual-e5-small
- Token lengths: descriptions average 1,098.341 content tokens; 947 exceed 512 before metadata and special tokens.
- Chunking: 384-token windows with 48-token overlap, advancing by 336 tokens. Character offsets preserve source substrings. Retokenization can slightly change counts at boundaries; the observed maximum chunk content count is 385.
- Result: 3,618 chunks cover all 1,000 jobs, with no empty chunks or duplicate chunk IDs. Each chunk retains source job metadata.
- Header: passage prefix, job title, company, supplied category, and experience level precede the chunk text. Full location lists originally caused 35 inputs to exceed the limit (maximum 740). Excluding location from embedding text reduced the maximum to 438, with no oversized inputs. Full locations remain metadata; explicit location constraints still need filtering or keyword handling.
- Environment: Python 3.10.11, torch 2.11.0+cu128, sentence-transformers 6.2.0, transformers 5.20.2. GPU inference was confirmed on cuda:0.
- Encoding: normalized NumPy embeddings, batch size 16, progress bar disabled after a notebook stall. The stall's cause was not established. Small GPU, terminal, and notebook inference checks subsequently passed.
- Validation: array shape (3618, 384), finite values, unit norms within tolerance, exact vector-array reload equality, chunk-record reload equality after missing-value normalization, and one vector per record.
- Artifacts: data/processed/job_embeddings.npy and data/processed/job_chunks.jsonl. Row positions associate vectors with chunk records; they must be saved and loaded together without independently reordering either artifact.
- Scope: numerical and persistence checks do not establish retrieval quality.

## 2026-10-28: Vector, hybrid, and reranked retrieval

- Vector store: local Chroma PersistentClient at data/indexes/chroma, collection job_chunks_e5_small_v1, cosine distance, and explicitly supplied saved embeddings. Upsert in batches of 500 allows reruns for the current IDs. A stored record's text, metadata, and embedding were checked against the source. Collection and BM25 chunk-ID sets matched (3,618 IDs).
- BM25: rank_bm25.BM25Okapi over title, company, category, level, full locations, tags, and chunk text. Missing fields become empty text in the keyword corpus. Case folding, C++/C# preservation, and individual CJK/kana character separation form a basic tokenizer. No stemming, translation, linguistic CJK segmentation, or stopword removal is implemented. Only positive-scoring BM25 candidates are selected.
- Fusion: up to 20 chunks per retriever, equal contributions of 1/(60+rank), and deterministic chunk-ID tie breaking. The example query produced 34 unique candidate chunks. These parameters remain initial choices pending evaluation.
- Reranker: BAAI/bge-reranker-v2-m3 via Sentence Transformers CrossEncoder on CUDA, batch size four. Strip the E5 passage prefix; tokenize query and contextual chunk together. All 34 example pairs fit the configured 512-token limit (maximum 435). Scores are ranking signals, not calibrated guarantees of relevance.
- Job presentation: keep the first chunk per job after sorting by the applicable ranking. This ranks jobs by their best candidate chunk; it does not aggregate the full description.
- Observation: the Python/backend/API/database query ranked LF0247 first after fusion and LF0265 first after reranking. LF0265's retrieved text explicitly supports Python but does not establish API/database experience. This example does not establish a measured quality gain.
- Organization: notebook 03 contains vector storage, vector retrieval, and baseline observations; notebook 04 contains its own imports/data loading, BM25, Chroma reconnection, vector candidates, fusion, reranking, and evidence inspection. Each notebook loads its own inputs. Full execution results are recorded in the completion checklist.
- References: https://docs.trychroma.com/reference/python/client ; https://github.com/dorianbrown/rank_bm25 ; https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf ; https://huggingface.co/BAAI/bge-reranker-v2-m3

## Pending decisions

Reusable retrieval is now split into settings (`config.py`), tokenization/fusion
(`retrieval.py`), index loading (`index.py`), hybrid candidates (`search.py`),
pair scoring (`reranking.py`), and distinct-job orchestration (`pipeline.py`).
Models are loaded once per pipeline instance and reused. Oversized queries or
query/chunk pairs raise explicit errors instead of being silently truncated.
Fifty-eight automated checks pass. A real GPU run against the saved collection
reproduced notebook 04's top five distinct jobs and their evidence chunk IDs.
This verifies the refactor's example behavior, not general retrieval quality.

The reusable pipeline accepts an explicit optional location argument. It matches
literal substrings in recorded locations, ignoring case, and derives a shared
eligible chunk-ID subset. Chroma searches only those IDs and BM25 applies the
same subset before its result limit. Missing locations cannot satisfy the filter.
No eligible IDs returns an empty job table without embedding or reranker inference.
The filter does not infer locations from query text, resolve aliases, or establish
remote-work eligibility.

- Evaluation-driven refinements to embedding model, chunk boundaries, and HTML cleaning.
- Evaluation-driven refinements to candidate counts, fusion parameters, and job-level aggregation.
- Reranker latency budget and model refinements.
- Evaluation-driven prompt and response-schema refinements.
- Labeled evaluation queries and acceptance criteria.

## Package, ingestion, and answer interface

### Zero-service-cost requirement

The user requires no paid API, subscription, or hosting costs. Answer generation
will run locally through Ollama, initially using
qwen3:4b-instruct-2507-q4_K_M (approximately 2.5 GB download). No cloud fallback
will be added. Ollama's cloud features should be disabled in the server environment.
The local model and adapter are installed and a full CLI generation check has
completed. The adapter accepts only the selected local model on 127.0.0.1,
disables proxies/redirects, and has no cloud fallback. No API key is required.

The 8 GB GPU is also used by retrieval models. Initial integration can place
retrieval/reranking on CPU to reserve GPU memory for generation; benchmark actual
latency and memory before enabling simultaneous GPU residency for every model.

References: https://ollama.com/library/qwen3:4b-instruct-2507-q4_K_M and
https://docs.ollama.com/faq#how-do-i-disable-ollamas-cloud-features

The editable package exposes CLI entry points while keeping notebook experiments
available for explanation. Runtime dependencies match the direct requirements;
notebook/build tools are development dependencies. JSON search output separates
source evidence from optional ranking scores and sends model logs to stderr.

Canonical ingestion follows the notebook cleaning and chunking rules, checks
model input lengths and vector validity, and upserts a cosine-distance Chroma
collection. Stale collection IDs or a model/metric mismatch are rejected rather
than silently mixed. Generated artifacts are overwritten only after vector
generation and indexing checks pass; ingestion is not a cross-file/database
transaction, so interrupted builds must be rerun.

The answer layer uses a generator protocol with a local Ollama implementation.
Its prompt limits answers to provided evidence and requires explicit partial
matches and job-ID citations. The ask command defaults to two jobs and CPU
retrieval to reserve GPU memory for generation. It requests structured JSON
with temperature zero, an 8,192-token context, and a 512-token output limit.
A conservative input byte budget rejects oversized context before generation.
Reference checks reject unknown or inconsistent citations, but do not prove
semantic grounding. Free-form trials exposed requirement overstatements and
unreliable inline IDs. The current local adapter instead requests zero to three
numbered source lines per job and renders quotations/citations in application
code. Choices are schema-constrained and checked again before rendering; empty
selections explicitly abstain. This removes model-written paraphrases/citation
strings, while evidence relevance and completeness still need evaluation. See
docs/answer_design.md and docs/evaluation.md for the tradeoff and observed results.

## Local API

FastAPI exposes POST /api/query with typed JSON requests and evidence responses.
The lifespan loads one reusable answer service. CPU retrieval reserves GPU memory
for Ollama. One worker and a nonblocking inference lock avoid overlapping model
work; a busy request receives 503 with Retry-After. Bad input/model limits return
422, local pipeline failures return 503, and unexpected errors return 500 without
internal details. Health reports retrieval readiness, not Ollama availability.
No-match filters return successful empty results without generation. The server
binds to loopback for local review. The current 99-test suite includes 10 HTTP contract
tests with injected services; model quality is evaluated separately.
