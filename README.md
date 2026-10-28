# Job Search RAG

Job retrieval over the LF Jobs dataset using multilingual embeddings, BM25,
Reciprocal Rank Fusion (RRF), and cross-encoder reranking.

The experimental workflow runs in notebooks. A reusable retrieval pipeline under
`src/job_rag/` now combines index loading, BM25, query embeddings, vector retrieval,
rank fusion, cross-encoder reranking, and distinct-job presentation. Ingestion
and search also run through the installed command-line interface. Grounded answer
generation uses a local Qwen model through Ollama. FastAPI exposes this pipeline
through POST /api/query. Broader evaluation and submission preparation remain pending.

## Pipeline

```text
Excel -> HTML cleaning -> overlapping chunks -> embeddings -> Chroma
Query -> vector search + BM25 -> RRF -> cross-encoder -> distinct jobs
Distinct jobs + source excerpts -> local Qwen -> cited answer
```

| Component | Current configuration |
| --- | --- |
| Dataset | 1,000 jobs, 145 companies, 7 categories |
| Embedding model | intfloat/multilingual-e5-small; 384 dimensions |
| Chunks | 384-token windows with 48-token overlap |
| Vector store | Local Chroma with cosine distance |
| Keyword retrieval | rank_bm25.BM25Okapi |
| Fusion | Up to 20 chunks per retriever; RRF k=60 |
| Reranker | BAAI/bge-reranker-v2-m3; batches of 4 |
| Job presentation | Best-scoring retrieved chunk per job |
| Answer model | Local Ollama qwen3:4b-instruct-2507-q4_K_M |

## Setup

Development environment: Windows, Python 3.10.11, NVIDIA RTX 4060 Laptop GPU
with 8 GB VRAM. Model-loading cells currently use `device="cuda"`.

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
```

- `requirements.txt`: pinned direct data and retrieval dependencies.
- `requirements-dev.txt`: direct dependencies plus notebook and build tools.
- `requirements-lock.txt`: the full Windows environment snapshot, including
  transitive dependencies. Install this file instead for the recorded environment.

Dependency files include the PyTorch CUDA 12.8 wheel source. Installation in a
new environment remains a final submission check. Hugging Face model files
download on first use and are cached locally. The current retrieval notebooks
do not require an LLM API key.

The editable installation provides `job-rag` and makes `job_rag` importable without
setting `PYTHONPATH`. It uses the build tools installed with the development
requirements. The recorded environment uses setuptools 65.5.0 and wheel 0.48.0.

## Command-line workflow

Notebooks explain the experiments; these commands run the same data and search
stages without opening a notebook:

```powershell
job-rag ingest
job-rag search "Python backend developer working with APIs and databases" --top-k 5
job-rag search "Python backend developer" --location "New York" --top-k 3
job-rag ask "Python backend developer working with APIs and databases" --top-k 2
```

`ingest` reads `data/raw/LF Jobs.xlsx`, rebuilds cleaned records and aligned
embeddings under `data/processed/`, and upserts Chroma. It overwrites generated
artifacts, not the raw workbook. It rejects an existing collection with stale IDs
or a different model/distance configuration rather than mixing datasets. A failed
or interrupted ingestion should be rerun before serving searches; the three files
and database are not a single transactional store.

`search` prints JSON containing job metadata, evidence text, source chunk IDs, and
scores. Missing optional metadata/scores become JSON null. Progress/model messages
go to stderr. A missing dataset/index or invalid input returns a nonzero status.

Global options go before the command. For example:

```powershell
job-rag --device cpu search "Python developer"
job-rag --project-root C:\\path\\to\\project ingest --dataset C:\\path\\to\\jobs.xlsx
```

`python -m job_rag` accepts the same arguments. The CPU option is available but
performance verification so far uses the NVIDIA GPU. With a non-editable package
installation, use `--project-root` when running outside the data directory.

## Notebook order

In VS Code, select the project `.venv` kernel and run each notebook in order.

| Notebook | Purpose | Generated output |
| --- | --- | --- |
| [01 Data exploration](notebooks/01_data_exploration.ipynb) | Audit the workbook and clean HTML | jobs_cleaned.jsonl |
| [02 Chunking and embeddings](notebooks/02_chunking_and_embeddings.ipynb) | Create and verify chunks and vectors | job_chunks.jsonl, job_embeddings.npy |
| [03 Vector search](notebooks/03_vector_search.ipynb) | Populate Chroma and inspect the vector baseline | data/indexes/chroma/ |
| [04 Hybrid search](notebooks/04_hybrid_search.ipynb) | Build BM25, fuse rankings, rerank, and inspect evidence | Ranked chunk and job tables |

Processed files are written to `data/processed/`. They and the generated database
are ignored by Git and recreated in notebook order. Vector rows and chunk rows
must stay aligned. Each notebook loads its own inputs.

Alternatively, `job-rag ingest` creates all processed artifacts and the database.

## Verification

On October 28, 2026, all four notebooks completed in separate fresh kernels using
the project environment and cached model files. Dependency compatibility and
requirements-file checks passed. This verifies the current notebook workflow,
not a fresh environment installation or overall retrieval quality.

## Results and limitations

Preparation produces 3,618 chunks covering all jobs. Complete embedding inputs
have a maximum of 438 tokens, within the 512-token limit. Validation checks cover
unique IDs, job coverage, finite vectors, normalization, and artifact reloads.

The example query "Python backend developer working with APIs and databases"
produces 34 hybrid candidates. RRF ranks LF0247 first; reranking places LF0265
first. LF0265's inspected chunk explicitly supports Python but does not establish
API or database work. This is a functional example, not a measured quality gain.

- Historical publication dates do not establish whether a listing remains open.
- Location filtering matches literal text in the recorded field, ignoring case;
  it does not resolve aliases, geography, or remote-work eligibility.
- Keyword tokenization has a basic CJK fallback without translation or stemming.
- Token windows can split sentences and retain company boilerplate.
- Ranking by a single chunk can miss requirements elsewhere in a job description.
- Initial known-item and fixed-evidence generation checks are recorded in
  [evaluation notes](docs/evaluation.md); broader relevance/grounding evaluation
  and repeated latency benchmarks remain pending.

## Layout and documentation

```text
data/raw/       Supplied workbook
docs/           Assignment, decision notes, report draft, and checklist
notebooks/      Data preparation and retrieval experiments
src/job_rag/    Ingestion, retrieval, reranking, and local answer generation
scripts/        Repository secret-pattern checker
tests/          Component, CLI, and API contract checks
```

See [engineering decisions](docs/decisions.md), the [report draft](docs/report.md),
the [completion checklist](docs/project_plan.md), and the
[assignment](docs/GenAI_Takeaway_Assignment.pdf). Final delivery also requires
evaluation, fresh-environment verification, and a Google Docs report.

## Utility checks

From the project root in an activated environment:

```powershell
python -m unittest discover -s tests -v
```

The checks cover keyword tokenization, fusion, query/vector validation, score
alignment, reranker pair limits, job deduplication, and vector fallback when keyword retrieval has no hits. Models and
database calls are mocked in those checks; no weights are downloaded. A separate
run against the real saved collection reproduced the example's 34 fused candidates
and first-ranked LF0247 chunk. The complete reusable pipeline also reproduced
notebook 04's five distinct job IDs in the same order after reranking. These
checks do not evaluate overall relevance.

Ingestion, CLI parsing/JSON serialization, and the grounded-answer interface are
also checked. Generator outputs are mocked; those checks are not live LLM
validation. A full isolated CLI ingestion reproduced all 3,618 notebook chunks and
vectors, and the installed CLI returned LF0265 and LF0247 with evidence for the
example query.

## Local API

Keep Ollama running with cloud features disabled. From the project root in the
activated environment, start one worker bound to this computer:

```powershell
python -m uvicorn job_rag.api:app --host 127.0.0.1 --port 8000 --workers 1
```

`python -m uvicorn` starts the installed ASGI server. `job_rag.api:app` identifies
the application. `--host 127.0.0.1` keeps the listener on this computer, `--port
8000` selects its port, and `--workers 1` avoids loading multiple model copies.
Wait for `Application startup complete`. Stop with Ctrl+C. Do not run ingestion
against the same database while the server is using it.

Open [API docs](http://127.0.0.1:8000/docs), expand POST /api/query, choose
**Try it out**, and use:

```json
{"query": "Python backend developer working with APIs and databases", "top_k": 2}
```

`query` is the search question. `top_k` selects 1-5 distinct jobs (default 2).
Optional `location` applies the existing literal location filter. More sources
may exceed the local model's conservative evidence budget. The response contains
`answer`, `cited_job_ids`, full `jobs` evidence and scores, `query`, and
`location_filter`. No match is HTTP 200 with empty jobs/citations and an
explanatory answer; an unfiltered vector search can still return partial matches.

GET `/health` confirms startup and retrieval readiness; it does not probe Ollama.
Invalid bodies or model input/context limits return 422. Busy inference or an
unavailable/failed local generation pipeline returns 503; unexpected internal
errors return 500. Error bodies omit internal exception details. Only one query
runs at a time; busy callers receive `Retry-After: 5`. Retrieval models are reused
on CPU, and generation uses the local Ollama adapter. This configuration is for
local development and review, without public hosting or authentication.

See [the API walkthrough](docs/api_usage.md) for a PowerShell request and the
verification record. Swagger UI assets may require internet on first load;
PowerShell requests work without loading that UI.

## Reusable job search

After ingestion or notebooks 01-03 have created the artifacts and collection,
create one pipeline instance and reuse it for queries:

```python
from job_rag.config import Settings
from job_rag.pipeline import JobSearch

search = JobSearch(Settings())
jobs = search.search("Python backend developer working with APIs and databases", top_k=5)
print(jobs[["rank", "ID", "Job Title", "reranker_score"]])
```

Use `location="New York"` as an optional argument to `search.search()` to restrict
both retrievers before candidate selection. Matching is case-insensitive literal
substring matching against the full recorded location string. Missing locations
are excluded when a filter is supplied. An unknown location returns no jobs;
an omitted filter searches all locations. Blank filters are rejected.

The result retains the best chunk's identifier, source text, metadata, and scores
for each distinct job. Query and pair lengths are checked before scoring. Scores
are not relevance probabilities. This retrieval stage does not yet generate
an LLM answer. Locations written only in the natural-language query are not
automatically converted into a filter; pass the explicit argument when required.

## Credentials and private files

Local inference needs no model API key. Keep credentials, private keys, account
files, and personal documents outside the repository and out of screenshots or
shared logs. `.env`, common private-key formats, credential filenames, and local
account directories are ignored. `.env.example` documents the non-secret local-only flag.

The current checkout enables a repository-local pre-commit secret-pattern check.
For a new clone, enable it with:

```powershell
git config --local core.hooksPath .githooks
```

Run `python scripts/check_secrets.py --staged` to inspect staged text before a
commit, or omit `--staged` to inspect tracked working files. Reports show filenames
and issue types, never matched values. This is a common-pattern check, not a full
privacy audit; unfamiliar secret formats, images, and binary files need review.
Ignore rules do not remove anything already committed. If a real credential is
ever exposed, revoke/rotate it before addressing repository history.

## Grounded-answer interface

`answers.py` defines a provider-independent generator protocol and system prompt.
The prompt requires evidence-based answers, explicit partial matches, and job-ID
citations. The service rejects empty answers, unknown citations, or disagreement
between inline citations and declared IDs. Empty retrieval skips generation.

The `ask` command connects this service to the downloaded
`qwen3:4b-instruct-2507-q4_K_M` model through Ollama on `127.0.0.1:11434`.
Start Ollama with `OLLAMA_NO_CLOUD=1` in its server environment. No API key is
needed. The adapter rejects other hosts and models and disables proxies and
redirects. There is no cloud fallback.

`ask` defaults to two jobs and CPU retrieval/reranking to leave GPU memory for
generation. `search` and `ingest` retain their CUDA default; `--device` overrides
the selection. The adapter requests structured JSON, checks completion, and
uses a conservative UTF-8 byte budget to leave room for output within its
8,192-token context. Oversized evidence is rejected; reduce `--top-k` if needed.
The model is unloaded after the response. Citation validation checks reference
consistency; it does not prove factual grounding. Initial live checks showed
requirement overstatements, so answer quality still needs systematic evaluation.
