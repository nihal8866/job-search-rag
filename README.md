# Job Search RAG

Search the LF Jobs dataset using vector search and BM25, combine the rankings
with RRF, and rerank the candidates with a cross-encoder. A local Qwen model
selects evidence from the results, and the API returns source quotes with job IDs.
Both optional enhancements, hybrid search and reranking, are included.

The project runs locally with Ollama. No paid API or model API key is required.

```text
Excel -> clean HTML -> overlapping chunks -> embeddings -> Chroma
Question -> vector + BM25 -> RRF -> rerank -> distinct jobs
Job evidence -> local Qwen selects lines -> source quotes and citations
```

## Configuration

| Part | Setting |
| --- | --- |
| Dataset | 1,000 jobs, 145 companies, 7 categories |
| Embeddings | intfloat/multilingual-e5-small, 384 dimensions |
| Chunks | 384 content tokens, 48-token overlap |
| Vector store | Chroma, cosine distance |
| Keyword search | rank_bm25.BM25Okapi |
| Fusion | Up to 20 chunks per retriever, RRF k=60 |
| Reranker | BAAI/bge-reranker-v2-m3, batch size 4 |
| Results | Best candidate chunk for each distinct job |
| Local model | qwen3:4b-instruct-2507-q4_K_M through Ollama |

Settings are in `src/job_rag/config.py`. Ingestion and the notebooks use CUDA.
The API and `ask` command use CPU retrieval to leave GPU memory for Ollama.

## Setup

Tested on Windows with Python 3.10.11 and an RTX 4060 Laptop GPU (8 GB).
From a new checkout, run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip --isolated install --index-url https://pypi.org/simple -r requirements-dev.txt
python -m pip --isolated install --no-build-isolation --no-deps -e .
python -m pip check
```

The editable install provides the `job-rag` command. Dependency files use the
PyTorch CUDA 12.8 wheel index. Model weights download on first use.

- `requirements.txt`: direct application dependencies.
- `requirements-dev.txt`: application dependencies, notebooks, tests, and build tools.
- `requirements-lock.txt`: the recorded working environment.
- `evaluation/fresh_environment_2026-10-29.txt`: the separately verified fresh environment.

Fresh setup, all 106 tests, ingestion, and a real API query passed on October 29.
The same Windows machine and existing package/model caches were used. Details and
dependency differences are in [setup verification](docs/setup_verification.md).

## Prepare the data

The source workbook is `data/raw/LF Jobs.xlsx`.

```powershell
job-rag ingest
```

This creates cleaned records, chunk records, and embeddings in `data/processed/`
and builds the collection in `data/indexes/chroma/`. Generated files are ignored
by Git. The raw workbook is kept unchanged. Chunk rows and embedding rows must
remain aligned.

Stop the API before rebuilding its database. If ingestion is interrupted, rerun
it before searching. The files and database are not saved as one transaction.
Collections with stale IDs or different model/distance settings are rejected.

## Run the API

Ollama must be running with the downloaded Qwen model. Set `OLLAMA_NO_CLOUD=1`
in the environment that starts Ollama. `.env.example` is a reference; the app
does not load it automatically.

```powershell
python -m uvicorn job_rag.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Wait for `Application startup complete`, then open
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Use **Try it out** on
**POST /api/query** with this body:

```json
{
  "query": "Python backend developer working with APIs and databases",
  "top_k": 2
}
```

`top_k` defaults to 2 and accepts 1-5. Add `"location": "Singapore"` to filter the
recorded locations. Matching ignores case but uses literal text; a location in
the question alone is not converted into a filter. Missing locations do not match.

The response contains the answer, cited job IDs, source chunks, job metadata,
and ranking scores. No eligible jobs returns HTTP 200 with empty jobs/citations.
Invalid input or model limits return 422; a busy or failed local pipeline returns
503; unexpected internal errors return 500. GET `/health` checks retrieval startup,
not Ollama availability. Only one inference runs at a time; busy requests include
`Retry-After: 5`.

The server is bound to this computer for local use. Stop it with Ctrl+C.
See [API usage](docs/api_usage.md) for a PowerShell request and parameter details.
The Swagger UI may need internet to load its assets; direct API requests do not.

## Other commands

```powershell
job-rag search "Python backend developer" --top-k 5
job-rag search "Python developer" --location "New York" --top-k 3
job-rag ask "Python backend developer working with APIs and databases" --top-k 2
job-rag --device cpu search "Python developer"
job-rag --project-root "C:\path\to\project" ingest --dataset "C:\path\to\jobs.xlsx"
```

`search` returns ranked evidence; `ask` adds the local answer step. Both print JSON
to stdout and model logs to stderr. Missing values become JSON null. Global
options go before the command. `python -m job_rag` accepts the same arguments.
Use `--project-root` with a non-editable install when running outside the data folder.

## Notebooks

Select the project `.venv` kernel in VS Code and run cells in order.

| Notebook | Purpose |
| --- | --- |
| [01 Data exploration](notebooks/01_data_exploration.ipynb) | Audit and clean the workbook |
| [02 Chunking and embeddings](notebooks/02_chunking_and_embeddings.ipynb) | Create aligned chunks and vectors |
| [03 Vector search](notebooks/03_vector_search.ipynb) | Populate Chroma and inspect vector results |
| [04 Hybrid search](notebooks/04_hybrid_search.ipynb) | BM25, RRF, reranking, and source inspection |

All four notebooks completed in fresh kernels on October 28. `job-rag ingest`
also prepares the artifacts without running notebooks.

## Answers and limitations

Free-form model answers overstated skills and sometimes mistyped citations.
The current adapter asks Qwen to select up to three numbered source lines per job.
The application checks the choices, copies source text, and adds the job IDs.
Empty selections produce an insufficient-evidence answer. The model stays on
127.0.0.1; proxies, redirects, and cloud fallback are disabled.

The model uses an 8,192-token context with space reserved for output. A conservative
UTF-8 byte check rejects oversized evidence before generation; reduce `top_k` if
needed. Ollama unloads the model after the response. See [answer design](docs/answer_design.md).

Remaining limitations include historical listings, literal location matching,
basic CJK keyword splitting, sentence cuts at chunk boundaries, repeated source
passages, and ranking each job by one chunk. Selected quotes can be incomplete
or only partly relevant. Scores are ranking signals, not match probabilities.

## Evaluation

```powershell
python -m unittest discover -s tests -v
```

All 106 tests pass. Model/database calls are mocked in the unit tests; actual
ingestion, generation, and HTTP checks are recorded separately.

Eight source-derived known-item queries found every target in the top five for
all three methods. Vector/hybrid put all eight first; reranking put seven first.
The displaced target was behind another relevant internship. This small development
set does not show a general improvement from the optional enhancements.

Five fixed-evidence answer cases and twelve broader API cases are also recorded.
The API checks covered Chinese/Japanese queries, location filters, invalid input,
missing facts, and one misleading instruction. Repeated Chinese quotes and generic
secondary evidence remain documented issues. See [evaluation](docs/evaluation.md)
and [API checks](docs/api_evaluation.md) for results, label provenance, timings,
and limits. Independent review and repeated benchmarks remain pending.

## Files and private data

```text
data/raw/       Source workbook
notebooks/      Experiments
src/job_rag/    Ingestion, search, reranking, answers, and API
scripts/        Evaluation runners and credential-pattern checker
tests/          Component, CLI, and API tests
evaluation/     Query sets, observed results, and environment records
docs/           Assignment, decisions, report, and verification notes
```

Keep credentials and private files out of commits and shared logs. `.env`, common
key/credential files, and local account directories are ignored. For a new clone,
enable the repository's staged-file check:

```powershell
git config --local core.hooksPath .githooks
python scripts/check_secrets.py --staged
```

The checker reports filenames and issue types, not matched values. It covers common
patterns, not every secret format or binary/image content. Ignore rules do not
remove committed files; revoke an exposed credential before handling its history.

See the [technical report](docs/report.md), [decisions](docs/decisions.md),
[checklist](docs/project_plan.md), and [assignment](docs/GenAI_Takeaway_Assignment.pdf).
The [manager walkthrough](docs/submission.md) explains how to review and test the
project. The [Google Docs report](https://docs.google.com/document/d/14eDGQY6v6J13eL8WvSqqfh6yqrWuenemxUIjX3jOcuc/edit)
and editable [Word copy](docs/job_search_rag_report.docx) are ready. Both use
Times New Roman, 12-point black body text, and 1.5 body line spacing, with tighter
spacing in tables and command examples. The thirteen-page layout
was inspected. The repository is public, and the report is viewable by anyone
with its link; public editing is disabled.
It includes flowcharts for data preparation and question answering, with figure
files and editable SVG sources in `docs/figures/`.
The report opens with a title page and contents with checked page numbers; both
flowcharts have numbered captions and appear in the list of figures.
Dataset and technology tables, a recorded question walkthrough, and a manager quick-start
guide explain the implementation and how to review its answers.
Manager access must be granted and checked before submission.
