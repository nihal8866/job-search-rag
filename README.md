# Job Search RAG

Job retrieval over the LF Jobs dataset using multilingual embeddings, BM25,
Reciprocal Rank Fusion (RRF), and cross-encoder reranking.

The current implementation runs in notebooks. Reusable pipeline modules, LLM
answer generation, the FastAPI endpoint, and broader evaluation remain pending.

## Pipeline

```text
Excel -> HTML cleaning -> overlapping chunks -> embeddings -> Chroma
Query -> vector search + BM25 -> RRF -> cross-encoder -> distinct jobs
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

## Setup

Development environment: Windows, Python 3.10.11, NVIDIA RTX 4060 Laptop GPU
with 8 GB VRAM. Model-loading cells currently use `device="cuda"`.

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pip check
```

- `requirements.txt`: pinned direct data and retrieval dependencies.
- `requirements-dev.txt`: direct dependencies plus notebook tools.
- `requirements-lock.txt`: the full Windows environment snapshot, including
  transitive dependencies. Install this file instead for the recorded environment.

Dependency files include the PyTorch CUDA 12.8 wheel source. Installation in a
new environment remains a final submission check. Hugging Face model files
download on first use and are cached locally. The current retrieval notebooks
do not require an LLM API key.

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
- Full locations are metadata and BM25 text; strict location filtering is pending.
- Keyword tokenization has a basic CJK fallback without translation or stemming.
- Token windows can split sentences and retain company boilerplate.
- Ranking by a single chunk can miss requirements elsewhere in a job description.
- Labeled retrieval evaluation, latency measurements, LLM grounding, and API checks remain pending.

## Layout and documentation

```text
data/raw/       Supplied workbook
docs/           Assignment, decision notes, report draft, and checklist
notebooks/      Data preparation and retrieval experiments
src/job_rag/    Package scaffold for reusable code
scripts/        Reserved for pipeline entry points
tests/          Reserved for pipeline and API checks
```

See [engineering decisions](docs/decisions.md), the [report draft](docs/report.md),
the [completion checklist](docs/project_plan.md), and the
[assignment](docs/GenAI_Takeaway_Assignment.pdf). Final delivery also requires
the complete API, evaluation, and a Google Docs report.
