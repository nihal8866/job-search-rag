# Job Search RAG - Technical Report (Draft)

Status: working draft with data preparation, embeddings, and exploratory retrieval results. Reusable modules, LLM integration, API, broader evaluation, and final reproducibility details remain pending. A Google Docs version is required for final submission.

## 1. Problem and dataset

Build an API that retrieves relevant job listings and produces concise answers grounded in the supplied dataset. The workbook has 1,000 listings and nine columns, including HTML job descriptions and structured metadata.

## 2. Architecture

Planned ingestion: Excel -> audit -> HTML cleaning -> metadata-preserving chunks -> embeddings -> vector database and keyword index.

Planned query path: query -> vector and BM25 retrieval -> Reciprocal Rank Fusion -> cross-encoder reranking -> distinct jobs -> LLM response.

### Implemented data preparation and embedding stage

BeautifulSoup extracts plain text while retaining original HTML. All 1,000 cleaned descriptions are non-empty. Eleven descriptions contain characters in the checked CJK range; this is not full language detection. The initial model is intfloat/multilingual-e5-small, with multilingual support and a 512-token input limit (https://huggingface.co/intfloat/multilingual-e5-small).

Token windows target 384 description tokens with 48-token overlap. The resulting 3,618 chunks retain job metadata and identifiers. Embedding text includes the passage prefix, title, company, category, level, and chunk text. Long location lists initially caused 35 oversized inputs. Excluding locations from embedding text, while retaining them in metadata, reduced the maximum full input to 438 tokens. Explicit location requirements need separate retrieval/filter handling.

GPU encoding produced a normalized array of shape (3618, 384). Validation checks passed for finite values, unit norms, and saved-artifact reload consistency. Chunk records and embeddings are associated by row order. Revised reload checks compare missing-value masks as well as normalized values. These checks establish numerical and persistence properties, not relevance or grounded-answer quality. Execution verification is tracked in the completion checklist.

### Implemented retrieval experiments

Chroma stores 3,618 vectors with chunk text and metadata using cosine distance. A direct record check verified text, metadata, and vector consistency for one record. Notebook 03 provides the vector-only baseline; notebook 04 independently reloads chunks and reconnects to the populated collection.

BM25 searches metadata and chunk text using a shared query/document tokenizer. The tokenizer normalizes case, preserves C++ and C#, and separates common CJK/kana characters as a simple fallback. It does not translate, stem words, or perform linguistic CJK word segmentation. Full location values are searchable keywords, but strict location constraints remain unimplemented.

Up to 20 candidates from each retriever are combined using equal-weight Reciprocal Rank Fusion with k=60. The example query produced 34 unique chunks. BAAI/bge-reranker-v2-m3 scores query/contextual-chunk pairs on GPU in batches of four. All example pairs fit the configured 512-token pair limit, with a maximum of 435. Final job presentation keeps each job's best-scoring candidate chunk.

## 3. Setup and installation

The development environment is Windows with Python 3.10.11 and an NVIDIA GPU. The README documents dependency installation and notebook execution order. Model files require a download on first use. Reusable ingestion commands and API launch instructions will be added with those implementations.

## 4. Example usage

Planned endpoint: `POST /api/query`.

The endpoint is not implemented yet. Request/response examples will be documented after end-to-end validation.

## 5. Evaluation and interpretation

Pending: labeled query set, relevance criteria, metrics, comparison of vector-only/hybrid/reranked search, latency, grounding checks, and failure examples.

For the exploratory query "Python backend developer working with APIs and databases", fusion ranked LF0247 first, while reranking placed LF0265 first. LF0247's inspected chunk supports backend work, Python as an accepted language, and database technologies. LF0265 explicitly requires Python but its inspected chunk does not establish API or database work. The results illustrate partial matches and the need to evaluate against explicit relevance criteria. They do not demonstrate overall improvement from hybrid search or reranking.

## 6. Assumptions

- The supplied workbook is the source of job information.
- Publication dates do not establish that a listing remains open.
- The diabetes classification sentence appears unrelated to the job RAG requirements.


## 7. Drawbacks and future enhancements

Current limitations include historical listings, no strict location filtering, basic keyword tokenization, sentence boundaries within token windows, and job ranking from a single candidate chunk. The example also shows that a reranked result can omit requested skills. Future changes will be assessed using labeled queries rather than a single example.

## 8. Reproducibility and submission

On October 28, 2026, all four revised notebooks completed in separate fresh kernels in the existing project environment using cached model files. Dependency compatibility checks passed. Fresh-environment installation, full API verification, final report publication, and reviewer-access checks remain pending.
