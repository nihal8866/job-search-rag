# Engineering decision log

Record a decision when evidence supports it. Distinguish planned choices from implemented behavior.

## 2026-10-20: Project scope

- Decision: build job retrieval with RAG, including both optional enhancements.
- Reason: the assignment explicitly requests job search, and both hybrid search and reranking were selected for this project.
- Scope: Python, FastAPI, vector retrieval plus BM25, Reciprocal Rank Fusion, a cross-encoder reranker, and an LLM response stage.
- Status: planned; implementation and model choices are pending.

## 2026-10-20: Repository organization

- Decision: separate original data, experiments, reusable code, scripts, tests, and documentation.
- Reason: preserve original inputs and keep experiments separate from the eventual API implementation.
- Documentation: maintain Markdown during development and transfer the final report to Google Docs for submission.

## 2026-10-24: Initial data audit and cleaning

- Environment: Python 3.10.11 in the project virtual environment; installed versions are recorded in requirements.txt. The user reported that pip check found no broken requirements.
- Findings: 1,000 records, nine original columns, no fully duplicated rows, and no repeated job IDs. Nine locations and 880 tags are missing. Retain all records without inventing missing metadata.
- Dates: all publication dates parse as UTC; they span February 1, 2023, to August 1, 2025. They do not confirm that jobs remain open. Original date strings are retained in the export.
- Cleaning: use BeautifulSoup with html.parser and get_text(separator="\n", strip=True). Store the result in Clean Description, preserving original HTML. All 1,000 cleaned descriptions are non-empty.
- Language: 11 descriptions contain characters in the checked CJK ideograph range. This is not full language detection. Whitespace word counts undercount Chinese content, so token limits and multilingual coverage must guide model and chunking decisions.
- Persistence: export jobs to data/processed/jobs_cleaned.jsonl with Unicode preserved. The generated file is ignored by Git and can be recreated by the notebook.
- Verification: reloaded values match after replacing missing values with empty strings in temporary comparison copies. This comparison treats missing and empty text as equivalent; it does not verify identical missing-value representations.
- Status: exploration and initial cleaning are complete. Chunking, model selection, and pipeline implementation remain pending.

## 2026-10-27: Chunking and embeddings

- Model: intfloat/multilingual-e5-small, selected as an initial multilingual retrieval baseline. Its documented maximum input is 512 tokens. Model card: https://huggingface.co/intfloat/multilingual-e5-small
- Token lengths: descriptions average 1,098.341 content tokens; 947 exceed 512 before metadata and special tokens.
- Chunking: 384-token windows with 48-token overlap, advancing by 336 tokens. Character offsets preserve source substrings. Retokenization can slightly change counts at boundaries; the observed maximum chunk content count is 385.
- Result: 3,618 chunks cover all 1,000 jobs, with no empty chunks or duplicate chunk IDs. Each chunk retains source job metadata.
- Header: passage prefix, job title, company, supplied category, and experience level precede the chunk text. Full location lists originally caused 35 inputs to exceed the limit (maximum 740). Excluding location from embedding text reduced the maximum to 438, with no oversized inputs. Full locations remain metadata; explicit location constraints still need filtering or keyword handling.
- Environment: Python 3.10.11, torch 2.11.0+cu128, sentence-transformers 6.2.0, transformers 5.20.2. GPU inference was confirmed on cuda:0.
- Encoding: normalized NumPy embeddings, batch size 16, progress bar disabled after a notebook stall. The stall's cause was not established. Small GPU, terminal, and notebook inference checks subsequently passed.
- User-reported validation: array shape (3618, 384), finite values, unit norms within tolerance, exact vector-array reload equality, chunk-record reload equality after missing-value normalization, and one vector per record.
- Artifacts: data/processed/job_embeddings.npy and data/processed/job_chunks.jsonl. Row positions associate vectors with chunk records; they must be saved and loaded together without independently reordering either artifact.
- Status: embeddings generated. Saving the latest notebook and a final clean-kernel run remain to be verified. These checks do not establish retrieval quality.

## 2026-10-28: Vector, hybrid, and reranked retrieval

- Vector store: local Chroma PersistentClient at data/indexes/chroma, collection job_chunks_e5_small_v1, cosine distance, and explicitly supplied saved embeddings. Upsert in batches of 500 allows reruns for the current IDs. A stored record's text, metadata, and embedding were checked against the source. Collection and BM25 chunk-ID sets matched (3,618 IDs).
- BM25: rank_bm25.BM25Okapi over title, company, category, level, full locations, tags, and chunk text. Missing fields become empty text in the keyword corpus. Case folding, C++/C# preservation, and individual CJK/kana character separation form a basic tokenizer. No stemming, translation, linguistic CJK segmentation, or stopword removal is implemented. Only positive-scoring BM25 candidates are selected.
- Fusion: up to 20 chunks per retriever, equal contributions of 1/(60+rank), and deterministic chunk-ID tie breaking. The example query produced 34 unique candidate chunks. These parameters remain initial choices pending evaluation.
- Reranker: BAAI/bge-reranker-v2-m3 via Sentence Transformers CrossEncoder on CUDA, batch size four. Strip the E5 passage prefix; tokenize query and contextual chunk together. All 34 example pairs fit the configured 512-token limit (maximum 435). Scores are ranking signals, not calibrated guarantees of relevance.
- Job presentation: keep the first chunk per job after sorting by the applicable ranking. This ranks jobs by their best candidate chunk; it does not aggregate the full description.
- Observation: the Python/backend/API/database query ranked LF0247 first after fusion and LF0265 first after reranking. LF0265's retrieved text explicitly supports Python but does not establish API/database experience. Do not claim measured quality gains from this single example.
- Organization: notebook 03 contains vector storage, vector retrieval, and baseline observations; notebook 04 contains its own imports/data loading, BM25, Chroma reconnection, vector candidates, fusion, reranking, and evidence inspection. Original code and outputs were preserved when split. Schema and syntax checks passed; fresh-kernel execution of the separated notebooks remains pending.
- References: https://docs.trychroma.com/reference/python/client ; https://github.com/dorianbrown/rank_bm25 ; https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf ; https://huggingface.co/BAAI/bge-reranker-v2-m3

## Pending decisions

- Evaluation-driven refinements to embedding model, chunk boundaries, and HTML cleaning.
- Evaluation-driven refinements to candidate counts, fusion parameters, and job-level aggregation.
- Reranker latency budget and model refinements.
- LLM provider, model, prompt, and response schema.
- Labeled evaluation queries and acceptance criteria.
