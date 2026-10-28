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

## Pending decisions

- Vector database.
- Evaluation-driven refinements to embedding model, chunk boundaries, and HTML cleaning.
- Candidate counts, fusion parameters, and job-level aggregation.
- Reranker model and latency budget.
- LLM provider, model, prompt, and response schema.
- Labeled evaluation queries and acceptance criteria.
