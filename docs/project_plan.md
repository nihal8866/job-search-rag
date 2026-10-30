# Project checklist

## Completed

- [x] Repository structure and GitHub setup.
- [x] Workbook audit, HTML cleaning, and cleaned-data export.
- [x] Multilingual token checks and overlapping chunks with source IDs.
- [x] GPU embeddings, normalization, and saved-file alignment checks.
- [x] Chroma vector search, BM25, RRF, and cross-encoder reranking.
- [x] Distinct-job results and a literal location filter applied before retrieval.
- [x] Reusable ingestion/search modules and command-line entry points.
- [x] Local Ollama answers with checked source-line selection and citations.
- [x] FastAPI POST /api/query with validation and error handling.
- [x] Run the four notebooks in separate fresh kernels.
- [x] Compare retrieval methods on eight known-item queries.
- [x] Check five fixed-evidence answer cases and twelve live API cases.
- [x] Pass all 106 automated tests in the working and fresh environments.
- [x] Rebuild artifacts and test a real API query in a fresh environment.
- [x] Add setup instructions, API examples, and credential-pattern checks.
- [x] Finalize report content against the assignment and include recorded responses.
- [x] Create and inspect the Word report and native Google Docs conversion, including the eleven-page version with title page, contents, and numbered figures.
- [x] Write the manager demonstration and submission walkthrough.

## Remaining

- [ ] Grant and verify manager access to the Google Docs report and repository.
- [ ] Send the repository and Google Docs URLs to the manager.

## Further work

- Independent human review of relevance labels and answers.
- Improve repeated quotes and partly relevant secondary results.
- Extend hard/no-answer cases, source-prompt tests, and repeated timings.

These are documented limitations and possible improvements. They are not claimed
as completed work or as additional requirements from the assignment.

## Scope

The project uses existing hardware, local models, and a local API. It has no paid
API, subscription, hosting requirement, or automatic cloud fallback. Ordinary
internet/electricity use is separate from service charges.

The assignment is treated as job-data RAG. Its isolated diabetes-classification
sentence appears to be a copy-paste error. Both optional enhancements, hybrid
search and reranking, are included.

## Verification records

| Record | What it covers |
| --- | --- |
| [Evaluation](evaluation.md) | Eight source-derived known-item cases and answer trials |
| [API evaluation](api_evaluation.md) | Twelve actual HTTP cases and source review |
| [Answer design](answer_design.md) | Source-line selection, citations, and remaining limits |
| [Fresh setup](setup_verification.md) | Installation, 106 tests, rebuilt artifacts, and a real query |

Query/review provenance is recorded with the evaluation files. The current results
do not establish overall search accuracy or an independent human evaluation.
Fresh setup used the same Windows machine and existing package/model caches.
