# Job Search RAG — Technical report

Draft for submission. Technical checks are recorded below; independent relevance
review, the Google Docs version, and reviewer-access checks are still pending.

## 1. Problem and data

The goal is to search the supplied job listings and answer questions using their
source evidence. The input is data/raw/LF Jobs.xlsx: 1,000 listings, nine columns,
145 companies, and seven job categories. Descriptions contain HTML.

There are nine missing locations, 880 missing tag values, and no repeated job IDs
or fully duplicated rows. All descriptions remain nonempty after cleaning. All
publication dates parse as UTC and range from February 1, 2023 to August 1, 2025.
These dates do not confirm whether a vacancy is still open. Eleven descriptions
contain characters in the checked CJK range; this is a character check, not full
language detection.

## 2. Data preparation and search

BeautifulSoup removes HTML while the original description and date strings are
retained. Descriptions are split into 384-token windows with 48-token overlap.
This produces 3,618 chunks covering all jobs.

The embedding model is intfloat/multilingual-e5-small. Each input contains the
passage prefix, job title, company, category, level, and chunk text. Including long
location lists initially pushed 35 inputs over 512 tokens. Location is kept as
metadata instead; the final complete input has a maximum of 438 tokens.

GPU encoding produces normalized vectors with shape (3618, 384). Checks cover
finite values, unit norms, unique chunk IDs, job coverage, and saved-file alignment.
Reload checks compare both values and missing-value positions.

Chroma stores the vectors, source text, and metadata using cosine distance. BM25
indexes metadata and chunk text, including full locations. The shared keyword
tokenizer normalizes case, preserves C++/C#, and separates common CJK/kana characters.
It does not translate, stem words, or perform linguistic CJK word segmentation.

Vector and BM25 search each return up to 20 chunks. Equal-weight RRF combines
their rankings with k=60. BAAI/bge-reranker-v2-m3 then scores query/chunk pairs in
batches of four, after checking the 512-token pair limit. Results keep the best
candidate chunk per distinct job. This does not merge a job's full description.

An explicit location filter restricts both retrievers before their result limits.
It uses case-insensitive literal matching against recorded location strings.
It does not infer geography, aliases, or remote eligibility from the question.

## 3. Local answers

The answer model is qwen3:4b-instruct-2507-q4_K_M through local Ollama. Early
free-form answers overstated skills and produced incorrect inline citations.
Prompt-only revisions did not reliably fix those errors.

The current model selects up to three numbered source lines per job. A per-request
JSON schema limits the choices to that source. Application code checks the choices,
copies the text, and attaches the actual job ID. Empty selections produce an
insufficient-evidence answer; empty retrieval skips generation. Known metadata can
also be selected for company, location, level, and publication date.

This prevents model-written paraphrase and citation-typing errors in this adapter.
It does not guarantee relevant or complete evidence. Quotes can contain boilerplate,
repetition, or text cut at chunk boundaries. Historical data cannot establish
current application availability. See answer_design.md for the validation flow.

## 4. Setup and API

The tested machine uses Windows, Python 3.10.11, and an RTX 4060 Laptop GPU with
8 GB VRAM. Ingestion and notebooks use CUDA. The API uses CPU retrieval to leave
GPU memory for Ollama. It runs locally without a paid API or model API key.

The README contains installation and startup commands. job-rag ingest prepares
the data/index; search returns ranked evidence; ask adds the local answer step.
The original workbook is unchanged. If ingestion is interrupted, it must be rerun
because files and database updates are not one transaction.

POST /api/query accepts:

```json
{
  "query": "Python backend developer working with APIs and databases",
  "top_k": 2
}
```

top_k defaults to 2 and accepts 1-5. An optional location field applies the filter.
The response includes answer, cited_job_ids, jobs with source text/chunk IDs/scores,
and normalized request fields. Missing metadata becomes JSON null.

No eligible jobs returns HTTP 200 with empty results. Invalid input/model limits
return 422; busy or failed local inference returns 503; unexpected errors return
500 without internal details. One service loads at startup and processes one
inference at a time. GET /health checks retrieval readiness, not Ollama availability.
The listener is bound to 127.0.0.1 for local review. See api_usage.md for examples.

## 5. Evaluation

Eight assistant-curated, source-derived queries test whether a known target job
is found. They cover six categories and include a Chinese query. These are
development cases, not independent labels or a held-out set. Other jobs remain
unjudged, so exhaustive recall, precision, and general accuracy are not claimed.

| Method | Target first | Target in top five | MRR@5 | Mean retrieval seconds |
| --- | ---: | ---: | ---: | ---: |
| Vector | 8/8 | 8/8 | 1.0000 | 0.122 |
| Hybrid | 8/8 | 8/8 | 1.0000 | 0.173 |
| Hybrid + reranker | 7/8 | 8/8 | 0.9375 | 6.476 |

The displaced target was behind another relevant meal-plan internship. This small
set does not show a general improvement from either optional enhancement. Timing
is one CUDA run excluding startup and generation, not a repeated benchmark.

Five fixed-evidence answer cases checked partial skills, missing salary/current
status, a known salary range, and recorded location. All passed reference checks.
Assistant source review found supported quotes and appropriate abstentions, but
also a less relevant selected line and no explicit list of missing API evidence.

Twelve live API cases covered mixed-case location filters, unknown locations,
invalid inputs, Chinese/Japanese targets, unsupported facts, one misleading user
instruction, and metadata. All applicable contract checks passed. Three known
targets ranked first; four unsupported-fact cases abstained. Review found repeated
Chinese source text and generic/fragmentary Japanese secondary evidence.

The eight inference requests averaged 17.65 seconds excluding startup. Different
workloads and one run per case preclude benchmark claims. Unit tests mock models
and database calls; real ingestion, generation, and HTTP checks are separate.
All 106 automated tests pass. Results, raw outputs, query provenance, and review
limits are retained in evaluation/ and described in evaluation.md/api_evaluation.md.

## 6. Reproducibility

All four notebooks completed in separate fresh kernels on October 28. On October
29, a new Python 3.10.11 environment with no system site-packages installed the
development requirements and editable package. pip check and all 106 tests passed.

A separate source snapshot rebuilt 1,000 jobs, 3,618 chunks, normalized vectors,
and Chroma records. IDs, text, and embedding inputs matched working artifacts;
vectors matched within 1e-5. Import paths were checked. A fresh API query returned
the expected LF0265/LF0247 jobs with valid source quotes/citations in 37.86 seconds
after startup. The temporary server was stopped afterward.

The same Windows machine, driver, and existing package/model caches were used.
This does not test another platform, a clean user profile, or first-model downloads.
Ten indirect dependency versions differed from the earlier working snapshot; both
sets are retained. See setup_verification.md and the dated environment records.

## 7. Limitations and further work

Listings are historical. Location matching is literal, keyword segmentation is
basic, and token windows can split sentences. One-chunk job ranking can miss
requirements elsewhere in a description. Evidence selection can repeat passages,
include headings, or return only partial matches.

Next steps are independent relevance review, harder/no-answer queries, source-
prompt tests, multi-chunk evidence, and repeated latency/memory measurements.
Current checks do not establish production throughput or broad adversarial robustness.

## 8. Assumptions and submission

The supplied workbook is the source of job information. The assignment's isolated
diabetes-classification sentence is treated as a copy-paste error; the implementation
follows its job-RAG requirements. Both optional enhancements are included.

Code, notebooks, tests, and verification notes are in the repository. Final report
review, the required Google Docs version, and reviewer access are still pending.
