# Development evaluation

## Retrieval comparison

Eight assistant-curated, source-derived queries cover the six primary categories,
including a Chinese industrial-software query. Each case names one target job and
short supporting quotations checked against the cleaned description before the
run. These are development cases, not independent human labels or held-out test
queries. The query file was fixed before the comparison was executed.

This is a **known-item discovery** test: other returned jobs are unjudged and can
also be relevant. Hit@1 measures whether the target is first. Hit@5 measures
whether it appears in the first five distinct jobs. MRR@5 averages 1/target-rank,
with zero for a missing target. Precision, exhaustive recall, and nDCG are not
reported because the corpus does not have complete relevance judgments.

| Method | Target first | Target in top 5 | MRR@5 | Mean search seconds |
| --- | ---: | ---: | ---: | ---: |
| Vector | 8/8 | 8/8 | 1.0000 | 0.122 |
| Hybrid (vector + BM25 + RRF) | 8/8 | 8/8 | 1.0000 | 0.173 |
| Hybrid + cross-encoder | 7/8 | 8/8 | 0.9375 | 6.476 |

All methods use the same saved corpus and up to 20 chunks per retriever. Each
ranking keeps the highest-ranked chunk per distinct job. Hybrid fuses the vector
and keyword lists with RRF k=60; the reranker scores that union. Timing is one
sequential CUDA run on the development laptop and excludes shared model startup
(19.83 seconds) and answer generation. Hybrid and reranked timings include their
common prerequisite stages; repeated runs and confidence intervals were not
measured. These are not CPU API latency estimates or production benchmarks.

For K06, reranking moved target LF0832 to rank 2 behind LF0949. Source inspection
shows that LF0949 is also a Marketing Intern, Meal Plan Ambassador, with campus
meal-plan promotion duties. Therefore the lower known-target metric is not
evidence that this result is irrelevant. The set shows no improvement from hybrid
search or reranking and is too small/easy to establish general search quality.

Files: evaluation/known_item_queries.json contains cases and label provenance;
evaluation/retrieval_results.json contains rankings, times, settings, and query/
chunk SHA-256 hashes. The hashes identify the tested artifacts without bundling
the generated index or embedding files.

## Answer grounding and failures

Three fixed-evidence cases isolate local generation from retrieval: partial skill
matches, an unknown salary, and unknown current availability. They use real source
chunks and the actual local Qwen model with temperature zero. The answer service
still checks source-reference consistency. This is not an end-to-end quality
benchmark. Semantic review below is assistant source review; independent human
review is pending.

The initial run is retained in evaluation/answer_results.json. G01 returned known
citations but incorrectly grouped Kafka and MQ as databases and strengthened
"experience or general knowledge" to experience. G02 avoided inventing a salary
but also strengthened that qualifier. G03 was rejected by reference validation.
The diagnostic rerun in evaluation/answer_failure_diagnostic.json shows that its
declared citation LF0265 had no corresponding inline citation. It did not claim
the vacancy was currently open. Initial generation-only times were 20.34, 12.16,
and 13.37 seconds, respectively; the diagnostic rerun is separate.

The prompt was clarified to retain technical groupings and alternative knowledge/
experience wording, and to cite the inspected source even when information is
missing. Citation validation was not relaxed. Revised outputs and their prompt
hash are stored separately in evaluation/answer_results_revised.json.

That longer revision improved G01's technical grouping and knowledge/experience
qualifier, but called cloud familiarity optional without source support. It also
omitted the explicit missing-API qualification. G02 used the wrong inline ID
(LF0260 instead of LF0265), and G03 emitted an incomplete inline reference. Both
were rejected. Thus valid references and semantic grounding remain distinct
requirements, and this revision did not pass all three cases.

After the usage-limit interruption, a shorter prompt was tested on the same three
cases. Its outputs are preserved in evaluation/answer_results_final.json; “final”
here denotes the last trial, not the selected prompt. G01 passed reference checks
but regressed to unsupported API/direct-match claims and made Python mandatory
for both jobs. G02 omitted its inline citation and was rejected. G03 correctly
reported unknown availability with a valid citation. Times were 33.68, 11.01,
and 12.95 seconds. The shorter prompt was discarded; at that checkpoint the longer
revision was retained with a hash matching answer_results_revised.json. No citation
validation was weakened and these are development iterations, not held-out gains.

**Checkpoint finding:** prompt-only changes still had observed semantic and
citation failures. This prompted the source-selection change described below;
the free-form results are retained as development observations.

## Constrained source selection

The local adapter now requests numbered evidence lines instead of answer prose.
The application validates the choices, quotes source text, and renders actual
job-ID citations. An empty selection yields an insufficient-evidence answer.
The response API fields and retrieval pipeline remain unchanged.

The first live extractive run (evaluation/answer_results_extractive.json) passed
reference checks for G01-G03. G01 preserved the language alternatives but selected
only the Python and backend lines, omitting available database evidence. G02 and
G03 abstained without inventing salary/current status, with valid source IDs.
Generation-only times were 11.46, 8.64, and 8.43 seconds. These observations do
not establish general accuracy or a statistically supported latency gain.

The selection prompt was clarified to cover distinct requested requirements.
G04 (a known salary range) and G05 (recorded location metadata) were added as
development checks against indiscriminate abstention. Their expanded run is in
evaluation/answer_results_extractive_expanded.json. See docs/answer_design.md
for the validation flow and remaining selection/completeness limitations.

All five expanded cases passed reference checks. Assistant source review found
that G01 now included backend and database lines while preserving alternatives
and technical groupings; it also selected a less relevant design-solutions line.
It does not explicitly enumerate missing API evidence. G02/G03 abstained with
valid source IDs. G04 quoted the annualized salary range and its conditions; G05
quoted the recorded location without claiming remote eligibility. Generation-only
times were 12.79, 8.32, 8.52, 5.84, and 6.12 seconds. Detailed qualitative review
is in evaluation/answer_review_extractive.json. These five cases are not evidence
of general accuracy or a statistically supported latency improvement.

Previous real HTTP checks confirmed a successful generated response, an unknown
location returning empty jobs/citations without generation, and a blank query
returning 422. Unit HTTP checks cover busy requests (503/Retry-After), unavailable
generation (503), unexpected errors (500), invalid service output (500), and
startup failure. Simulated failures are not live fault-injection results.

## Reproduce

In the activated project environment, with processed data/indexes available:

```powershell
python scripts/evaluate_retrieval.py --device cuda
python scripts/evaluate_answers.py --output evaluation/answer_results_next.json
python -m unittest discover -s tests -v
```

Line 1 loads retrieval models once and compares the three rankings on the fixed
queries. `--device cuda` selects GPU retrieval; use cpu for a separate measured
run and give `--output` a different filename to preserve prior observations.
Line 2 requires local Ollama and generates answers for the fixed source
contexts, saving outputs for review. It does not automatically judge semantic
grounding. Line 3 runs component and HTTP tests with mocked model outputs.

The scripts overwrite their chosen result file, so use another `--output` to
retain an earlier run. Cached models allow offline retrieval execution; initial
model installation requires downloads. The local API can be stopped while
evaluating to reduce memory pressure. Evaluation does not rebuild the database.

## Remaining validation

The October 29 API extension is documented in docs/api_evaluation.md. Twelve
requests covered mixed-case locations, invalid input, Chinese/Japanese targets,
unsupported facts, a misleading user instruction, and metadata. Contract checks
passed; source review also found repeated Chinese evidence and partially relevant
Japanese secondary output. These observations extend rather than replace the
remaining validation listed below.

- Independent human review of queries, labels, and generated answers.
- A broader query set with judged relevant/irrelevant jobs and hard negatives.
- Requirements distributed across multiple chunks and ambiguous/no-answer queries.
- More multilingual and location-filter cases, plus prompt-injection robustness.
- Repeated timing/memory measurements and CPU API performance under realistic use.
- Other-platform/clean-profile setup verification and final submission/reviewer-access checks.

Fresh installation on the existing Windows machine subsequently passed; see
docs/setup_verification.md for the tested environment, rebuilding, API checks,
and cache-reuse limitations.
