# Evidence-based local answers

Free-form trials with the 4B model invented API relevance, changed alternatives
into mandatory requirements, and mistyped job-ID citations. More prompt wording
did not reliably fix those failures. The local answer path now uses the model
to select source lines and the application to render quoted evidence.

## Request flow

1. The existing pipeline retrieves/reranks chunks and keeps distinct jobs.
2. build_user_prompt in answers.py passes the question, source text, and metadata
   as JSON, without ranking scores.
3. selection_request in evidence_answers.py splits each excerpt into numbered,
   nonempty lines. It also provides source metadata lines for company, location,
   level, and publication date. It builds a JSON schema whose keys are the actual
   job IDs and whose arrays may contain only that job's line numbers.
4. OllamaGenerator sends that schema and selection instructions to the same local
   model. Loopback-only routing, no proxies/redirects, response limits, completion
   checks, and the conservative byte budget remain in place.
5. render_selection checks exact job keys, integer line numbers, bounds, uniqueness,
   and the maximum of three selections per job. It rejects invalid selections.
6. The renderer copies selected text in source order and adds known job citations.
   If no line was selected, it says that the selected evidence does not establish
   an answer. It does not claim the full original listing lacks that information.
7. GroundedAnswerService checks references again and returns the existing answer,
   cited_job_ids, and jobs fields. Brackets inside Markdown blockquotes are source
   text, not application citation markers.

The model supplies neither answer prose nor citation text. This prevents the
observed paraphrase and identifier-typing failures in this adapter. The generic
generator protocol remains available, but another implementation would need its
own equivalent grounding safeguards.

## What the checks establish

Rendered job facts are copied from the provided excerpt or recorded metadata,
and citations are attached to that same source. Alternatives such as "Java,
Golang, or Python" stay intact. Technical categories are not rewritten. An empty
selection is a supported outcome rather than a malformed uncited model response.

The model can still choose irrelevant/incomplete evidence or abstain unnecessarily.
A source line can contain a negation qualified elsewhere, boilerplate, or a fragment
created by chunking. Quoting does not verify the listing's truth or current status,
and does not establish that a candidate/job meets all requirements. Broader human
review, multi-chunk evidence, and no-answer robustness remain necessary.

## Verification

Automated tests cover source-specific schemas, metadata, preserved source order,
verbatim alternatives, empty selections, unknown jobs, duplicate/out-of-range/
boolean selections, free-form output rejection, and quoted bracket handling.
The initial three-case live run passed reference checks and avoided the observed
paraphrase errors, but omitted an available database line in the skill response.
That omission motivated a selection-coverage clarification and an expanded run.
See docs/evaluation.md and the stored extractive result files for actual outcomes;
passing schema/reference checks alone is not a semantic relevance score.

No packages, API keys, or paid services were added for this change. Restart an
already-running FastAPI process to load the updated code; no ingestion or embedding
regeneration is needed.
