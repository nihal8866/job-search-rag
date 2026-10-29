# Broader local API checks — October 29, 2026

Twelve sequential requests tested the existing server. Cases and expectations were
saved before execution in evaluation/api_cases.json. Expectations are assistant-
curated from source inspection, not independent labels or a held-out benchmark.
No index rebuild, model change, server restart, or cloud service was needed.

| Case | Check | HTTP | Seconds | Observation |
| --- | --- | ---: | ---: | --- |
| A01 | Mixed-case Singapore filter | 200 | 43.11 | Filter held; LF0247 first |
| A02 | Unknown location | 200 | 0.02 | Empty jobs/citations |
| A03 | Blank question | 422 | 0.01 | Rejected |
| A04 | top_k above 5 | 422 | 0.01 | Rejected |
| A05 | Blank location | 422 | 0.01 | Rejected |
| A06 | Chinese industrial-software query | 200 | 32.16 | LF0204 first; Shanghai filter held |
| A07 | Japanese SAP AMS query | 200 | 31.66 | LF0304 first; Tokyo filter held |
| A08 | Unknown exact salary | 200 | 7.10 | Insufficient evidence |
| A09 | Current application availability | 200 | 6.54 | Insufficient evidence |
| A10 | Unsupported astronaut qualifications | 200 | 6.88 | Insufficient evidence |
| A11 | Misleading salary/status instruction | 200 | 6.83 | Insufficient evidence |
| A12 | Recorded EPAM location | 200 | 6.96 | Brezno, Slovakia quoted |

All cases passed applicable automatic contract checks: expected HTTP status,
nonempty answer, normalized request fields, distinct jobs/result limit, matching
known citations, literal location filtering, constrained answer format, and quote
membership in the cited source/metadata. The checker was expanded during this HTTP
run; stored responses were rechecked with the final checker without inference.
Original bodies/timings are retained with a separate rescore timestamp.

Target ranks and expected abstentions are separate quality observations, not an
accuracy score. Three known targets ranked first; four unsupported-fact/misleading-
instruction cases abstained. This does not judge every returned job as relevant.
Abstention on a retrieved job is distinct from no jobs: vector retrieval can return
partial matches when the requested fact is absent.

Assistant review found limitations. The Chinese answer repeated source text present
twice at different positions. The Japanese target selected SAP AMS duties, but its
secondary job supplied generic project-management text, an incomplete heading, and
company metadata without SAP evidence. Source fidelity passes while selection
quality remains incomplete. A11 tests one misleading user instruction, not source-
embedded instructions or comprehensive adversarial robustness.

The eight inference requests averaged 17.65 seconds (median 7.03); invalid/empty-
location requests are excluded. Startup is excluded; CPU retrieval and local
Ollama were reused. Workloads differ and each case ran once, so these are not
repeated latency/memory benchmarks or throughput measurements. The automated suite
has 106 passing tests, including seven checker tests. Human review, fully judged
relevance queries and source-prompt attacks remain pending. Fresh installation
on the existing Windows machine subsequently passed; see docs/setup_verification.md.

With the API and Ollama running, use the activated project environment:

```powershell
python scripts/evaluate_api.py --output evaluation/api_results_next.json
```

This sends cases one at a time to 127.0.0.1:8000 and saves responses, timings,
contract checks, and separate quality observations. It does not start/stop the
server. Avoid other queries during the run because inference is serialized. Use
`--cases` for another suite and preserve the original case file.

To recheck stored responses without model calls:

```powershell
python scripts/evaluate_api.py --rescore evaluation/api_results_2026-10-29.json --output evaluation/api_results_rechecked.json
```

`--rescore` applies current checks to stored status/body pairs. The case-file hash
must match, preventing changed expectations from being assigned to old observations.
It creates no new latency measurements. Different output names preserve prior runs.
Raw responses are in evaluation/api_results_2026-10-29.json; assistant review is in
evaluation/api_review_2026-10-29.json. Neither contract passes nor target ranks
should be described as overall search accuracy.
