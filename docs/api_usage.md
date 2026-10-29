# Local API walkthrough

Run commands from the project root with `.venv` active. Ollama must already be
running with `OLLAMA_NO_CLOUD=1` and the downloaded Qwen model available. The
cleaned chunk file and Chroma collection must exist. Dependencies are recorded
in requirements.txt; API tests also need requirements-dev.txt.

Start the server in one terminal:

```powershell
python -m uvicorn job_rag.api:app --host 127.0.0.1 --port 8000 --workers 1
```

This loads the application from api.py, binds it to local port 8000, and starts
one worker. Retrieval models load once during startup, on CPU to reserve GPU
memory for Ollama. Wait for the startup-complete message and leave this terminal
running. Loading cached models can still take time.

Open a second PowerShell terminal with the project environment active:

```powershell
$body = @{ query = "Python backend developer working with APIs and databases"; top_k = 2 } | ConvertTo-Json
$result = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/query" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 360
$result | ConvertTo-Json -Depth 10
```

Line 1 builds a request object and converts it to JSON. Line 2 sends it to the
local POST endpoint and waits up to 360 seconds. Line 3 displays the returned
object as JSON, including nested job evidence. No API key is needed. These
commands use only the local API; first-time embedding/reranker downloads are a
separate setup step.

Add `location = "Singapore"` to the request object to filter recorded locations.
An unknown location yields empty jobs and skips generation. A location included
only in the question is not automatically extracted into a filter.

The endpoint validates query text (1-1,000 characters after trimming), an integer
top_k from 1 to 5, and an optional nonblank location (up to 300 characters).
Unknown request fields and malformed JSON are rejected. Model token limits and
the conservative generation-context budget are checked separately.

The API module separates request models, response models, service construction,
startup/shutdown, and HTTP routes. The lifespan creates one answer service. A
synchronous query route runs blocking model work in FastAPI's worker thread;
a lock prevents simultaneous inference. The response model documents and checks
the evidence structure. Existing citation checks remain in the answer service.
They establish valid references, not factual correctness.

## Verification

The current 106-test suite includes 10 API tests using an injected service: successful
responses, normalization, service reuse, malformed/invalid inputs, no matches,
busy inference, sanitized failures, invalid service output, health/OpenAPI, and
startup failures. These do not measure model quality. Real-model HTTP validation
used a temporary Uvicorn server on 127.0.0.1:8765 with the actual saved index and
cached models. Health, a two-job generated response, an empty location-filter
result, and a blank-query 422 all passed. The generated query took about 55
seconds after startup; the server was then stopped. This is one observed run,
not a latency benchmark. Its text still incorrectly grouped Kafka and MQ as
databases, illustrating why valid HTTP responses/citations do not establish
semantic grounding. See the project checklist for the verification record.

The server is intended for local use. Start one worker and avoid rebuilding the
database while it runs. Stop it with Ctrl+C when finished. Labeled retrieval
evaluation, broader grounding checks, and a fresh-environment installation are
still required before final submission.
