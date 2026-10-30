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

## Browser walkthrough

Open http://127.0.0.1:8000/docs after startup. Expand **POST /api/query**, click
**Try it out**, replace the editable request body, and click **Execute**. The
answer appears under **Response body**, followed by the job details and evidence.
Replace placeholder values such as `"string"`; a literal location of `"string"`
will normally match no jobs. The root URL has no homepage and returns 404.

The API is local to the computer running it. A manager can run the repository on
their machine or view a screen-shared demonstration; your loopback URL does not
give them remote access. See [submission walkthrough](submission.md).

## Recorded examples and verification

[API examples](api_examples.json) contains three complete request/response pairs
copied from the actual October 29 development run: a Singapore job search, an
unknown location, and an unsupported salary question. Results are observations;
future model selections can vary.

The current answer adapter uses local Qwen to select source lines and renders
exact quotes with job-ID citations. It replaced earlier free-form output after
that output overstated requirements and grouped messaging products as databases.
Those earlier trials are retained in the evaluation records.

All 106 automated tests passed in the working and fresh environments. The suite
includes 10 API tests using injected services; these check HTTP behavior rather
than model quality. Twelve actual API development requests also passed their
applicable contract checks. They covered location filters, invalid inputs,
Chinese/Japanese queries, unsupported facts, metadata, and one misleading user
instruction. Relevant secondary evidence and repeated quotes remain limitations.
See [API evaluation](api_evaluation.md) and [fresh setup](setup_verification.md)
for exact scope and recorded timings. No general accuracy claim is made.

Start one worker, avoid rebuilding the database while it runs, and stop the API
with Ctrl+C when finished.
