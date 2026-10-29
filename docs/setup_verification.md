# Fresh-environment verification — October 29, 2026

The development setup passed in a separate Python 3.10.11 environment on the
existing Windows/NVIDIA machine. The working .venv, data, and index were not
installation targets or rebuilt in place.

## Procedure and results

1. Copied 73 tracked/non-ignored working-tree files, including pending evaluation
   changes, into a temporary source snapshot. Ignored artifacts and Git metadata
   were excluded; this was not an exact checkout of only the last Git commit.
2. Created a fresh venv with include-system-site-packages=false.
3. Installed requirements-dev.txt from public PyPI and the recorded CUDA wheel
   index with isolated pip configuration. Cached installation took 578.22 seconds,
   which is one observation rather than a first-download estimate.
4. Installed the editable package using the documented build tools. pip check and
   all 106 tests passed in the new environment.
5. Ingestion rebuilt 1,000 jobs, 3,618 chunks, normalized (3618, 384) vectors, and
   3,618 Chroma records in the snapshot's own generated directories.
6. Chunk IDs, job IDs, text, and embedding inputs matched working artifacts exactly.
   Vectors were finite, normalized, and matched with atol=rtol=1e-5. The workbook
   hash matched the source.
7. Interpreter/package paths were confirmed to belong to the fresh environment
   and snapshot; working .venv site-packages were absent from sys.path.
8. A fresh loopback Uvicorn server passed health and a real local-Ollama query.
   Expected jobs LF0265/LF0247, citations, source quotes, and response checks passed.
   The query took 37.86 seconds after startup; the temporary server was then stopped.

The machine-readable record is evaluation/fresh_install_2026-10-29.json, including
step results, timings, setup hashes, isolation checks, versions, and artifact/API
checks. The full environment is in evaluation/fresh_environment_2026-10-29.txt;
the editable local path is excluded. Detailed logs and the temporary environment
stay under ignored tmp/.

## Dependencies

Ten indirect dependencies differed from the earlier requirements-lock.txt snapshot,
including filelock 4.1.0 versus 3.32.3 and python-dateutil 3.0.0 versus 2.9.0.post0.
The new set passed the checks above. Both snapshots are retained, rather than
silently replacing the working environment's history. The fresh snapshot also
records pip and build tools.

For the exact fresh dependency set, install that snapshot instead of requirements-
dev.txt in a newly created environment, then perform the editable installation.
The snapshot was exported after successful installation; installing that exported
snapshot itself was not a second independent run. Model weights and Ollama setup
remain separate prerequisites.

## Scope

The same OS, base Python, GPU driver, and user machine were used. Public pip cache
and existing Hugging Face/Ollama model caches could be reused. This checks package
isolation and artifact rebuilding, not first-time model downloads, another OS or
computer, a clean user profile, or CPU-only ingestion. Explicit interpreter paths
were used, so another user's PowerShell activation policy was not tested. The four
notebooks were not rerun here; equivalent ingestion and automated tests were.

The one API query is a setup check, not a quality or performance benchmark.
Independent human relevance review remains pending. Continue normal work in the
original .venv. The temporary environment, generated index/vectors, and model
downloads are not added to Git.
