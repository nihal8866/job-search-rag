# Submission and manager walkthrough

The assignment requires the complete repository and a Google Docs report. It
does not require public hosting or a separate chat website. The local FastAPI
documentation page provides the question-and-answer interface.

## Deliverables

- Repository: https://github.com/nihal8866/job-search-rag
- Technical report: [Markdown](report.md) and [Word version for Google Docs](job_search_rag_report.docx).
- Recorded [request and response examples](api_examples.json).
- [Google Docs report](https://docs.google.com/document/d/14eDGQY6v6J13eL8WvSqqfh6yqrWuenemxUIjX3jOcuc/edit).

The native Google Doc was imported from the Word file on October 30. Both copies
use Times New Roman, 12-point black body text, and 1.5 line spacing. All fifteen pages
were inspected, including headings, code, and the tables.
The report includes a title page, a table of contents with checked page numbers,
and a list of figures matching the two flowchart captions. Contents page numbers
are fixed for this version and should be rechecked after layout changes.
Its permissions currently show owner-only access. Manager access is still pending.
Keep the report copies consistent if the text changes.
The report includes dataset and technology decisions tables, a recorded question
walkthrough, a validation summary, and a manager quick-start guide.

## How to review the code

Start with the README for setup and commands, then read the technical report.
The notebooks show exploration, chunking, vector retrieval, and hybrid retrieval.
Reusable application code is in src/job_rag, and tests are in tests.
The source workbook is data/raw/LF Jobs.xlsx. Generated vectors and indexes are
ignored by Git and rebuilt with job-rag ingest.

The tested setup is Windows, Python 3.10.11, and an NVIDIA RTX 4060 Laptop GPU.
The full fresh-environment record is in [setup verification](setup_verification.md).
Other platforms and CPU-only ingestion have not been independently verified.
If the manager lacks suitable hardware, demonstrate the running project on the
existing laptop through screen sharing instead of requiring a new purchase.

## Where to ask questions

After following the README installation and ingestion steps, start local Ollama
with the selected Qwen model, then run this in the activated project terminal:

```powershell
python -m uvicorn job_rag.api:app --host 127.0.0.1 --port 8000 --workers 1
```

This starts one API worker on local port 8000. Leave that terminal running.
Wait for Application startup complete, then:

1. Open http://127.0.0.1:8000/docs on the same computer.
2. Expand POST /api/query and click Try it out.
3. Replace the editable request body with a real question and click Execute.
4. Read answer, cited_job_ids, and jobs under Response body.
5. Inspect evidence and source_chunk_id for each returned job.

For a demonstration of matching and a location constraint:

```json
{
  "query": "Python backend developer working with databases",
  "location": "Singapore",
  "top_k": 2
}
```

The recorded run used mixed-case sInGaPoRe and returned LF0247 first. Check that
locations contain Singapore, each result has a distinct job ID, and answer quotes
belong to the cited evidence. Other relevant results or selected lines may vary.
top_k is a maximum, not a guarantee that all returned jobs meet every preference.

For a missing-information check:

```json
{
  "query": "What is the exact annual salary for the EPAM Senior Python Developer?",
  "location": "Brezno",
  "top_k": 1
}
```

The recorded answer abstained from giving a salary. Also try an unknown location
to see an empty result and a blank query to see HTTP 422 validation. Full observed
outputs are in api_examples.json. These examples are development checks, not an
independent accuracy benchmark.

The dataset is historical; do not use this demonstration to confirm that a job is
still accepting applications. To stop the API, press Ctrl+C in its terminal.

## Automated checks

```powershell
python -m pip check
python -m unittest discover -s tests -v
python scripts/check_secrets.py
```

The first command checks dependencies. The second runs the automated tests;
106 passed in the verified environments. The third checks tracked text for common
credential patterns without printing matched values. It is not a full security
audit and does not inspect binary/image contents.

## Google Docs and access

The Google Docs conversion is complete. Retain restricted access unless the
assignment requires broader sharing. Give the manager Viewer access using their
confirmed account and verify that they can open it. Do not put credentials,
personal environment files, private keys, or logs into the report or repository.

If the repository is private, invite the manager's confirmed GitHub account.
Verify access with the manager; being able to open a link in the owner's account
does not prove that the reviewer can open it.

## Final handoff

- [x] Required pipeline and API implemented; both optional enhancements included.
- [x] Setup, assumptions, engineering decisions, and limitations documented.
- [x] Example requests and observed responses included.
- [x] Automated, actual-model, and fresh-environment checks recorded.
- [x] Native Google Docs report created and conversion inspected.
- [ ] Manager access to the repository and report verified.
- [ ] Final changes committed and pushed; working tree checked.
- [ ] Repository and Google Docs links sent to the manager.

Independent relevance review, larger evaluations, better source deduplication,
multi-chunk evidence, and repeated performance measurements are future work.
They must not be presented as completed validation.

Suggested handoff message after access is confirmed:

> The job-search RAG assignment is ready for review. The repository contains the
> code, notebooks, setup instructions, tests, and recorded evaluation outputs.
> The Google Docs report explains the architecture, decisions, examples, and
> limitations. After starting the project locally, you can ask questions at
> http://127.0.0.1:8000/docs using POST /api/query. I can also demonstrate it on
> my laptop. The dataset is historical, and answers return cited source evidence.

Include the actual repository and Google Docs URLs when sending the message.
