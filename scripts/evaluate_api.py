"""Run sequential checks against an already-running local API; no server changes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener, Request

from job_rag.api_evaluation import inspect_response
from job_rag.config import PROJECT_ROOT
from job_rag.ollama_local import NoRedirects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "evaluation/api_cases.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evaluation/api_results_current.json")
    parser.add_argument("--rescore", type=Path, help="Recheck stored responses without sending HTTP requests.")
    args = parser.parse_args()
    suite_bytes = args.cases.read_bytes()
    suite = json.loads(suite_bytes)
    checker_hash = hashlib.sha256((PROJECT_ROOT / "src/job_rag/api_evaluation.py").read_bytes()).hexdigest()
    if args.rescore:
        result = json.loads(args.rescore.read_text(encoding="utf-8"))
        if result["case_sha256"] != hashlib.sha256(suite_bytes).hexdigest():
            raise ValueError("The stored run and case file differ; do not reassign case expectations.")
        cases = {case["id"]: case for case in suite["cases"]}
        for row in result["results"]:
            row.update(inspect_response(cases[row["case_id"]], row["status"], row["response"]))
        result["rescored_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["response_checker_sha256"] = checker_hash
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        print(f"Rechecked {len(result['results'])} stored responses without HTTP requests.")
        return
    url = urlsplit(args.base_url)
    if (url.scheme != "http" or url.hostname != "127.0.0.1" or url.username or url.password
            or url.query or url.fragment or url.path not in {"", "/"}):
        raise ValueError("Use a loopback-only HTTP base URL.")
    opener = build_opener(ProxyHandler({}), NoRedirects())
    base = args.base_url.rstrip("/")
    with opener.open(base + "/health", timeout=10) as response:
        assert json.load(response)["status"] == "ready"
    rows = []
    result = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "response_checker_sha256": checker_hash,
              "case_sha256": hashlib.sha256(suite_bytes).hexdigest(),
              "method": "Sequential actual HTTP requests to an existing local API; startup excluded; one run per case.",
              "limitations": "Development cases with assistant/source-derived expectations. Source-fidelity checks do not establish relevance or general robustness.",
              "results": rows}
    for case in suite["cases"]:
        request = Request(base + "/api/query", method="POST",
                          data=json.dumps(case["request"], ensure_ascii=False).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        try:
            with opener.open(request, timeout=360) as response:
                status, payload = response.status, json.load(response)
        except HTTPError as error:
            status, payload = error.code, json.load(error)
        row = {"case_id": case["id"], "kind": case["kind"], "status": status,
               "seconds": time.perf_counter() - start, "response": payload,
               **inspect_response(case, status, payload)}
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        print(f"{case['id']} {case['kind']}: HTTP {status}; contracts={all(row['contract_checks'].values())}; observations={row['quality_observations']}", flush=True)


if __name__ == "__main__":
    main()
