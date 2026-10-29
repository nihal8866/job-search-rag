"""Separate HTTP/source-fidelity checks from exploratory relevance observations."""

import re


def inspect_response(case: dict, status: int, payload: dict) -> dict:
    checks = {"expected_status": status == case["status"]}
    observations = {}
    if status != 200:
        return {"contract_checks": checks, "quality_observations": observations}
    jobs = payload.get("jobs", [])
    ids = [job["job_id"] for job in jobs]
    answer = payload.get("answer", "")
    checks["nonempty_answer"] = bool(answer.strip())
    checks["request_fields_match"] = (
        payload.get("query") == case["request"]["query"].strip()
        and payload.get("location_filter") == case["request"].get("location")
    )
    declared = payload.get("cited_job_ids", [])
    nonquoted = "\n".join(line for line in answer.splitlines() if not line.startswith("> "))
    inline = set(re.findall(r"\[([^\[\]\r\n]+)\]", nonquoted))
    checks["distinct_jobs_and_limit"] = len(ids) == len(set(ids)) and len(ids) <= case["request"].get("top_k", 2)
    checks["citations_match_jobs"] = set(declared) == inline == set(ids)
    location = case["request"].get("location")
    if location:
        checks["literal_location_filter"] = all(location.casefold() in (job.get("location") or "").casefold() for job in jobs)

    allowed = {}
    for job in jobs:
        lines = {line.strip() for line in job["evidence"].splitlines() if line.strip()}
        for key, label in [("company", "Company"), ("location", "Location"),
                           ("level", "Job level"), ("publication_date", "Publication date")]:
            if job.get(key) is not None:
                lines.update(f"{label}: {job[key]}".splitlines())
        allowed[job["job_id"]] = lines
    active = None
    quotes = []
    fidelity = True
    constrained = True
    for line in answer.splitlines():
        heading = re.fullmatch(r"Selected evidence \[(LF\d{4})\]:", line)
        if heading:
            active = heading.group(1)
        elif line.startswith("> "):
            quote = line[2:]
            quotes.append(quote)
            fidelity = fidelity and quote in allowed.get(active, set())
        elif not line.strip():
            active = None
        elif re.fullmatch(r"An answer is not established by the selected evidence for this job \[LF\d{4}\]\.", line):
            active = None
        elif not jobs and line == "No jobs were retrieved for the requested search and filters.":
            active = None
        else:
            constrained = False
    checks["quoted_lines_match_cited_evidence"] = fidelity
    checks["constrained_answer_format"] = constrained
    if case.get("expect_empty_jobs"):
        checks["empty_jobs"] = not jobs and not declared
    if "target_job_id" in case:
        target = case["target_job_id"]
        observations["target_rank"] = ids.index(target) + 1 if target in ids else None
    if case.get("expect_no_quotes"):
        observations["expected_abstention"] = not quotes
    if "expected_quote" in case:
        observations["expected_quote_selected"] = case["expected_quote"] in quotes
    return {"contract_checks": checks, "quality_observations": observations}
