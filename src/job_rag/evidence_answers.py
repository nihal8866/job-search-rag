"""Constrain local generation to selecting source lines, then render citations."""

import json
import re

from job_rag.answers import GeneratedAnswer


def selection_request(user_prompt: str) -> tuple[str, dict, dict[str, list[str]]]:
    """Build a per-source schema; the model cannot supply answer prose or IDs."""
    context = json.loads(user_prompt)
    sources = context["sources"]
    units = {}
    records = []
    for source in sources:
        job_id = source["job_id"]
        if not re.fullmatch(r"LF\d{4}", job_id) or job_id in units:
            raise ValueError("Evidence sources require distinct valid job IDs.")
        lines = [line.strip() for line in source["evidence"].splitlines() if line.strip()]
        if not lines:
            raise ValueError("Evidence sources must contain nonempty text.")
        for key, label in [("company", "Company"), ("location", "Location"),
                           ("level", "Job level"), ("publication_date", "Publication date")]:
            if source.get(key) is not None:
                lines.append(f"{label}: {source[key]}")
        units[job_id] = lines
        records.append({"job_id": job_id, "title": source["title"],
                        "lines": [{"number": i, "text": line} for i, line in enumerate(lines)]})
    if not units:
        raise ValueError("At least one evidence source is required.")
    schema = {
        "type": "object",
        "properties": {job_id: {"type": "array", "items": {"type": "integer", "enum": list(range(len(lines)))},
                                "maxItems": 3, "uniqueItems": True}
                       for job_id, lines in units.items()},
        "required": list(units), "additionalProperties": False,
    }
    prompt = json.dumps({"question": context["question"], "sources": records}, ensure_ascii=False)
    return prompt, schema, units


def render_selection(selected: object, units: dict[str, list[str]]) -> GeneratedAnswer:
    """Reject invalid selections and quote only the supplied source text."""
    if not isinstance(selected, dict) or set(selected) != set(units):
        raise RuntimeError("The local selection does not match the required schema.")
    paragraphs = []
    for job_id, lines in units.items():
        numbers = selected[job_id]
        if (not isinstance(numbers, list) or len(numbers) > 3
                or any(type(i) is not int or i < 0 or i >= len(lines) for i in numbers)
                or len(set(numbers)) != len(numbers)):
            raise RuntimeError("The local selection contains invalid source-line references.")
        if numbers:
            # Preserve source order, even when the model returns a different order.
            quotes = "\n".join("> " + line for i in sorted(numbers) for line in lines[i].splitlines())
            paragraphs.append(f"Selected evidence [{job_id}]:\n{quotes}")
        else:
            paragraphs.append(f"An answer is not established by the selected evidence for this job [{job_id}].")
    return GeneratedAnswer("\n\n".join(paragraphs), list(units))
