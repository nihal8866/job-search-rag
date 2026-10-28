"""Structured answer generation through loopback-only Ollama HTTP calls."""

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from job_rag.answers import GeneratedAnswer
from job_rag.config import Settings


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "minLength": 1},
        "cited_job_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["text", "cited_job_ids"],
    "additionalProperties": False,
}
LOCAL_MODEL = "qwen3:4b-instruct-2507-q4_K_M"


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OllamaGenerator:
    """Use the downloaded local model without credentials or a cloud fallback."""

    def __init__(self, settings: Settings):
        parsed = urlsplit(settings.llm_base_url)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"}):
            raise ValueError("Ollama must use an HTTP endpoint on 127.0.0.1 only.")
        if settings.llm_model != LOCAL_MODEL:
            raise ValueError("Only the selected downloaded local Qwen model is enabled.")
        if settings.llm_timeout_seconds <= 0 or settings.llm_max_output_tokens < 1:
            raise ValueError("LLM timeout and output-token limit must be positive.")
        if settings.llm_context_length <= settings.llm_max_output_tokens + 128:
            raise ValueError("The LLM context must leave space for the input and output.")
        self.settings = settings
        self.endpoint = settings.llm_base_url.rstrip("/") + "/api/chat"
        # Do not route local evidence through environment proxies or HTTP redirects.
        self.opener = build_opener(ProxyHandler({}), NoRedirects())

    def generate(self, system_prompt: str, user_prompt: str) -> GeneratedAnswer:
        system = (
            system_prompt + "\nReturn one JSON object with this schema:\n"
            + json.dumps(ANSWER_SCHEMA)
        )
        # A conservative byte budget for this byte-tokenized model avoids silently
        # losing source text. It reserves output tokens and chat-template overhead.
        input_bytes = len((system + user_prompt).encode("utf-8"))
        budget = self.settings.llm_context_length - self.settings.llm_max_output_tokens - 128
        if input_bytes > budget:
            raise ValueError("The evidence context is too large. Reduce top_k or shorten the query.")

        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "format": ANSWER_SCHEMA,
            "stream": False,
            "keep_alive": 0,
            "options": {
                "temperature": 0,
                "num_ctx": self.settings.llm_context_length,
                "num_predict": self.settings.llm_max_output_tokens,
            },
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=self.settings.llm_timeout_seconds) as response:
                body = response.read(2_000_001)
        except HTTPError as error:
            raise RuntimeError(f"Local Ollama returned HTTP {error.code}.") from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError("Cannot reach local Ollama or generation timed out. Check ollama serve.") from None

        if len(body) > 2_000_000:
            raise RuntimeError("Local Ollama returned an oversized response.")
        try:
            response_data = json.loads(body)
            if response_data.get("done") is not True or response_data.get("done_reason") == "length":
                raise RuntimeError("Local Ollama did not finish the answer within its output limit.")
            generated = json.loads(response_data["message"]["content"])
        except (ValueError, KeyError, TypeError, AttributeError):
            raise RuntimeError("Local Ollama returned an invalid structured response.") from None

        if (not isinstance(generated, dict) or set(generated) != {"text", "cited_job_ids"}
                or not isinstance(generated["text"], str)
                or not isinstance(generated["cited_job_ids"], list)
                or not all(isinstance(item, str) for item in generated["cited_job_ids"])):
            raise RuntimeError("The local answer does not match the required schema.")
        return GeneratedAnswer(generated["text"], generated["cited_job_ids"])
