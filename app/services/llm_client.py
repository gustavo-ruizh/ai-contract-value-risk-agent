import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_FENCE = "```"


class LLMError(Exception):
    """Raised when the LLM call fails or returns unparseable output."""


class LLMClient:
    """
    Thin wrapper around the Anthropic messages API.
    Exposes one method: generate_json(), which always returns a parsed dict.
    The underlying client is lazy-imported so the module loads without anthropic installed.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                # httpx_client with no read timeout — streaming handles long responses
                # without a hard wall-clock cutoff.
                import httpx
                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    http_client=httpx.Client(timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)),
                )
            except ImportError:
                raise LLMError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        timeout: int = 20,
    ) -> dict:
        """
        Send prompt to the model and return a parsed JSON dict.
        Retries once on transient failure. Raises LLMError on persistent failure.
        """
        if not self.api_key:
            raise LLMError(
                "No Anthropic API key configured. Set ANTHROPIC_API_KEY in your environment."
            )

        system_prompt = system or (
            "You are a contract analysis assistant. "
            "Always respond with valid JSON only. No markdown fences, no explanation."
        )

        for attempt in range(2):
            try:
                client = self._get_client()
                # Use streaming so the connection stays alive for large prompts
                # and long responses — avoids wall-clock timeout errors.
                with client.messages.stream(
                    model=self.model,
                    max_tokens=8192,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    text = stream.get_final_text().strip()

                # Strip markdown code fences if the model adds them
                if text.startswith(_FENCE):
                    lines = text.splitlines()
                    inner = [
                        ln for ln in lines[1:]
                        if not ln.strip().startswith(_FENCE)
                    ]
                    text = "\n".join(inner).strip()

                text = _fix_mojibake(text)
                parsed = json.loads(text)
                return _fix_mojibake_obj(parsed)

            except json.JSONDecodeError as exc:
                logger.warning("LLM returned invalid JSON on attempt %d: %s", attempt + 1, exc)
                if attempt == 1:
                    raise LLMError(f"LLM returned invalid JSON after retry: {exc}") from exc

            except LLMError:
                raise

            except Exception as exc:
                logger.error("LLM API call failed on attempt %d: %s", attempt + 1, exc)
                if attempt == 1:
                    raise LLMError(f"LLM API call failed after retry: {exc}") from exc
                time.sleep(1)

        raise LLMError("generate_json failed after retries")


# --- helpers ---

# Mojibake: UTF-8 typographic characters whose bytes were decoded as Latin-1.
# U+2014 em dash (UTF-8: E2 80 94) arrives as U+00E2 U+0080 U+0094, etc.
_MOJIBAKE = (
    ("â", "—"),  # em dash
    ("â", "–"),  # en dash
    ("â", "’"),  # right single quotation mark
    ("â", "‘"),  # left single quotation mark
    ("â", "“"),  # left double quotation mark
    ("â", "”"),  # right double quotation mark
    ("â¦", "…"),  # horizontal ellipsis
    ("Â ",       " "),  # non-breaking space
)


def _fix_mojibake(text: str) -> str:
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)
    return text


def _fix_mojibake_obj(obj):
    """Recursively fix mojibake in all string values of a parsed JSON structure."""
    if isinstance(obj, str):
        return _fix_mojibake(obj)
    if isinstance(obj, dict):
        return {k: _fix_mojibake_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_mojibake_obj(item) for item in obj]
    return obj
