"""Where the judgement comes from, and how to change it without changing the agent.

The proposer needs one thing from a language model: a direction, a conviction,
and a sentence of reasoning. Everything else -- which vendor, which API shape,
whether structured output is enforced by the server or parsed by us -- is an
implementation detail that belongs behind an interface.

This module is that interface. `ModelBackend` is the whole contract:

    ask(system, user) -> RawVerdict

Two implementations ship. The Anthropic one uses schema-validated structured
output, so a malformed answer is impossible by construction. The OpenAI-compatible
one (Featherless, and anything else with that API shape) asks for JSON in the
prompt and parses it defensively, because open models served that way do not all
enforce a schema.

**That difference is real and worth stating plainly rather than papering over.**
On Anthropic the response cannot be malformed; on Featherless it can, and when it
is, the parse fails and the proposal comes back as "no view". The safety property
is the same either way -- a failure means no trade -- but the failure rate is not,
and the journal will show which backend produced each decision.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

ANTHROPIC_MODEL = "claude-opus-5"
ANTHROPIC_EFFORT = "high"

FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"
FEATHERLESS_MODEL = "Qwen/Qwen2.5-72B-Instruct"

MAX_TOKENS = 16_000

# Generous on purpose. A reasoning model spends most of its output budget inside
# its <think> block before writing a single character of the answer, and R1-class
# models routinely think for several thousand tokens. At the 2,000 this started
# at, the reasoning consumed the whole allowance and the JSON was truncated --
# which surfaces as "no JSON object in the reply" on every single call, looking
# like a parsing bug rather than a budget one.
OPENAI_MAX_TOKENS = 8_000


class ModelUnavailable(RuntimeError):
    """The model could not be reached, or would not answer usefully."""


@dataclass(frozen=True, slots=True)
class RawVerdict:
    """What a backend returns, before the agent gives it any meaning."""

    direction: str           # "up", "down" or "none"
    confidence: float
    rationale: str
    reasoning: str = ""      # the model's own account of how it decided, if any


class ModelBackend(Protocol):
    """The entire contract between the agent and a language model.

    Two methods, because there are two genuinely different questions. `ask`
    wants a directional verdict and gets a schema-validated one where the
    provider supports it. `ask_text` wants free-form output for a question that
    is not a verdict -- ranking a slate, for instance -- and returns whatever
    the model said, leaving the caller to parse it.
    """

    name: str

    def ask(self, system: str, user: str) -> RawVerdict: ...

    def ask_text(self, system: str, user: str) -> str: ...


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------

class AnthropicBackend:
    """Claude, with the answer validated against a schema by the API itself."""

    def __init__(self, client=None, *, model: str = ANTHROPIC_MODEL,
                 effort: str = ANTHROPIC_EFFORT) -> None:
        self.name = f"anthropic:{model}"
        self._client = client
        self._model = model
        self._effort = effort

    def _connect(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def ask(self, system: str, user: str) -> RawVerdict:
        from agent.proposer import ProposalSchema

        response = self._connect().messages.parse(
            model=self._model,
            max_tokens=MAX_TOKENS,
            # Adaptive thinking lets the model decide how hard to think.
            # `display: summarized` is the operational half: without it the
            # reasoning happens but returns empty, and the journal would record
            # a decision with no account of why.
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self._effort},
            system=[{
                "type": "text",
                "text": system,
                # Every symbol in a pass shares this prompt, so caching it means
                # it is charged once rather than thirty times. Works only
                # because nothing volatile sits above this point.
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user}],
            output_format=ProposalSchema,
        )

        # A safety classifier may decline. That arrives as a normal 200, so it
        # has to be checked rather than caught.
        if getattr(response, "stop_reason", None) == "refusal":
            raise ModelUnavailable("the model declined to answer")

        verdict = getattr(response, "parsed_output", None)
        if verdict is None:
            raise ModelUnavailable("no structured answer returned")

        return RawVerdict(
            direction=verdict.direction,
            confidence=float(verdict.confidence),
            rationale=verdict.rationale,
            reasoning=_anthropic_thinking(response),
        )


    def ask_text(self, system: str, user: str) -> str:
        """A free-form answer, for questions that are not directional verdicts.

        No structured output here on purpose: the schema describes a proposal,
        and forcing an unrelated question through it would produce a verdict
        about nothing.
        """
        response = self._connect().messages.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self._effort},
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise ModelUnavailable("the model declined to answer")

        parts = [block.text for block in getattr(response, "content", []) or []
                 if getattr(block, "type", None) == "text"]
        text = "\n".join(parts).strip()
        if not text:
            raise ModelUnavailable("empty answer returned")
        return text


def _anthropic_thinking(response) -> str:
    parts = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "thinking":
            text = (getattr(block, "thinking", "") or "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# OpenAI-compatible: Featherless, and anything else with that API shape
# --------------------------------------------------------------------------

# Appended to the system prompt for backends that cannot enforce a schema. The
# agent's own prompt says nothing about JSON, because on Anthropic the format is
# guaranteed by the API and saying so would be noise.
JSON_INSTRUCTION = """

OUTPUT FORMAT
Reply with a single JSON object and nothing else. No prose before or after it, no markdown fence.

{"direction": "up" | "down" | "none", "confidence": <number between 0 and 1>, "rationale": "<one or two sentences>"}

Use "none" with confidence 0 when the evidence does not support either side."""


class OpenAICompatibleBackend:
    """Any provider speaking the OpenAI chat-completions API.

    Written against the shape rather than the vendor, so pointing it at a
    different host is a base URL change. Featherless is the one in use; the same
    class would serve OpenRouter, Together, or a local server.
    """

    def __init__(self, client=None, *, model: str = FEATHERLESS_MODEL,
                 base_url: str = FEATHERLESS_BASE_URL,
                 api_key: str | None = None,
                 temperature: float = 0.3,
                 max_tokens: int = OPENAI_MAX_TOKENS,
                 label: str = "featherless") -> None:
        self.name = f"{label}:{model}"
        self._client = client
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        # Low but not zero. This is a judgement task, and a deterministic
        # sampler on an open model tends to produce the same hedge every time
        # rather than a considered answer.
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _connect(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    def ask_text(self, system: str, user: str) -> str:
        """A free-form answer. No JSON instruction is appended -- the caller
        states its own format, because the shape it wants is not a verdict."""
        return _openai_text(self, system, user)

    def ask(self, system: str, user: str) -> RawVerdict:
        response = self._connect().chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system + JSON_INSTRUCTION},
                {"role": "user", "content": user},
            ],
        )

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ModelUnavailable("no answer returned")

        message = choices[0].message
        text = (getattr(message, "content", "") or "").strip()
        if not text:
            raise ModelUnavailable("empty answer returned")

        # A reasoning model's working arrives one of two ways, and which one
        # depends on the host rather than the model. DeepSeek-style APIs return
        # it in a separate field and strip the tags from the content; others
        # leave <think> blocks inline. Check the field first, then the tags,
        # because a model that does neither must not silently lose its
        # reasoning -- that record is most of what makes the agent reviewable.
        reasoning = _reasoning_field(message)
        inline, text = _split_thinking(text)
        reasoning = reasoning or inline
        payload = _extract_json(text)

        direction = str(payload.get("direction", "none")).strip().lower()
        if direction not in ("up", "down", "none"):
            raise ModelUnavailable(f"unusable direction {direction!r}")

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            raise ModelUnavailable("confidence was not a number") from None

        return RawVerdict(
            direction=direction,
            # Clamped rather than rejected. A model answering 1.2 means "very
            # sure", and refusing the whole answer over a value out of range
            # would discard reasoning that is otherwise fine.
            confidence=max(0.0, min(1.0, confidence)),
            rationale=str(payload.get("rationale", "")).strip()[:800],
            reasoning=reasoning,
        )


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _reasoning_field(message) -> str:
    """The separate reasoning field, if this host uses one.

    `reasoning_content` is DeepSeek's spelling and the most common; `reasoning`
    is used by several gateways. Both are checked because the agent should not
    have to know which host is behind the base URL.
    """
    for attribute in ("reasoning_content", "reasoning"):
        value = getattr(message, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _openai_text(backend, system: str, user: str) -> str:
    """Shared plumbing for a free-form call on an OpenAI-compatible host."""
    response = backend._connect().chat.completions.create(
        model=backend._model,
        max_tokens=backend._max_tokens,
        temperature=backend._temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ModelUnavailable("no answer returned")
    text = (getattr(choices[0].message, "content", "") or "").strip()
    if not text:
        raise ModelUnavailable("empty answer returned")
    return text


def _split_thinking(text: str) -> tuple[str, str]:
    """Separate a reasoning model's <think> block from its answer.

    Several open reasoning models emit their working this way. Pulling it out
    serves two purposes: the JSON parser sees only the answer, and the journal
    gets the reasoning -- the same record the Anthropic backend provides through
    summarised thinking.
    """
    blocks = _THINK_BLOCK.findall(text)
    if not blocks:
        return "", text
    return "\n\n".join(b.strip() for b in blocks), _THINK_BLOCK.sub("", text).strip()


def _extract_json(text: str) -> dict:
    """Find the JSON object in a reply that was asked for JSON only.

    Necessary because a model without enforced structured output will sometimes
    wrap it in a markdown fence or add a sentence of preamble regardless of
    instruction. Tries the whole string first, then the outermost braced span.

    This is the exact cleanup the Anthropic path does not need, and the clearest
    illustration of what schema-validated output actually buys.
    """
    for candidate in (text, _braced_span(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ModelUnavailable(f"no JSON object in the reply: {text[:160]}")


def _braced_span(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else ""


# --------------------------------------------------------------------------
# Choosing one
# --------------------------------------------------------------------------

def build_backend(provider: str | None = None) -> ModelBackend:
    """The backend named by the environment.

        MODEL_PROVIDER=anthropic     (default) Claude, schema-validated
        MODEL_PROVIDER=featherless   an open model over the OpenAI-compatible API

    Chosen by environment rather than by config file so that switching provider
    needs no code change, no redeploy, and no edit to a file under version
    control -- it takes effect on the next scheduled pass.
    """
    provider = (provider or os.getenv("MODEL_PROVIDER") or "anthropic").strip().lower()

    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ModelUnavailable("MODEL_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        return AnthropicBackend()

    if provider in ("featherless", "openai-compatible"):
        api_key = os.getenv("FEATHERLESS_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ModelUnavailable(
                f"MODEL_PROVIDER={provider} but FEATHERLESS_API_KEY is not set")
        return OpenAICompatibleBackend(
            model=os.getenv("FEATHERLESS_MODEL", FEATHERLESS_MODEL),
            base_url=os.getenv("FEATHERLESS_BASE_URL", FEATHERLESS_BASE_URL),
            api_key=api_key,
            temperature=float(os.getenv("FEATHERLESS_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("FEATHERLESS_MAX_TOKENS", OPENAI_MAX_TOKENS)),
            label=provider,
        )

    raise ModelUnavailable(
        f"unknown MODEL_PROVIDER {provider!r}; use 'anthropic' or 'featherless'")
