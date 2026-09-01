"""Choosing between trades that are all individually allowed.

Every other decision in this system is made about ONE trade in isolation. The
model judges a symbol without seeing the others; each gate asks whether this
contract is acceptable, never whether it is the best use of a slot. That works
until more candidates survive than there are slots to put them in -- and then
something has to choose, and until now that something was a sort on the model's
own confidence scores.

**Why that sort was weak.** Those scores come from separate calls. A 0.65 on
NVDA and a 0.65 on GLD were produced in different contexts, minutes apart, with
no knowledge of each other. They look comparable and are not. Models are
markedly better at "which of these five" than at emitting absolute numbers that
happen to sort correctly, so asking the comparative question directly is a
better use of the same model.

**What this is not.** It is not a gate and must never behave like one. Gates
refuse; this ranks. If the model could also veto here, a trade that did not
happen would be unattributable -- you could not tell whether a rule stopped it or
the model simply felt differently this pass, and "the gates dispose" would stop
being true. Anything that reaches this module has already been approved, and
every candidate remains eligible; the only question is order.

**When it runs.** Only when the slate is larger than the number of free slots.
With two candidates and five slots there is nothing to decide, and calling a
model to confirm that would be spending money to introduce doubt.

**How it fails.** Softly, to confidence order. This is the one component in the
system that fails SAFE rather than CLOSED, and the difference is principled: a
sensible default ordering already exists, whereas in the proposer the model's
absence means no decision exists at all.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from agent.domain import OrderDraft
from agent.models import ModelBackend
from agent.pricing import daily_decay_pct, premium_richness
from agent.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One approved trade, with everything needed to compare it to another."""

    draft: OrderDraft
    group: str

    @property
    def underlying(self) -> str:
        return self.draft.contract.underlying

    @property
    def confidence(self) -> float:
        return self.draft.proposal.confidence


SYSTEM_PROMPT = """You are the portfolio selection stage of an automated options trading agent.

Several candidate trades have each been independently analysed and have each passed every risk check the system applies -- position limits, liquidity, delta, expiry, premium richness, time decay, concentration and available cash. Every one of them is permitted. There are fewer free slots than candidates, so they must be put in order.

YOUR ONLY JOB IS TO RANK THEM.
You cannot reject a candidate, add one, change a size, or alter a contract. Anything you leave out of your ranking is simply placed last. The risk rules have already decided what is allowed; you are deciding what is best.

WHAT TO WEIGH
- Conviction, but not alone. These confidence figures were produced in separate analyses with no knowledge of each other, so treat them as one input rather than as a scale you can trust to sort correctly.
- Premium richness (IV/RV). Below 1.0 means the option is priced for less movement than the stock has actually been delivering -- the favourable side for a buyer. Above 1.3 means paying up.
- Daily decay. What the position loses each day the thesis takes to work.
- Cost. A cheaper expression of a similar view leaves room for another position.
- Correlation group. Two candidates in the same group are closer to one bet than two. Prefer a book that can be wrong about one thing without being wrong about everything.

A slightly lower-conviction trade at a genuinely cheap premium in an uncrowded group is usually a better use of a slot than a high-conviction one that is expensive and duplicates exposure you already have. Say so when that is your reasoning."""


def _rank_schema(symbols: list[str]) -> str:
    return ('{"ranking": ["SYMBOL", "SYMBOL", ...], '
            '"reasoning": "<one or two sentences>"}')


def render_slate(candidates: list[Candidate], settings: Settings,
                 slots: int, today) -> str:
    """The comparison table the model reads.

    One row per candidate with the numbers that actually distinguish them. Kept
    compact on purpose: the model has already read a full brief for each of
    these symbols, and repeating it would bury the comparison in context it has
    seen before.
    """
    lines = [
        f"{len(candidates)} approved candidates, {slots} free slot(s).",
        "All have passed every risk check. Rank them best-first.",
        "",
        f"  {'symbol':<8}{'dir':>5}{'conf':>6}{'IV/RV':>7}{'decay':>8}"
        f"{'cost':>9}  group",
    ]

    for c in candidates:
        contract = c.draft.contract
        richness = premium_richness(contract.implied_volatility, c.draft.realized_vol)
        decay = daily_decay_pct(contract, c.draft.spot, today)
        lines.append(
            f"  {c.underlying:<8}"
            f"{c.draft.proposal.direction.value:>5}"
            f"{c.confidence:>6.2f}"
            f"{(f'{richness:.2f}' if richness else '--'):>7}"
            f"{(f'{decay:.2%}' if decay else '--'):>8}"
            f"{c.draft.total_cost:>9,.0f}  {c.group}"
        )

    lines.append("")
    lines.append("Current book by group: " + (_book_summary(candidates) or "empty"))
    lines.append("")
    lines.append("Reply with JSON only, no prose or fence:")
    lines.append(_rank_schema([c.underlying for c in candidates]))
    return "\n".join(lines)


def _book_summary(candidates: list[Candidate]) -> str:
    counts: dict[str, int] = {}
    for c in candidates:
        counts[c.group] = counts.get(c.group, 0) + 1
    return ", ".join(f"{group} x{n}" for group, n in sorted(counts.items()))


def _extract_ranking(text: str) -> tuple[list[str], str]:
    """Pull the ordering out of the reply.

    Same defensive parsing as the proposer's OpenAI-compatible path, and for the
    same reason: a model without enforced structured output will sometimes fence
    its JSON or add a sentence in front of it.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if not (0 <= start < end):
        raise ValueError(f"no JSON object in the reply: {text[:120]}")

    payload = json.loads(cleaned[start:end + 1])
    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError("no ranking list in the reply")

    return ([str(s).strip().upper() for s in ranking],
            str(payload.get("reasoning", "")).strip())


def apply_ranking(candidates: list[Candidate], ordering: list[str]) -> list[Candidate]:
    """Reorder the slate, tolerating an imperfect answer.

    Three things a model gets wrong here, all handled rather than rejected:

      * a symbol it invented, which is dropped;
      * a symbol it repeated, where only the first mention counts;
      * a symbol it forgot, which keeps its place at the end in confidence order.

    Nothing is ever lost. A candidate omitted from the ranking is still eligible
    and still gets opened if a slot survives to reach it -- because this module
    ranks and does not refuse, and an omission is not a refusal.
    """
    by_symbol = {c.underlying: c for c in candidates}
    ranked: list[Candidate] = []
    seen: set[str] = set()

    for symbol in ordering:
        candidate = by_symbol.get(symbol)
        if candidate is not None and symbol not in seen:
            ranked.append(candidate)
            seen.add(symbol)

    leftovers = sorted((c for c in candidates if c.underlying not in seen),
                       key=lambda c: c.confidence, reverse=True)
    return ranked + leftovers


def by_confidence(candidates: list[Candidate]) -> list[Candidate]:
    """The default ordering, and the fallback whenever selection fails."""
    return sorted(candidates, key=lambda c: c.confidence, reverse=True)


def select(candidates: list[Candidate], *, backend: ModelBackend | None,
           settings: Settings, slots: int, today) -> tuple[list[Candidate], str]:
    """Order the slate, and say how the order was arrived at.

    Returns the ranking and a note for the journal. The note matters: a pass
    where the agent chose GLD over NVDA should record why, and "ranked by
    confidence (no selection needed)" is as useful an answer as the model's own
    reasoning.
    """
    if len(candidates) <= 1:
        return candidates, "single candidate, nothing to rank"

    # Nothing is competing for anything. Ranking would cost a call and could
    # only introduce doubt about a decision the gates have already made.
    if slots >= len(candidates):
        return by_confidence(candidates), (
            f"{len(candidates)} candidates for {slots} slots -- all fit, "
            f"ordered by conviction")

    if backend is None:
        return by_confidence(candidates), "ranked by conviction (no selector configured)"

    try:
        answer = backend.ask_text(
            SYSTEM_PROMPT, render_slate(candidates, settings, slots, today))
        ordering, reasoning = _extract_ranking(answer)
    except Exception as exc:
        logger.warning("selection failed, falling back to confidence: %s", exc)
        return by_confidence(candidates), (
            f"ranked by conviction -- selection unavailable ({type(exc).__name__})")

    ranked = apply_ranking(candidates, ordering)
    order = " > ".join(c.underlying for c in ranked[:slots])
    return ranked, f"selected {order}. {reasoning}"[:600]
