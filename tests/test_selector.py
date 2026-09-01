"""Tests for the portfolio selection stage.

The invariant this file exists to protect: **the selector ranks and never
refuses.** Every candidate reaching it has already been approved by the gates,
and a trade that does not happen must always be attributable to a rule rather
than to a model's mood on one particular pass.
"""

from __future__ import annotations

from datetime import date

from agent.domain import Direction, OptionContract, OrderDraft, Proposal
from agent.selector import (
    Candidate,
    apply_ranking,
    by_confidence,
    render_slate,
    select,
)
from agent.settings import Settings

TODAY = date(2026, 9, 4)
SETTINGS = Settings()


def candidate(symbol: str, confidence: float = 0.7, group: str = "tech",
              iv: float = 0.20, realized: float = 0.16,
              cost_each: float = 15.75) -> Candidate:
    contract = OptionContract(f"{symbol}261016C00310000", symbol, "call", 310.0,
                              date(2026, 10, 16), cost_each - 0.25, cost_each + 0.25,
                              0.65, iv, 900)
    proposal = Proposal(symbol, Direction.UP, confidence, "test view")
    draft = OrderDraft(proposal, contract, 2, cost_each, spot=313.0,
                       realized_vol=realized)
    return Candidate(draft=draft, group=group)


class FakeBackend:
    name = "fake:selector"

    def __init__(self, answer=None, error=None):
        self.answer = answer
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def ask(self, system, user):
        raise AssertionError("the selector must use ask_text, not ask")

    def ask_text(self, system, user):
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        return self.answer


def ranking_of(candidates) -> list[str]:
    return [c.underlying for c in candidates]


# --- When it runs at all ---------------------------------------------------

def test_no_model_call_when_everything_fits():
    """Three candidates and five slots is not a choice. Calling a model to
    confirm that would spend money to introduce doubt."""
    backend = FakeBackend('{"ranking": ["C", "B", "A"]}')
    slate = [candidate("A", 0.9), candidate("B", 0.8), candidate("C", 0.7)]
    ordered, note = select(slate, backend=backend, settings=SETTINGS,
                           slots=5, today=TODAY)
    assert backend.calls == []
    assert ranking_of(ordered) == ["A", "B", "C"]
    assert "all fit" in note


def test_a_single_candidate_is_never_ranked():
    backend = FakeBackend('{"ranking": ["A"]}')
    ordered, note = select([candidate("A")], backend=backend, settings=SETTINGS,
                           slots=1, today=TODAY)
    assert backend.calls == []
    assert "nothing to rank" in note


def test_the_model_is_consulted_when_slots_are_scarce():
    backend = FakeBackend('{"ranking": ["C", "A", "B"], "reasoning": "C is cheapest"}')
    slate = [candidate("A", 0.9), candidate("B", 0.8), candidate("C", 0.7)]
    ordered, note = select(slate, backend=backend, settings=SETTINGS,
                           slots=1, today=TODAY)
    assert len(backend.calls) == 1
    assert ranking_of(ordered) == ["C", "A", "B"]
    assert "C is cheapest" in note


def test_the_chosen_order_is_recorded_for_the_journal():
    """A pass that chose GLD over NVDA should say why."""
    backend = FakeBackend('{"ranking": ["C", "A"], "reasoning": "cheaper premium"}')
    slate = [candidate("A", 0.9), candidate("C", 0.7)]
    _, note = select(slate, backend=backend, settings=SETTINGS, slots=1, today=TODAY)
    assert "selected C" in note
    assert "cheaper premium" in note


# --- Ranking, not refusing -------------------------------------------------

def test_a_candidate_left_out_of_the_ranking_is_kept_not_dropped():
    """An omission is not a refusal. It still opens if a slot reaches it."""
    slate = [candidate("A", 0.6), candidate("B", 0.9), candidate("C", 0.7)]
    ordered = apply_ranking(slate, ["C"])
    assert ranking_of(ordered) == ["C", "B", "A"]      # B and A by confidence


def test_every_candidate_survives_any_ranking():
    slate = [candidate("A"), candidate("B"), candidate("C")]
    for ordering in ([], ["B"], ["C", "A"], ["A", "B", "C"]):
        assert len(apply_ranking(slate, ordering)) == 3


def test_an_invented_symbol_is_ignored():
    slate = [candidate("A"), candidate("B")]
    assert ranking_of(apply_ranking(slate, ["ZZZZ", "B", "A"])) == ["B", "A"]


def test_a_repeated_symbol_counts_once():
    slate = [candidate("A"), candidate("B")]
    assert ranking_of(apply_ranking(slate, ["A", "A", "B"])) == ["A", "B"]


def test_case_and_whitespace_in_the_answer_are_tolerated():
    backend = FakeBackend('{"ranking": [" b ", "a"]}')
    slate = [candidate("A", 0.5), candidate("B", 0.4)]
    ordered, _ = select(slate, backend=backend, settings=SETTINGS,
                        slots=1, today=TODAY)
    assert ranking_of(ordered) == ["B", "A"]


# --- Failing safe, not closed ---------------------------------------------

def test_a_failed_selection_falls_back_to_confidence():
    """The one component that fails SAFE rather than CLOSED: a sensible default
    ordering already exists, so refusing to trade would be the wrong answer."""
    backend = FakeBackend(error=RuntimeError("rate limited"))
    slate = [candidate("A", 0.6), candidate("B", 0.9)]
    ordered, note = select(slate, backend=backend, settings=SETTINGS,
                           slots=1, today=TODAY)
    assert ranking_of(ordered) == ["B", "A"]
    assert "selection unavailable" in note


def test_prose_instead_of_json_falls_back():
    backend = FakeBackend("I think you should buy B, honestly.")
    slate = [candidate("A", 0.6), candidate("B", 0.9)]
    ordered, note = select(slate, backend=backend, settings=SETTINGS,
                           slots=1, today=TODAY)
    assert ranking_of(ordered) == ["B", "A"]
    assert "unavailable" in note


def test_no_backend_configured_uses_confidence():
    slate = [candidate("A", 0.6), candidate("B", 0.9)]
    ordered, note = select(slate, backend=None, settings=SETTINGS,
                           slots=1, today=TODAY)
    assert ranking_of(ordered) == ["B", "A"]
    assert "no selector configured" in note


def test_a_fenced_answer_still_parses():
    backend = FakeBackend('```json\n{"ranking": ["B", "A"]}\n```')
    slate = [candidate("A", 0.9), candidate("B", 0.5)]
    ordered, _ = select(slate, backend=backend, settings=SETTINGS,
                        slots=1, today=TODAY)
    assert ranking_of(ordered) == ["B", "A"]


def test_a_reasoning_block_is_stripped_before_parsing():
    backend = FakeBackend('<think>weighing them up</think>{"ranking": ["B", "A"]}')
    slate = [candidate("A", 0.9), candidate("B", 0.5)]
    ordered, _ = select(slate, backend=backend, settings=SETTINGS,
                        slots=1, today=TODAY)
    assert ranking_of(ordered) == ["B", "A"]


# --- What the model is shown ----------------------------------------------

def test_the_slate_shows_the_numbers_that_distinguish_candidates():
    slate = [candidate("NVDA", 0.7, group="semis"),
             candidate("GLD", 0.6, group="metals")]
    text = render_slate(slate, SETTINGS, slots=1, today=TODAY)
    assert "NVDA" in text and "GLD" in text
    assert "semis" in text and "metals" in text
    assert "IV/RV" in text and "decay" in text


def test_the_slate_states_how_many_slots_are_contested():
    text = render_slate([candidate("A"), candidate("B")], SETTINGS,
                        slots=1, today=TODAY)
    assert "2 approved candidates, 1 free slot" in text


def test_the_prompt_forbids_rejection_explicitly():
    from agent.selector import SYSTEM_PROMPT
    assert "cannot reject" in SYSTEM_PROMPT.lower()
    assert "only job is to rank" in SYSTEM_PROMPT.lower()


def test_confidence_ordering_is_stable_and_descending():
    slate = [candidate("A", 0.3), candidate("B", 0.9), candidate("C", 0.6)]
    assert ranking_of(by_confidence(slate)) == ["B", "C", "A"]
