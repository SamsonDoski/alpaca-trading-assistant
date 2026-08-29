"""Tests for the Discord notifier.

Nothing here reaches the network -- `_post` is replaced, and the tests assert on
the payloads that would have been sent.

The chunking tests exist because of a real failure: on a pass where all nine
symbols declined, the summary silently dropped the ninth and cut every rationale
at 110 characters. A pass that trades nothing is only legible through its
reasons, so losing them loses the whole message.
"""

from __future__ import annotations

from agent.notify import Notifier, _chunked

LONG = ("GOOGL is chopping in a tight band (last 10 closes all $340-348) at 34% of its "
        "82-day range with mildly negative 20- and 60-day drift, so there is no trend to "
        "lean on either way. Headlines are mostly about Marvell's Google TPU relationship "
        "rather than Google itself, and semiconductor tariff risk cuts the other way.")


class Recording(Notifier):
    """A notifier that keeps its payloads instead of posting them."""

    def __init__(self):
        super().__init__("https://example.invalid/webhook")
        self.posted: list[dict] = []

    def _post(self, payload):
        self.posted.append(payload)

    def descriptions(self) -> list[str]:
        return [p["embeds"][0]["description"] for p in self.posted]

    @property
    def everything(self) -> str:
        return "\n".join(self.descriptions())


def summarise(notifier, refusals):
    notifier.pass_summary(
        equity=93_196, available=84_491, open_count=0, opened=0, closed=0,
        considered=len(refusals), realized=0.0, unrealized=0.0,
        book=[], refusals=refusals, dry_run=True)


# --- The bug this file exists for -----------------------------------------

def test_every_decline_appears_even_when_all_nine_symbols_decline():
    """Nine symbols, nine declines. The ninth used to vanish."""
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"]
    notifier = Recording()
    summarise(notifier, [(s, LONG) for s in symbols])

    for symbol in symbols:
        assert symbol in notifier.everything, f"{symbol} was dropped from the summary"


def test_a_rationale_is_never_cut_mid_sentence():
    notifier = Recording()
    summarise(notifier, [("GOOGL", LONG)])
    assert LONG in notifier.everything


def test_long_declines_are_split_across_messages_rather_than_trimmed():
    notifier = Recording()
    summarise(notifier, [(f"SYM{i}", LONG) for i in range(20)])
    assert len(notifier.posted) > 2          # summary plus several decline cards
    for i in range(20):
        assert f"SYM{i}" in notifier.everything


def test_the_headline_card_comes_first():
    notifier = Recording()
    summarise(notifier, [("GOOGL", LONG)])
    assert "equity" in notifier.descriptions()[0]


def test_no_decline_card_is_posted_when_nothing_was_declined():
    notifier = Recording()
    summarise(notifier, [])
    assert len(notifier.posted) == 1


# --- The chunking rule itself ---------------------------------------------

def test_chunking_splits_between_blocks_never_inside_one():
    blocks = ["a" * 300 for _ in range(10)]
    for batch in _chunked(blocks, 1000):
        for block in batch:
            assert block == "a" * 300


def test_every_chunk_fits_the_limit():
    blocks = [f"block {i} " + "x" * 500 for i in range(12)]
    for batch in _chunked(blocks, 1000):
        assert len("\n\n".join(batch)) <= 1000


def test_no_block_is_lost_in_chunking():
    blocks = [f"block-{i}" for i in range(50)]
    flattened = [b for batch in _chunked(blocks, 100) for b in batch]
    assert flattened == blocks


def test_a_single_oversized_block_is_marked_as_truncated():
    """The one case where cutting is unavoidable -- say so rather than let the
    reader think the model stopped mid-thought."""
    [batch] = _chunked(["y" * 5000], 1000)
    assert batch[0].endswith("...")
    assert len(batch[0]) <= 1000


def test_an_empty_list_produces_no_batches():
    assert _chunked([], 1000) == []


# --- Fail-open -------------------------------------------------------------

def test_a_notifier_with_no_webhook_does_nothing_and_does_not_raise():
    quiet = Notifier(None)
    assert not quiet.enabled
    quiet.pass_summary(equity=1, available=1, open_count=0, opened=0, closed=0,
                       considered=0, realized=0.0, unrealized=0.0)
    quiet.alert("something happened")


def test_a_broken_webhook_never_raises_into_the_trading_loop():
    """A chat service being down must not stop a position being managed."""
    broken = Notifier("http://127.0.0.1:1/does-not-exist", timeout=0.2)
    broken.alert("this will fail to send")
    broken.closed("AAPL261016C00310000", "stop loss", won=False)
