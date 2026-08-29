"""Every tunable number in one place, with the reason it has the value it has.

Two things are deliberate here.

First, **the settings object is immutable and passed down**, never read from a
global. A function that needs a limit takes it as an argument, which is what
makes the gates testable: a test constructs a Settings with the values it wants
to exercise and never touches a config file.

Second, **the reasoning lives next to the number**. A bare `stop_loss = 0.25` in
a config file tells a reader what the code will do but not whether they may
change it. The comments below record what was measured, so a future reader can
tell the difference between a value that was chosen and a value that was merely
typed. Several of these were carried over from the earlier equities-and-options
research project and are marked with what that measurement showed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class Settings:
    """The complete rule set the agent runs under."""

    # --- Universe -----------------------------------------------------------
    # Liquid names only. Measuring quoted spreads across a wider list showed
    # several names quoting 5-7% wide, which is more than the edge on an average
    # trade. Those were removed rather than traded carefully.
    symbols: tuple[str, ...] = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA",
                                "AMZN", "META", "GOOGL", "TSLA")

    # --- Entry screening ----------------------------------------------------
    # Five concurrent positions. Tested against 8, 10 and 15 over 2.4 years and
    # on both halves independently; 5 won in each half separately, which is what
    # distinguishes it from one period's accident. Raising the cap does not add
    # more of the same trades, it reaches into lower-ranked candidates that are
    # worse than the ones already being taken. The cap screens for quality.
    max_positions: int = 5

    # Days to wait before re-entering an underlying after a STOP LOSS exit.
    # Never applied after a win: a win means the thesis worked. Added because a
    # live run re-bought the same name minutes after stopping out on the same
    # signal. No single value is provably optimal -- a sweep of 0/1/2/3/5 was
    # jagged and non-monotonic on both halves -- but 2 is positive in both, and
    # it fixes the behaviour that was actually observed.
    cooldown_days: int = 2

    # The model states its own conviction. Below this, the proposal is recorded
    # and discarded. Set at the midpoint so that "genuinely unsure" costs
    # nothing: a skipped entry is free, because the next pass is 15 minutes away.
    min_confidence: float = 0.5

    # --- Contract selection -------------------------------------------------
    # Target slightly in the money. A contract near 0.65 delta moves about 65
    # cents per dollar of underlying, loses a small share of its premium to time
    # decay, and quotes tightly. Far out of the money is a lottery ticket -- rare
    # enormous wins -- which is the wrong instrument for a strategy that needs to
    # be right often enough to matter over four trading days.
    delta_min: float = 0.55
    delta_max: float = 0.75

    dte_min: int = 30
    dte_max: int = 45

    # Reject a contract quoting wider than this. Note the free market data tier
    # provides Alpaca's *indicative* options feed, so this is measured against an
    # estimated NBBO rather than full OPRA; the threshold is set loose enough
    # that feed noise alone should not trip it.
    max_spread_pct: float = 0.05

    # --- Exits --------------------------------------------------------------
    # Options carry no broker-side trailing stop -- Alpaca supports trailing
    # stops for stocks only. Every stop here is therefore a SOFTWARE stop,
    # evaluated on each pass, which is why the schedule runs every 15 minutes
    # rather than twice a day.
    #
    # A 25% stop at 2:1 puts the target 50% up and needs roughly a 33% win rate
    # to break even. The target is derived from the stop rather than configured
    # separately, so the two cannot drift apart and silently move that number.
    stop_loss_pct: float = 0.25
    reward_to_risk: float = 2.0

    # Close this many days before expiry regardless of P&L. Time decay
    # accelerates into the final week and the position stops behaving like the
    # directional bet it was opened as.
    close_before_expiry: int = 7

    # --- Sizing -------------------------------------------------------------
    # Share of equity one trade may commit. The measured tradeoff against 2%:
    # more trades on an expensive watchlist, but profit factor fell from 1.11 to
    # 1.02 and drawdown rose from ~17k to ~44k over 2.4 years. Kept at 4% as a
    # deliberate choice about activity on a short measurement window, not
    # because it tested better.
    risk_per_trade: float = 0.04
    max_contracts: int = 50

    # --- Execution ----------------------------------------------------------
    # 0 = bid, 1 = ask. Entries stay patient: a missed entry costs nothing since
    # the signal is re-evaluated in 15 minutes, so paying the full spread to
    # guarantee a fill would be a certain cost to avoid a harmless outcome.
    entry_aggression: float = 0.6

    # Exits hit the bid outright and escalate to a market order if that times
    # out. A missed exit is NOT harmless -- the position keeps moving while the
    # order is retried. Measured: at 0.8, four stops meant for -25% filled
    # between -27% and -33%, which turns a 2:1 rule into roughly 1.6:1 and lifts
    # the break-even win rate from 33% to about 38%.
    exit_aggression: float = 1.0

    @property
    def take_profit_pct(self) -> float:
        """Derived, never configured. See the note on stop_loss_pct."""
        return self.stop_loss_pct * self.reward_to_risk

    @property
    def break_even_win_rate(self) -> float:
        """The win rate this reward-to-risk ratio needs just to tread water.

        Worth surfacing wherever the levels are set. It is the single number
        that says whether the strategy is asking something plausible of itself.
        """
        return 1.0 / (1.0 + self.reward_to_risk)

    def with_overrides(self, **changes) -> Settings:
        """A copy with some fields replaced, for tests and command-line flags."""
        return replace(self, **changes)


def load_settings(path: str | Path = "config.yaml") -> Settings:
    """Read settings from YAML, falling back to the defaults above.

    Only keys that exist on Settings are accepted; an unknown key raises rather
    than being silently ignored, because a typo in a risk limit that quietly
    keeps the default is exactly the kind of failure that is invisible until it
    costs money.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    known = {f for f in Settings.__dataclass_fields__}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown setting(s) in {path}: {', '.join(sorted(unknown))}")

    if "symbols" in raw:
        raw["symbols"] = tuple(s.strip().upper() for s in raw["symbols"])

    return Settings(**raw)
