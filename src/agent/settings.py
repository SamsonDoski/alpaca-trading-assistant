"""Every tunable number in one place, with the reason it has the value it has.

The settings object is immutable and passed down, never read from a global, so a
test can construct one with the values it wants and never touch a config file.

The reasoning lives next to the number, because a bare `stop_loss = 0.25` tells a
reader what the code does but not whether they may change it. Values carried over
from an earlier options research project are marked with what was measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class Settings:
    """The complete rule set the agent runs under."""

    # --- Universe -----------------------------------------------------------
    # Liquid names only. A wider list showed several quoting 5-7% wide, which is
    # more than the edge on an average trade.
    symbols: tuple[str, ...] = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA",
                                "AMZN", "META", "GOOGL", "TSLA")

    # --- Entry screening ----------------------------------------------------
    # Beat 8, 10 and 15 over 2.4 years, winning in both split halves separately.
    # The cap works as a quality filter: raising it reaches into lower-ranked
    # candidates rather than adding more of the same trades.
    max_positions: int = 5

    # Days before re-entering an underlying after a STOP LOSS. Never after a win.
    # Added because a live run re-bought a name minutes after stopping out on the
    # same signal. A sweep of 0/1/2/3/5 was jagged, but 2 is positive in both
    # halves and fixes the behaviour actually observed.
    cooldown_days: int = 2

    # Below this the model's own stated conviction is recorded and discarded. Set
    # at the midpoint so "unsure" costs nothing -- the next pass is 15 minutes on.
    min_confidence: float = 0.5

    # --- Concentration ------------------------------------------------------
    # Which symbols move together. Declared rather than computed: a rolling
    # correlation matrix would be more precise and also unstable, expensive, and
    # impossible to explain inside a one-line refusal. Empty by default so the cap
    # is opt-in from config.yaml, where the universe is actually defined.
    correlation_groups: dict[str, str] = field(default_factory=dict)

    # Every other gate reasons about a single trade. Without this, eight slots
    # could hold eight versions of one bet with eight chances to be wrong at once.
    max_per_group: int = 3

    # Sector limits catch "all technology"; this catches eight well-spread sectors
    # that are all long calls, which is still one bet on the market going up.
    max_same_direction: int = 5

    # --- Contract selection -------------------------------------------------
    # Slightly in the money. Near 0.65 delta an option moves about 65 cents per
    # dollar of underlying, loses little premium to decay, and quotes tightly. Far
    # out of the money is a lottery ticket, which is the wrong instrument for a
    # strategy that has to be right often enough to matter.
    delta_min: float = 0.55
    delta_max: float = 0.75

    dte_min: int = 30
    dte_max: int = 45

    # Reject a contract quoting wider than this. The gate changed meaning once the
    # feed was measured, which is worth stating rather than hiding behind a number.
    #
    # At 5% it was a cost-of-crossing rule against real OPRA data: open across a 5%
    # spread and the position starts 5% down. We do not have that data. The free
    # Basic plan serves Alpaca's INDICATIVE feed, and measuring 44 symbols on
    # 31 Aug 2026 showed MSFT at 12.5% and AAPL at 12.7% -- two of the most liquid
    # option markets in existence, which genuinely trade near 1-2%.
    #
    # So at 0.15 this is a SANITY rule, rejecting quotes that are broken rather
    # than merely inflated. It rests on the unverified assumption that reported
    # spreads on liquid names overstate the truth by about an order of magnitude;
    # if one is ever genuine, this gate will let us pay it. Gating on open interest
    # would be feed-independent and is the better long-term fix.
    max_spread_pct: float = 0.15

    # --- What the premium costs ---------------------------------------------
    # Reject a contract whose implied volatility exceeds the underlying's realized
    # volatility by more than this multiple.
    #
    # Implied volatility is the PRICE of an option, and the system used to display
    # it and act on it not at all: SPY at 13% IV and SMCI at 70% went through
    # identical machinery. Above roughly 1.5 the variance risk premium is large
    # enough that a directionally correct trade can still lose, because implied
    # collapses toward realized once the event passes.
    #
    # Not a true IV rank -- that needs a per-contract IV series, and contracts
    # expire and roll so the series is not continuous. IV against realized asks the
    # same question with data already in hand.
    max_iv_to_realized: float = 1.4

    # Sessions used to measure realized volatility. About a month: long enough to
    # be a measurement, short enough to reflect the current regime.
    realized_vol_lookback: int = 20

    # Daily decay ceiling as a fraction of premium. 1.5% a day is roughly a fifth
    # of the position over a two-week thesis, gone before direction can matter.
    # Long premium is a race between the move and the clock, and nothing in this
    # system could see the clock until this gate existed.
    max_daily_decay: float = 0.015

    # --- Exits --------------------------------------------------------------
    # Options carry no broker-side trailing stop -- Alpaca supports those for
    # stocks only -- so every stop here is a SOFTWARE stop evaluated each pass.
    # That is why the schedule is 15 minutes rather than twice a day.
    #
    # A 25% stop at 2:1 puts the target 50% up and needs about a 33% win rate to
    # break even. The target is derived from the stop, so the two cannot drift
    # apart and silently move that number.
    #
    # This is an ABSOLUTE FLOOR on premium, not an alternative to the underlying
    # stop below -- both run, and the position closes on whichever comes first.
    # Keying stops only to the underlying let positions travel to an average 52%
    # premium loss before anything could close them, against sizing that assumed
    # this number. Measured on the live book, 4 Sep 2026.
    stop_loss_pct: float = 0.25
    reward_to_risk: float = 2.0

    # --- Where the stop actually sits ---------------------------------------
    # The stop is keyed to the UNDERLYING, at this multiple of its average true
    # range. A flat 3% is a real move on a quiet name and noise on a volatile one,
    # and a stop inside the noise fires on days where nothing happened. 2.0 sits
    # roughly two typical days away.
    stop_atr_multiple: float = 2.0

    # Used only when there is not enough history to measure a range.
    fallback_stop_pct: float = 0.03

    # The same idea on the winning side. Stop and target deliberately key to
    # different things: a premium stop fires on noise, so the stop belongs on the
    # underlying, but a large premium gain is not noise -- it is money, and holding
    # out for the underlying risks handing it back. Seen on 2 Sep 2026, a PLTR put
    # at +50.7% with its underlying target five percent away and no rule to take it.
    premium_target_backstop_pct: float = 0.50

    # Close this many days before expiry regardless of P&L. Decay accelerates into
    # the final week and the position stops behaving like the bet it was opened as.
    close_before_expiry: int = 7

    # --- Sizing -------------------------------------------------------------
    # Share of equity one trade may commit. Measured against 2%: more trades on an
    # expensive watchlist, but profit factor fell from 1.11 to 1.02 and drawdown
    # rose from ~17k to ~44k over 2.4 years. Kept at 4% as a deliberate choice
    # about activity, not because it tested better.
    risk_per_trade: float = 0.04
    max_contracts: int = 50

    # --- Concurrency --------------------------------------------------------
    # How many symbols are worked at once. This exists because of an outage, not a
    # theory. Fanning out every symbol was fine at nine and killed the agent at
    # thirty -- nine symbols is 36 concurrent calls through one stdio connection,
    # thirty is 120. From 14:30 ET on 31 Aug 2026 every pass died with "Connection
    # closed", taking the exit checks with it, and a position sat at -42% through a
    # -25% stop for two hours.
    #
    # The fix is a ceiling rather than a smaller watchlist: six at a time is about
    # 24 concurrent reads, and a pass still finishes well inside fifteen minutes
    # because the waiting still overlaps, just in batches.
    max_concurrent_symbols: int = 6

    # --- Execution ----------------------------------------------------------
    # 0 = bid, 1 = ask. Entries stay patient: a missed entry costs nothing when the
    # signal is re-evaluated in 15 minutes, so paying the full spread to guarantee
    # a fill is a certain cost to avoid a harmless outcome.
    entry_aggression: float = 0.6

    # Exits hit the bid and escalate to market on a timeout. A missed exit is NOT
    # harmless -- the position keeps moving while the order is retried. Measured:
    # at 0.8, four stops meant for -25% filled between -27% and -33%, turning a 2:1
    # rule into 1.6:1 and lifting the break-even win rate from 33% to about 38%.
    exit_aggression: float = 1.0

    @property
    def take_profit_pct(self) -> float:
        """Derived, never configured. See the note on stop_loss_pct."""
        return self.stop_loss_pct * self.reward_to_risk

    @property
    def break_even_win_rate(self) -> float:
        """The win rate this reward-to-risk ratio needs just to tread water.

        The single number that says whether the strategy is asking something
        plausible of itself, so it is worth surfacing wherever levels are set.
        """
        return 1.0 / (1.0 + self.reward_to_risk)

    def with_overrides(self, **changes) -> Settings:
        """A copy with some fields replaced, for tests and command-line flags."""
        return replace(self, **changes)


def load_settings(path: str | Path = "config.yaml") -> Settings:
    """Read settings from YAML, falling back to the defaults above.

    An unknown key raises rather than being silently ignored: a typo in a risk
    limit that quietly keeps the default is invisible until it costs money.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    known = {f for f in Settings.__dataclass_fields__}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown setting(s) in {path}: {', '.join(sorted(unknown))}")

    if "symbols" in raw:
        raw["symbols"] = tuple(s.strip().upper() for s in raw["symbols"])

    return Settings(**raw)
