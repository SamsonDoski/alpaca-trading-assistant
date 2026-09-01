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

from dataclasses import dataclass, field, replace
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

    # --- Concentration ------------------------------------------------------
    # Which symbols move together. Declared rather than computed: a rolling
    # correlation matrix would be more precise and also unstable, expensive, and
    # impossible to explain inside a one-line refusal.
    #
    # Empty by default so the cap is opt-in from config.yaml, where the universe
    # that needs grouping is actually defined.
    correlation_groups: dict[str, str] = field(default_factory=dict)

    # Positions allowed in any one group. Every other gate reasons about a
    # single trade; without this, eight slots could hold eight versions of the
    # same bet -- one macro position with eight chances to be wrong together.
    max_per_group: int = 3

    # Positions allowed on the same side. Sector limits catch "all technology";
    # this catches the subtler case of eight well-spread sectors that are all
    # long calls, which is still a single bet on the market going up.
    max_same_direction: int = 5

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

    # Reject a contract quoting wider than this.
    #
    # This gate changed meaning once the feed was measured, and the change is
    # worth being explicit about rather than hiding behind a number.
    #
    # At 5% it was a COST-OF-CROSSING rule, calibrated against real OPRA data:
    # a position opened across a 5% spread starts 5% down, which is more than
    # the edge on an average trade. That is the right rule and the right number
    # -- against the right data.
    #
    # We do not have that data. The free Basic plan supplies Alpaca's INDICATIVE
    # options feed, and measuring 44 symbols during market hours on 31 Aug 2026
    # showed what that costs: MSFT quoted 12.5% wide and AAPL 12.7%, in the
    # 30-45 day window at 0.65 delta. Those are two of the most liquid option
    # markets in existence and genuinely trade near 1-2%. The number describes
    # the feed, not the market.
    #
    # So at 0.15 this is no longer a cost rule. It is a SANITY rule: it rejects
    # quotes that are broken rather than merely inflated -- the 34% and 37% ones
    # where the feed has nothing real at all. The assumption it now rests on is
    # that reported spreads on liquid names overstate the true spread by roughly
    # an order of magnitude. That assumption is reasonable and it is not
    # verified; if a reported spread is ever genuine, this gate will let us pay
    # it. Gating on open interest instead would be feed-independent and is the
    # better long-term fix.
    max_spread_pct: float = 0.15

    # --- What the premium costs ---------------------------------------------
    # Reject a contract whose implied volatility exceeds the underlying's own
    # realized volatility by more than this multiple.
    #
    # This is the gate the strategy was missing. Implied volatility is the PRICE
    # of an option, and until now the system displayed it and acted on it not at
    # all: SPY at 13% IV and SMCI at 70% went through identical machinery. For a
    # premium buyer that is trading blind on the one number that decides whether
    # the trade is cheap.
    #
    # 1.4 means: pay up to forty percent over what the stock has actually been
    # doing, and refuse beyond that. Above roughly 1.5 the variance risk premium
    # is large enough that a directionally correct trade can still lose, because
    # implied volatility collapses toward realized once the event passes.
    #
    # Note what this is NOT: a true IV rank, comparing today's implied against
    # its own history. That needs an IV time series per contract, and contracts
    # expire and roll so the series is not continuous. IV against realized asks
    # the same question with data already in hand and a clearer economic meaning.
    max_iv_to_realized: float = 1.4

    # Sessions of history used to measure the underlying's realized volatility.
    # 20 is about a month of trading -- long enough to be a measurement rather
    # than a reading of the last few days, short enough to reflect the regime
    # the option is actually being priced in.
    realized_vol_lookback: int = 20

    # Refuse a contract that decays faster than this each day, as a fraction of
    # the premium. 1.5% a day is roughly a fifth of the position over a two-week
    # thesis -- gone before direction has had a chance to matter.
    #
    # Long premium is a race between the move and the clock. Nothing in this
    # system could see the clock until now.
    max_daily_decay: float = 0.015

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

    # --- Where the stop actually sits ---------------------------------------
    # The stop is keyed to the UNDERLYING, at this multiple of its average true
    # range. Distance measured in the stock's own units, because a flat 3% is a
    # meaningful move on a quiet name and pure noise on a volatile one -- and a
    # stop inside the noise fires on days where nothing happened.
    #
    # 2.0 puts the stop roughly two typical days away: far enough that ordinary
    # movement does not reach it, close enough that a real reversal does.
    stop_atr_multiple: float = 2.0

    # Used only when there is not enough history to measure a range.
    fallback_stop_pct: float = 0.03

    # The premium stop, demoted to a backstop and widened accordingly.
    #
    # It was the primary rule and should not have been. A 25% fall in a
    # 0.65-delta option is under a 2% move in the underlying, so it fired on
    # noise and on volatility crushes while the thesis was intact -- and fired
    # hardest on exactly the high-volatility names where premium swings most,
    # which is the opposite of what a risk rule should do.
    #
    # At 50% its job is different: catch an option that has been gutted by
    # collapsing implied volatility even though the stock did nothing. That is a
    # broken position, not a losing one, and it needs closing for a different
    # reason.
    premium_backstop_pct: float = 0.50

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

    # --- Concurrency --------------------------------------------------------
    # How many symbols are worked on at once.
    #
    # This exists because of an outage, not a theory. The first version fanned
    # out every symbol simultaneously, which was fine at nine and killed the
    # agent at thirty: nine symbols is 36 concurrent tool calls through a single
    # stdio connection to the MCP server, and thirty is 120. From 14:30 ET on
    # 31 Aug 2026 every pass died with "MCPError: Connection closed" -- taking
    # the exit checks with it, so a position sat at -42% through a -25% stop for
    # the last two hours of trading.
    #
    # The fix is a ceiling rather than a smaller watchlist. Six symbols at a
    # time is roughly 24 concurrent reads, comfortably inside what the server
    # handles, and a pass still finishes in a fraction of its fifteen minutes
    # because the waiting still overlaps -- just in batches instead of all at
    # once.
    max_concurrent_symbols: int = 6

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
