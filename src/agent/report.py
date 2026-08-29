"""Rendering the journal as a single self-contained HTML page.

A user interface is not required for this agent, and it deliberately does not
have a live one -- there is no server, no port, and nothing to keep running. What
there is instead is a report: run a command, get one HTML file, open it in a
browser. No assets, no fonts to fetch, no JavaScript. Copy it anywhere and it
still works.

That constraint is what makes it worth having. A dashboard is a service you have
to keep alive and secure; a file is just a file. It can be screenshotted for a
presentation, attached to an email, or dropped in a web root, and none of those
require the agent to be running.

**What it is for.** The journal already answers "what did it do". The part worth
showing a human is *why it didn't* -- the refusal that stopped each trade, and
which gate produced it. An agent that trades twice in a week looks identical to
a broken one until you can see the ninety decisions it made in between.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from html import escape
from pathlib import Path

from agent.domain import AccountState, OpenPosition
from agent.journal import Journal

_STYLE = """
:root {
  --bg: #fbfaf8; --card: #ffffff; --ink: #1a1a18; --muted: #6b6b63;
  --line: #e5e3dc; --accent: #1d9e75; --warn: #ba7517; --down: #d85a30;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16161a; --card: #1e1e23; --ink: #ececea; --muted: #9a9a92;
    --line: #2e2e35; --accent: #5dcaa5; --warn: #efa027; --down: #f0997b;
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 2.5rem 1.25rem 4rem; background: var(--bg); color: var(--ink);
  font: 15px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width: 880px; margin: 0 auto; }
h1 { font-size: 1.5rem; font-weight: 500; margin: 0 0 .25rem; letter-spacing: -.01em; }
h2 { font-size: 1rem; font-weight: 500; margin: 2.5rem 0 .875rem; }
.sub { color: var(--muted); font-size: .875rem; margin: 0 0 2rem; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .75rem; }
.stat { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.125rem; }
.stat .k { color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; }
.stat .v { font-size: 1.5rem; font-weight: 500; margin-top: .25rem; font-variant-numeric: tabular-nums; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
table { width: 100%; border-collapse: collapse; font-size: .875rem; }
th { text-align: left; font-weight: 500; color: var(--muted); font-size: .75rem;
  text-transform: uppercase; letter-spacing: .06em; padding: .75rem 1rem; border-bottom: 1px solid var(--line); }
td { padding: .75rem 1rem; border-bottom: 1px solid var(--line); vertical-align: top; }
tr:last-child td { border-bottom: none; }
.num { font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
.sym { font-weight: 500; white-space: nowrap; }
code { font: 13px/1.5 ui-monospace, "SF Mono", Menlo, monospace; }
.pill { display: inline-block; padding: .125rem .5rem; border-radius: 999px;
  font-size: .75rem; font-weight: 500; white-space: nowrap; }
.pill.open { background: color-mix(in srgb, var(--accent) 18%, transparent); color: var(--accent); }
.pill.skip { background: color-mix(in srgb, var(--muted) 16%, transparent); color: var(--muted); }
.up { color: var(--accent); } .dn { color: var(--down); }
details { margin-top: .375rem; }
summary { cursor: pointer; color: var(--muted); font-size: .8125rem; }
details p { margin: .5rem 0 0; color: var(--muted); font-size: .8125rem;
  border-left: 2px solid var(--line); padding-left: .75rem; }
.bar { height: 6px; border-radius: 3px; background: var(--muted); opacity: .5; }
.empty { padding: 1.5rem 1rem; color: var(--muted); font-size: .875rem; }
footer { margin-top: 3rem; color: var(--muted); font-size: .75rem; }
"""


def _money(value: float, *, signed: bool = False) -> str:
    sign = "+" if signed and value >= 0 else "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def _reason_family(reason: str) -> str:
    """Group a refusal into the thing that caused it.

    Nine distinct sentences about nine symbols are hard to read as a pattern.
    Collapsed into families, the same data answers a much better question: is
    this agent being stopped by its own risk limits, or by the market not
    offering anything worth taking?
    """
    text = reason.lower()
    if "already holding" in text:
        return "already held"
    if "cooldown" in text:
        return "cooling off"
    if "slots" in text:
        return "no free slot"
    if "kill switch" in text:
        return "kill switch"
    if "market is closed" in text:
        return "market closed"
    if "confidence" in text and "floor" in text:
        return "below confidence floor"
    if "wide" in text:
        return "spread too wide"
    if "delta" in text:
        return "no contract in delta band"
    if "expiry" in text:
        return "expiry window"
    if "budget" in text or "uncommitted" in text or "costs" in text:
        return "position too expensive"
    if "unavailable" in text:
        return "data or model unavailable"
    return "no directional view"


def _gate_counts(journal: Journal) -> Counter:
    """How often each gate produced the refusal, across the whole journal."""
    counts: Counter = Counter()
    with sqlite3.connect(str(journal.path)) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT gate_trace FROM decisions WHERE approved = 0"):
            try:
                trace = json.loads(row["gate_trace"] or "[]")
            except json.JSONDecodeError:
                continue
            for name, decision, _ in trace:
                if decision == "deny":
                    counts[name] += 1
    return counts


def render(journal: Journal, *, account: AccountState | None = None,
           positions: tuple[OpenPosition, ...] = (), dry_run: bool = True) -> str:
    """Build the whole page as one string."""
    decisions = journal.decisions_for_day()
    totals = journal.summary()
    events = journal.recent(limit=25)

    considered = sum(1 for d in decisions if d.confidence is not None)
    opened_today = sum(1 for d in decisions if d.approved)
    unrealized = sum(p.unrealized_pnl for p in positions)

    parts: list[str] = [
        "<div class=wrap>",
        "<h1>Alpaca Trading Assistant</h1>",
        f"<p class=sub>{'Dry run' if dry_run else 'Live'} &middot; generated "
        f"{datetime.now(UTC):%d %B %Y, %H:%M} UTC</p>",
    ]

    # --- headline numbers -------------------------------------------------
    stats = [
        ("Equity", _money(account.equity) if account else "--"),
        ("Open positions", str(len(positions))),
        ("Unrealised", f"<span class='{'up' if unrealized >= 0 else 'dn'}'>"
                       f"{_money(unrealized, signed=True)}</span>"),
        ("Considered today", str(considered)),
        ("Opened today", str(opened_today)),
        ("Realised today", _money(journal.realized_for_day(), signed=True)),
    ]
    parts.append("<div class=stats>")
    for key, value in stats:
        parts.append(f"<div class=stat><div class=k>{escape(key)}</div>"
                     f"<div class=v>{value}</div></div>")
    parts.append("</div>")

    # --- open book --------------------------------------------------------
    parts.append("<h2>Open positions</h2><div class=card>")
    if positions:
        parts.append("<table><tr><th>Contract</th><th>Qty</th><th class=num>Entry</th>"
                     "<th class=num>Now</th><th class=num>P&amp;L</th></tr>")
        for p in positions:
            css = "up" if p.unrealized_pnl >= 0 else "dn"
            parts.append(
                f"<tr><td><code>{escape(p.occ_symbol)}</code></td>"
                f"<td class=num>{p.quantity}</td>"
                f"<td class=num>${p.entry_price:,.2f}</td>"
                f"<td class=num>${p.current_price:,.2f}</td>"
                f"<td class='num {css}'>{_money(p.unrealized_pnl, signed=True)} "
                f"({p.return_pct:+.1%})</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<div class=empty>No open option positions.</div>")
    parts.append("</div>")

    # --- why it did not trade --------------------------------------------
    families = Counter(_reason_family(d.reason) for d in decisions if not d.approved)
    parts.append("<h2>Why it declined</h2><div class=card>")
    if families:
        widest = max(families.values())
        parts.append("<table>")
        for name, count in families.most_common():
            width = int(100 * count / widest)
            parts.append(
                f"<tr><td class=sym>{escape(name)}</td>"
                f"<td style='width:60%'><div class=bar style='width:{width}%'></div></td>"
                f"<td class=num>{count}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<div class=empty>Nothing declined yet today.</div>")
    parts.append("</div>")

    gates = _gate_counts(journal)
    if gates:
        parts.append("<h2>Refusals by gate, all time</h2><div class=card><table>")
        for name, count in gates.most_common():
            parts.append(f"<tr><td class=sym><code>{escape(name)}</code></td>"
                         f"<td class=num>{count}</td></tr>")
        parts.append("</table></div>")

    # --- the decision log -------------------------------------------------
    parts.append("<h2>Decisions today</h2><div class=card>")
    if decisions:
        parts.append("<table><tr><th>Time</th><th>Symbol</th><th></th>"
                     "<th class=num>Conviction</th><th>Outcome</th></tr>")
        for d in decisions:
            pill = "open" if d.approved else "skip"
            label = "opened" if d.approved else "declined"
            conviction = f"{d.confidence:.2f}" if d.confidence is not None else "&mdash;"
            reasoning = ""
            if d.thinking:
                reasoning = (f"<details><summary>reasoning</summary>"
                             f"<p>{escape(d.thinking)}</p></details>")
            parts.append(
                f"<tr><td class=num>{escape(d.at[11:16])}</td>"
                f"<td class=sym>{escape(d.underlying)}</td>"
                f"<td><span class='pill {pill}'>{label}</span></td>"
                f"<td class=num>{conviction}</td>"
                f"<td>{escape(d.reason)}{reasoning}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<div class=empty>No decisions recorded today.</div>")
    parts.append("</div>")

    # --- trades -----------------------------------------------------------
    if events:
        parts.append("<h2>Recent activity</h2><div class=card><table>")
        for e in events:
            pnl = ""
            if e.pnl is not None:
                css = "up" if e.pnl >= 0 else "dn"
                pnl = f"<span class='{css}'>{_money(e.pnl, signed=True)}</span>"
            parts.append(f"<tr><td class=num>{escape(e.at[:16].replace('T', ' '))}</td>"
                         f"<td class=sym>{escape(e.action)}</td>"
                         f"<td><code>{escape(e.symbol)}</code></td>"
                         f"<td>{escape(e.detail[:120])}</td>"
                         f"<td class=num>{pnl}</td></tr>")
        parts.append("</table></div>")

    parts.append(
        f"<footer>Lifetime: {totals['closed']} closed &middot; "
        f"{totals['win_rate']:.0%} won &middot; "
        f"{_money(totals['total_pnl'], signed=True)}. "
        f"Model proposes, gates dispose.</footer></div>")

    body = "\n".join(parts)
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Alpaca Trading Assistant</title><style>{_STYLE}</style></head>"
            f"<body>{body}</body></html>")


def write(journal: Journal, path: Path | str = Path("state/report.html"),
          **kwargs) -> Path:
    """Render and save. Returns where it landed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render(journal, **kwargs), encoding="utf-8")
    return destination
