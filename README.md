# Alpaca Trading Assistant

An autonomous options trading agent built for the Alpaca AI Trading Agents
Hackathon, 28 August - 4 September 2026.

The agent runs unattended every 15 minutes while the US market is open. On each
pass it closes what needs closing, then looks for one new position at a time,
and writes down everything it considered — including what it refused and why.

---

**Hackathon submission**

| | |
| --- | --- |
| Alpaca paper account number | `PA3B2PDNZ732` |
| Alpaca account id | `48ebdce5-2b6b-4a96-b13b-3f3c0e34e7c6` |
| One-page write-up | **[docs/WRITEUP.md](docs/WRITEUP.md)** — thesis, AI logic, every risk gate, the Alpaca integration, live results, and disclosure of pre-existing work |
| Alpaca technologies used | Alpaca MCP Server (all market and account reads), Alpaca CLI (all order writes), Alpaca Trading API paper environment, Alpaca Market Data — option chains with Greeks and implied volatility, quotes, bars, corporate news, and the market clock |
| Tests | 312, no network and no API key required — `pytest -q` |

---

## The idea in one sentence

**The model proposes; the gates dispose.**

A language model reading option chains, Greeks and news is good at noticing
things and bad at being consistently disciplined. So it is given the first job
and denied the second. Claude never holds a tool that can place an order. It
produces a *view on a symbol* — a direction, a stated conviction, and a written
rationale — and that view then has to survive a chain of plain deterministic
Python rules that have no opinions and cannot be argued with.

Two invariants make that claim structural rather than decorative, and both are
asserted in code rather than described in a comment:

1. A gate may **reject or shrink**. It may never enlarge an order and may never
   invent one. The worst thing the risk chain can do to the account is nothing.
2. **Exits are never gated.** A rule that can block an exit is a rule that can
   trap a loss.

## How it talks to Alpaca

| Direction | Surface | Why |
| --- | --- | --- |
| Reads — chains, quotes, Greeks, positions, account | **Alpaca MCP server** | Gives the model real Greeks and implied volatility to reason over, through structured tools |
| Writes — every order, without exception | **Alpaca CLI** (`alpaca order submit`) | A separate, deterministic execution path the model cannot reach. Each order carries a `--client-order-id` we generate, so every fill maps back to the proposal and the gate verdicts that allowed it |

Splitting the two is the point. The reasoning surface and the execution surface
are different processes with different privileges, so "the model cannot place an
order" is a property of the architecture rather than a promise.

Long options have no broker-side trailing stop — Alpaca supports those for
stocks only — so every stop in this system is a **software stop** re-evaluated on
each pass. That is why the schedule is 15 minutes rather than twice a day.

## Running it

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

```bash
python run.py trade --only-when-open
```

```bash
python -m pytest -q
```

The risk gates are tested with no network, no broker and no model — that is the
reason they take a context object instead of fetching what they need. The part of
this agent that must not be wrong is also the part that can be exercised
exhaustively in under a second.

## Deploying it

```bash
./deploy/install.sh
```

Installs the virtual environment, `uv`, and the Alpaca CLI; checks that `.env`
holds every required key; confirms with the broker that the account is a paper
account and refuses to schedule if it is not; runs the tests; and installs the
crontab entry. Safe to re-run. Add `--live` to schedule real orders instead of a
dry run.

The schedule is every 15 minutes, weekdays, 13:00–20:59 UTC — deliberately wider
than market hours. Encoding 09:30–16:00 Eastern precisely means the schedule is
silently wrong twice a year when the clocks change, so cron fires generously and
the market-open gate decides whether there is anything to do. A pass outside
market hours reads the clock, finds it closed, and exits in about a second
having spent nothing.

**Fifteen minutes is a risk parameter, not a convenience.** Alpaca supports
trailing stops for stocks, not options, so there is no protective order resting
at the broker. Every stop in this system only exists while a pass is running,
which makes the interval between passes the distance a position can travel
unwatched.

Watch it with `tail -f state/pass.log`; stop it by deleting the line from
`crontab -e`.

## Disclosure of pre-existing work

Per the hackathon FAQ, which permits reuse of a participant's own prior work
provided it is disclosed:

**Written during the hackathon window (28 Aug – 4 Sep 2026):** the entire agent
in `src/agent/` — the domain model, the risk gate chain, the MCP read layer, the
Claude proposer, the CLI execution path, and the scheduling and locking that make
it autonomous.

**Carried over from my own earlier personal project,** an options backtesting and
paper-trading system built in August 2026 before the hackathon was announced:
the contract-selection and position-sizing arithmetic, the SQLite journal, the
Discord notifier, and the measured parameter values recorded in
`src/agent/settings.py`. Those values come from a 2.4-year backtest with
split-half out-of-sample testing on that earlier project; the results are cited
in `docs/` as supporting evidence for why the settings are what they are, not as
hackathon deliverables. That project is mine, unpublished, and MIT-licensed.

**Not written by me:** nothing else. No third-party strategy code is included.

## Layout

```
src/agent/
  domain.py     what a contract, account, position and proposal ARE.
                Depends on nothing, so everything can depend on it.
  settings.py   every tunable number, with the measurement behind it.
  gates.py      the risk chain. Pure functions over a context snapshot.
tests/          the risk system, exercised without a network.
docs/           architecture notes and the backtest evidence.
```
