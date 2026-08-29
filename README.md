# Alpaca Trading Assistant

An autonomous options trading agent built for the Alpaca AI Trading Agents
Hackathon, 28 August - 4 September 2026.

The agent runs unattended every 15 minutes while the US market is open. On each
pass it closes what needs closing, then looks for one new position at a time,
and writes down everything it considered — including what it refused and why.

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
