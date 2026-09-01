#!/usr/bin/env bash
#
# One scheduled pass. This is what cron runs.
#
# A wrapper exists rather than cron calling Python directly because cron's
# environment is almost nothing like a login shell, and every difference is a
# way for a scheduled job to fail in a manner that never reproduces by hand.
# Four of those differences are handled here, in order of how often they bite:
#
#   1. PATH. Cron gives you /usr/bin:/bin and nothing else. Both `uvx` (which
#      launches the MCP server) and `alpaca` (which places orders) live in
#      ~/.local/bin, which is not on that list. This is the single most common
#      reason a job that works interactively does nothing on a schedule.
#   2. Working directory. Cron starts in $HOME, so config.yaml, .env and the
#      journal would all resolve to the wrong place.
#   3. Overlap. If a pass ever runs longer than the interval, cron starts a
#      second one on top of it, and two passes reading "four of five slots
#      used" will both open a position. The lock makes a late run skip.
#   4. Output. Cron mails stdout to a local mailbox nobody reads. Everything
#      goes to a log file instead.
#
# Arguments are passed through to `run.py trade`, so the crontab line is where
# --live is decided rather than this script.

#
# Usage:
#     ./deploy/run_pass.sh                  the default account
#     ./deploy/run_pass.sh --profile beta   a second account, same code
#
# A profile separates credentials, state and the lock, and nothing else. One
# copy of the software runs both accounts, because two copies means fixing every
# bug twice and discovering later that they diverged.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# The profile is consumed here and passed to Python through the environment, so
# every path decision is made in one place rather than by each caller.
PROFILE=""
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --profile) PROFILE="${2:-}"; shift 2 ;;
        --profile=*) PROFILE="${1#*=}"; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
export ATA_PROFILE="$PROFILE"

# ~/.local/bin first: it holds uvx and alpaca. The rest is a sane minimum,
# because cron's own PATH cannot be relied on for anything.
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

# Per-profile state. The lock especially: two accounts sharing one would make
# the second account's pass skip whenever the first was still running, silently,
# and looking exactly like a quiet market.
if [ -n "$PROFILE" ]; then
    STATE_DIR="state/$PROFILE"
else
    STATE_DIR="state"
fi
mkdir -p "$STATE_DIR"
LOCK_FILE="$STATE_DIR/pass.lock"
LOG_FILE="$STATE_DIR/pass.log"

# The lock is taken here rather than in the crontab line so that it also
# protects a manual run. Two passes colliding is the same hazard whether the
# second one came from cron or from a terminal.
#
# File descriptor 9 is held open for the life of this script, so the lock is
# released when the process ends -- including if it is killed, which a lock
# implemented in Python would not survive.
exec 9>"$LOCK_FILE"
if ! flock --nonblock 9; then
    echo "$(date -u +%FT%TZ)  skipped: the previous ${PROFILE:-default} pass is still running" >> "$LOG_FILE"
    exit 0
fi

STATUS=0
{
    echo ""
    echo "=== $(date -u +%FT%TZ) ${PROFILE:-default} ==="
    ./.venv/bin/python run.py trade ${ARGS+"${ARGS[@]}"}
} >> "$LOG_FILE" 2>&1 || STATUS=$?

# A pass that dies must say so, out loud, in the place you are actually looking.
#
# The agent notifies Discord from inside Python, which works for every failure
# it can catch and not at all for the ones that kill the process. On 31 Aug 2026
# the MCP connection collapsed under load and eight consecutive passes died
# before reaching any notification code -- so the stops never ran, and the whole
# thing looked exactly like a quiet afternoon with nothing to trade.
#
# Silence and success are not allowed to look the same. This is the only place
# that can tell them apart, because it is outside the process that crashed.
if [ "$STATUS" -ne 0 ]; then
    echo "$(date -u +%FT%TZ)  ${PROFILE:-default} pass FAILED with exit $STATUS" >> "$LOG_FILE"

    ENV_FILE=".env${PROFILE:+.$PROFILE}"
    WEBHOOK="$(grep -E '^DISCORD_WEBHOOK_URL=' "$ENV_FILE" 2>/dev/null | cut -d= -f2- | tr -d '"'"'"'"' | tr -d '\r')"
    if [ -n "$WEBHOOK" ]; then
        TAIL="$(tail -n 12 "$LOG_FILE" | sed 's/"/\\"/g' | tr '\n' '~' | sed 's/~/\\n/g')"
        curl -sS -m 10 -X POST "$WEBHOOK" \
            -H 'Content-Type: application/json' \
            -H 'User-Agent: AlpacaTradingAssistant (+scheduler, 1.0)' \
            -d "{\"embeds\":[{\"author\":{\"name\":\"[ATA ${PROFILE:-default}]  Pass FAILED\"},\"title\":\"exit ${STATUS} -- stops did not run\",\"description\":\"\`\`\`\n${TAIL}\n\`\`\`\",\"color\":15158332}]}" \
            >/dev/null 2>&1 || true
    fi
fi

exit "$STATUS"
