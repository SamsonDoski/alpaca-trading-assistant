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

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# ~/.local/bin first: it holds uvx and alpaca. The rest is a sane minimum,
# because cron's own PATH cannot be relied on for anything.
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

mkdir -p state
LOCK_FILE="state/pass.lock"
LOG_FILE="state/pass.log"

# The lock is taken here rather than in the crontab line so that it also
# protects a manual run. Two passes colliding is the same hazard whether the
# second one came from cron or from a terminal.
#
# File descriptor 9 is held open for the life of this script, so the lock is
# released when the process ends -- including if it is killed, which a lock
# implemented in Python would not survive.
exec 9>"$LOCK_FILE"
if ! flock --nonblock 9; then
    echo "$(date -u +%FT%TZ)  skipped: the previous pass is still running" >> "$LOG_FILE"
    exit 0
fi

{
    echo ""
    echo "=== $(date -u +%FT%TZ) ==="
    ./.venv/bin/python run.py trade "$@"
} >> "$LOG_FILE" 2>&1
