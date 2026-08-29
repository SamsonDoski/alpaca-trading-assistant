#!/usr/bin/env bash
#
# Set the agent up on a fresh Linux box, and verify each step rather than
# assuming it worked. Safe to re-run: everything here is idempotent.
#
#     ./deploy/install.sh            install, schedule in dry run
#     ./deploy/install.sh --live     install, schedule with real orders
#
# Deliberately noisy. A silent installer that half-succeeds leaves you
# debugging a schedule at nine on Monday morning.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

LIVE_FLAG=""
[[ "${1:-}" == "--live" ]] && LIVE_FLAG=" --live"

say() { printf '\n>>> %s\n' "$1"; }
fail() { printf '    FAILED: %s\n' "$1"; exit 1; }

export PATH="$HOME/.local/bin:$PATH"

say "1. Python virtual environment"
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv || fail "could not create .venv"
fi
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt || fail "dependencies would not install"
echo "    ok: $(./.venv/bin/python --version)"

say "2. uv (launches the Alpaca MCP server)"
if ! command -v uvx >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh || fail "uv would not install"
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uvx >/dev/null 2>&1 || fail "uvx is still not on PATH"
echo "    ok: $(command -v uvx)"

say "3. Alpaca CLI (places every order)"
if ! command -v alpaca >/dev/null 2>&1; then
    VERSION="0.0.14"
    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64) ARCH=amd64 ;;
        aarch64|arm64) ARCH=arm64 ;;
        *) fail "no published binary for $ARCH" ;;
    esac
    TMP="$(mktemp -d)"
    curl -sL "https://github.com/alpacahq/cli/releases/download/v${VERSION}/cli_${VERSION}_linux_${ARCH}.tar.gz" \
        -o "$TMP/cli.tgz" || fail "could not download the CLI"
    tar xzf "$TMP/cli.tgz" -C "$TMP" || fail "could not unpack the CLI"
    mkdir -p "$HOME/.local/bin"
    install -m 0755 "$TMP/alpaca" "$HOME/.local/bin/alpaca" || fail "could not install the CLI"
    rm -rf "$TMP"
fi
command -v alpaca >/dev/null 2>&1 || fail "alpaca is still not on PATH"
echo "    ok: alpaca $(alpaca version)"

say "4. Credentials"
[[ -f .env ]] || fail ".env is missing. It is never committed -- copy it across by hand."
for key in ALPACA_API_KEY ALPACA_SECRET_KEY ANTHROPIC_API_KEY; do
    grep -q "^${key}=." .env || fail "$key is missing or empty in .env"
done
echo "    ok: all required keys present"
grep -q "^DISCORD_WEBHOOK_URL=." .env \
    && echo "    ok: Discord webhook configured" \
    || echo "    note: no Discord webhook -- the agent will run silently"

say "5. Broker reachable, and it is a paper account"
set -a; . ./.env; set +a
ACCOUNT="$(alpaca account get --quiet --jq .account_number 2>/dev/null | tr -d '"')"
[[ -n "$ACCOUNT" ]] || fail "the CLI could not reach Alpaca -- check the keys in .env"
[[ "$ACCOUNT" == PA* ]] || fail "account $ACCOUNT is not a paper account. Refusing to schedule."
echo "    ok: paper account $ACCOUNT"

say "6. Tests"
./.venv/bin/python -m pytest -q 2>&1 | tail -3 || fail "the test suite does not pass"

say "7. Scheduling every 15 minutes, weekdays, 13:00-20:59 UTC"
echo "    server clock: $(date)"
echo "    UTC:          $(date -u)"

LINE="*/15 13-20 * * 1-5 ${PROJECT_DIR}/deploy/run_pass.sh${LIVE_FLAG}"
chmod +x "${PROJECT_DIR}/deploy/run_pass.sh"

# CRON_TZ pins the schedule to UTC regardless of what the server's clock is set
# to. Without it the hours below mean local time, so the same crontab traded the
# right window on a UTC box and the wrong one everywhere else -- and nothing
# would report the mistake, because a job that runs at the wrong hour looks
# exactly like a job that runs.
#
# Any previous entry for this project is replaced rather than stacked. Two lines
# for the same agent means two passes racing for the same lock, which works, but
# only by accident.
(
    crontab -l 2>/dev/null \
        | grep -v "${PROJECT_DIR}/deploy/run_pass.sh" \
        | grep -v "^CRON_TZ="
    echo "CRON_TZ=UTC"
    echo "$LINE"
) | crontab -
crontab -l | grep -F "run_pass.sh" || fail "the crontab entry did not take"
crontab -l | grep -q "^CRON_TZ=UTC" || echo "    warning: CRON_TZ did not stick; check that the server clock is UTC"

say "Done."
if [[ -n "$LIVE_FLAG" ]]; then
    echo "    Scheduled in LIVE mode. Real orders will be placed on ${ACCOUNT}."
else
    echo "    Scheduled in DRY RUN. Re-run with --live to place real orders."
fi
echo "    Watch it with:  tail -f ${PROJECT_DIR}/state/pass.log"
echo "    Stop it with:   crontab -e   (delete the run_pass.sh line)"
