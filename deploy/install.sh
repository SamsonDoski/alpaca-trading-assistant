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

# Usage:
#     ./deploy/install.sh                        default account, dry run
#     ./deploy/install.sh --live                 default account, real orders
#     ./deploy/install.sh --profile beta --live  a second account, same code
#
# A profile gets its own credentials, state and crontab line. The software is
# shared: two copies would mean fixing every bug twice.
LIVE_FLAG=""
PROFILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --live) LIVE_FLAG=" --live"; shift ;;
        --profile) PROFILE="${2:-}"; shift 2 ;;
        --profile=*) PROFILE="${1#*=}"; shift ;;
        *) echo "unknown argument: $1"; exit 1 ;;
    esac
done

export ATA_PROFILE="$PROFILE"
ENV_FILE=".env${PROFILE:+.$PROFILE}"
PROFILE_ARG="${PROFILE:+ --profile $PROFILE}"
LABEL="${PROFILE:-default}"

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
set -a; . "./$ENV_FILE"; set +a
ACCOUNT="$(alpaca account get --quiet --jq .account_number 2>/dev/null | tr -d '"')"
[[ -n "$ACCOUNT" ]] || fail "the CLI could not reach Alpaca -- check the keys in .env"
[[ "$ACCOUNT" == PA* ]] || fail "account $ACCOUNT is not a paper account. Refusing to schedule."
echo "    ok: paper account $ACCOUNT"

say "6. Tests"  # shared code, so this covers every profile
./.venv/bin/python -m pytest -q 2>&1 | tail -3 || fail "the test suite does not pass"

say "7. Scheduling every 15 minutes on weekdays"

# The schedule is written in the SERVER'S LOCAL TIME, and the hours are derived
# from its UTC offset rather than hardcoded.
#
# The first version of this used CRON_TZ=UTC with fixed UTC hours, which is the
# textbook answer and did not work: this server's cron ignores CRON_TZ, so the
# hours were silently read as local time and no pass fired for the first three
# and a half hours of the trading day. Nothing reported it, because a job that
# runs at the wrong hour looks exactly like a job that runs.
#
# So: compute. The US market trades 09:30-16:00 Eastern. Convert that to the
# server's own clock and give it an hour of margin either side, because the
# market-open gate -- not the schedule -- is the authority on whether a pass
# does anything.
SERVER_OFFSET=$(date +%z)                       # e.g. -0400
EASTERN_OFFSET=$(TZ=America/New_York date +%z)  # e.g. -0400
SHIFT=$(( (${SERVER_OFFSET%??} - ${EASTERN_OFFSET%??}) ))
START_HOUR=$(( (9 + SHIFT + 24) % 24 ))
END_HOUR=$(( (16 + SHIFT + 24) % 24 ))

echo "    server clock:  $(date)"
echo "    US Eastern:    $(TZ=America/New_York date)"
echo "    trading window: ${START_HOUR}:00-${END_HOUR}:59 server local"

if (( START_HOUR > END_HOUR )); then
    fail "the trading window wraps midnight in this server's timezone; set the
    server to US Eastern or UTC, or write the crontab by hand."
fi

# A second profile is offset by five minutes rather than firing at the same
# instant. Both accounts hitting the MCP server and the model simultaneously
# doubles the peak load for no benefit; staggering costs nothing.
MINUTES="*/15"
[[ -n "$PROFILE" ]] && MINUTES="5,20,35,50"

LINE="${MINUTES} ${START_HOUR}-${END_HOUR} * * 1-5 ${PROJECT_DIR}/deploy/run_pass.sh${PROFILE_ARG}${LIVE_FLAG}"
chmod +x "${PROJECT_DIR}/deploy/run_pass.sh"

# Replace only THIS profile's line. Two lines for one profile would mean two
# passes racing for the same lock; deleting another profile's line means an
# account silently stops trading.
#
# That second failure happened: installing the beta profile removed the default
# account's schedule, because the filter matched on the script path alone. It
# was live and scored at the time, and nothing reported it -- a missing crontab
# line looks exactly like a market with nothing to do.
#
# The match must therefore include the profile argument AND be anchored so that
# an empty one cannot match a line carrying `--profile something`. `--live` is
# excluded from the pattern deliberately: a profile switching between dry run
# and live must still replace its own line rather than stack a second one.
BEFORE="$(crontab -l 2>/dev/null || true)"
KEEP="$(printf '%s\n' "$BEFORE" \
    | grep -v -E "run_pass\.sh${PROFILE_ARG}( --live)?[[:space:]]*$" \
    | grep -v "^CRON_TZ=" || true)"

printf '%s\n%s\n' "$KEEP" "$LINE" | grep -v '^$' | crontab -

crontab -l | grep -qF "$LINE" || fail "the crontab entry did not take"
echo "    installed: $LINE"

# Every other profile's line must have survived. Counting them is cheap and it
# is the only check that would have caught the default account being unscheduled.
BEFORE_COUNT="$(printf '%s\n' "$BEFORE" | grep -c "run_pass\.sh" || true)"
AFTER_COUNT="$(crontab -l | grep -c "run_pass\.sh" || true)"
if [ "$AFTER_COUNT" -lt "$BEFORE_COUNT" ]; then
    fail "installing '${LABEL}' removed another profile's schedule
    (${BEFORE_COUNT} run_pass lines before, ${AFTER_COUNT} after). Restore with:
      crontab -e"
fi
echo "    ${AFTER_COUNT} profile(s) scheduled:"
crontab -l | grep "run_pass\.sh" | sed 's/^/      /'

say "Done  [profile: ${LABEL}]"
if [[ -n "$LIVE_FLAG" ]]; then
    echo "    Scheduled in LIVE mode. Real orders will be placed on ${ACCOUNT}."
else
    echo "    Scheduled in DRY RUN. Re-run with --live to place real orders."
fi
echo "    Watch it with:  tail -f ${PROJECT_DIR}/state/${PROFILE:+$PROFILE/}pass.log"
echo "    Stop it with:   crontab -e   (delete the '${LABEL}' run_pass.sh line)"
