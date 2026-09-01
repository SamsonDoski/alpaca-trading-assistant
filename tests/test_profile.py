"""Tests for running two accounts from one copy of the code.

What must be separated, and why each one matters:

  * credentials -- the point of the exercise;
  * state -- a shared journal would put a cooldown on the second account
    because the first stopped out, and `holdings` rows keyed by contract symbol
    would collide the moment both bought the same option;
  * the lock -- a shared one would make the second account's pass skip whenever
    the first was still running, silently, looking exactly like a quiet market.

What must NOT be separated: the code. Two copies means fixing every bug twice.
"""

from __future__ import annotations

import pytest

from agent import profile
from agent.journal import Journal


@pytest.fixture(autouse=True)
def clean_profile(monkeypatch):
    monkeypatch.delenv(profile.ENV_VAR, raising=False)


def test_the_default_profile_uses_the_original_paths():
    """Backwards compatible: an existing single-account install keeps working
    with no environment variable set at all."""
    assert profile.env_file(".").name == ".env"
    assert profile.state_dir(".").name == "state"
    assert profile.label() == "default"


def test_a_named_profile_gets_its_own_env_file(monkeypatch):
    monkeypatch.setenv(profile.ENV_VAR, "beta")
    assert profile.env_file(".").name == ".env.beta"


def test_a_named_profile_gets_its_own_state_directory(monkeypatch, tmp_path):
    monkeypatch.setenv(profile.ENV_VAR, "beta")
    assert profile.state_dir(tmp_path).name == "beta"
    assert profile.state_dir(tmp_path).parent.name == "state"


def test_the_state_directory_is_created_on_demand(monkeypatch, tmp_path):
    monkeypatch.setenv(profile.ENV_VAR, "gamma")
    assert profile.state_dir(tmp_path).is_dir()


def test_whitespace_around_the_profile_name_is_ignored(monkeypatch):
    monkeypatch.setenv(profile.ENV_VAR, "  beta  ")
    assert profile.env_file(".").name == ".env.beta"


def test_an_empty_profile_is_the_default(monkeypatch):
    monkeypatch.setenv(profile.ENV_VAR, "")
    assert profile.env_file(".").name == ".env"


# --- Config falls back rather than being duplicated ------------------------

def test_a_profile_config_is_used_when_it_exists(monkeypatch, tmp_path):
    monkeypatch.setenv(profile.ENV_VAR, "beta")
    (tmp_path / "config.beta.yaml").write_text("symbols: [SPY]")
    (tmp_path / "config.yaml").write_text("symbols: [QQQ]")
    assert profile.config_file(tmp_path).name == "config.beta.yaml"


def test_a_profile_without_its_own_config_shares_the_common_one(monkeypatch, tmp_path):
    """Two accounts usually want identical rules. Forcing a duplicate config
    would reintroduce exactly the drift profiles exist to prevent."""
    monkeypatch.setenv(profile.ENV_VAR, "beta")
    (tmp_path / "config.yaml").write_text("symbols: [QQQ]")
    assert profile.config_file(tmp_path).name == "config.yaml"


# --- The journals are genuinely separate -----------------------------------

def test_two_profiles_get_different_journals(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setenv(profile.ENV_VAR, "alpha")
    alpha = Journal()
    monkeypatch.setenv(profile.ENV_VAR, "beta")
    beta = Journal()

    assert alpha.path != beta.path


def test_one_profiles_stop_loss_does_not_cool_off_the_other(monkeypatch, tmp_path):
    """The concrete hazard: a shared journal would block the second account
    from a name it never traded."""
    monkeypatch.chdir(tmp_path)

    monkeypatch.setenv(profile.ENV_VAR, "alpha")
    alpha = Journal()
    alpha.record("closed", "NVDA", "NVDA261016C00200000", "stop loss -- down 30%",
                 pnl=-500)

    monkeypatch.setenv(profile.ENV_VAR, "beta")
    beta = Journal()

    assert alpha.cooling_off(within_days=2) == {"NVDA": 2}
    assert beta.cooling_off(within_days=2) == {}


def test_holdings_do_not_collide_when_both_buy_the_same_contract(monkeypatch, tmp_path):
    """`holdings` is keyed by contract symbol. Shared, the second account's
    entry levels would silently overwrite the first's."""
    monkeypatch.chdir(tmp_path)
    symbol = "SPY261016C00700000"

    monkeypatch.setenv(profile.ENV_VAR, "alpha")
    alpha = Journal()
    alpha.open_holding(occ_symbol=symbol, underlying="SPY", direction="up",
                       entry_spot=700.0, entry_premium=10.0,
                       stop_spot=680.0, target_spot=740.0)

    monkeypatch.setenv(profile.ENV_VAR, "beta")
    beta = Journal()
    beta.open_holding(occ_symbol=symbol, underlying="SPY", direction="up",
                      entry_spot=705.0, entry_premium=12.0,
                      stop_spot=685.0, target_spot=745.0)

    assert alpha.holding(symbol).entry_spot == 700.0
    assert beta.holding(symbol).entry_spot == 705.0


def test_an_explicit_path_still_overrides_the_profile(monkeypatch, tmp_path):
    """Tests and one-off tools pass a path directly; that must keep working."""
    monkeypatch.setenv(profile.ENV_VAR, "beta")
    journal = Journal(tmp_path / "explicit.db")
    assert journal.path.name == "explicit.db"
