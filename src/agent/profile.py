"""Running more than one account from one copy of the code.

A second account needs its own credentials, its own journal and its own lock. It
does not need its own copy of the software -- that would mean fixing every bug
twice and discovering months later that the two copies had quietly diverged.

So one variable decides everything that must differ:

    ATA_PROFILE unset       .env          state/          config.yaml
    ATA_PROFILE=beta        .env.beta     state/beta/     config.beta.yaml

Three things are separated and the separation matters for a different reason
each time:

  * **Credentials.** Obvious, and the whole point.
  * **State.** Two accounts sharing a journal would produce a cooldown on one
    because the other stopped out, and a `holdings` row keyed by contract symbol
    would collide the moment both bought the same option.
  * **The lock.** Sharing it would make the second account's pass skip whenever
    the first was still running -- silently, and looking exactly like a quiet
    market.

Config falls back rather than being separated: `config.beta.yaml` is used if it
exists, and `config.yaml` otherwise. Two accounts usually want the same rules
and occasionally want different ones, so the common case should need no extra
file.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "ATA_PROFILE"


def name() -> str:
    """The active profile, or an empty string for the default one."""
    return (os.getenv(ENV_VAR) or "").strip()


def label() -> str:
    """A human-readable name, for logs and Discord."""
    return name() or "default"


def env_file(root: Path | str = ".") -> Path:
    """Which .env holds this profile's credentials."""
    profile = name()
    return Path(root) / (f".env.{profile}" if profile else ".env")


def state_dir(root: Path | str = ".") -> Path:
    """Where this profile's journal, lock and log live.

    Created on demand, because everything that writes here expects the directory
    to exist and a missing state directory is never a condition worth reporting.
    """
    profile = name()
    path = Path(root) / "state" / profile if profile else Path(root) / "state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def journal_path(root: Path | str = ".") -> Path:
    return state_dir(root) / "journal.db"


def config_file(root: Path | str = ".") -> Path:
    """This profile's config, falling back to the shared one.

    Falls back rather than requiring a file per profile: two accounts usually
    want identical rules, and forcing a duplicate config would reintroduce
    exactly the drift this module exists to prevent.
    """
    profile = name()
    if profile:
        specific = Path(root) / f"config.{profile}.yaml"
        if specific.exists():
            return specific
    return Path(root) / "config.yaml"
