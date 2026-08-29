"""Discord notifications.

**Fail-open, always.** A webhook outage, a rate limit, a malformed payload: each
one logs a warning and returns. The loop calling this places real orders, and a
chat service being down must never stop a position being managed or block the
next one. Notification is a courtesy, not a dependency -- which is the opposite
of the policy in the market reader, where a failed read stops the pass.

Stdlib `urllib` rather than `requests`. This process holds broker credentials,
and posting to a webhook is not worth widening the dependency surface for.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

_GREEN = 0x2ECC71
_RED = 0xE74C3C
_BLUE = 0x3498DB
_AMBER = 0xF39C12
_GREY = 0x95A5A6

# Discord's front end answers 403 to urllib's default user agent, and the
# rejection is indistinguishable from a bad token because both return an empty
# body. Sending a real one is the difference between every alert working and
# every alert silently failing.
_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "AlpacaTradingAssistant (+https://github.com/SamsonDoski, 1.0)",
}

# The account also runs an older bot against a shared webhook. The tag makes it
# obvious at a glance which agent is speaking.
_TAG = "[ATA]"


class Notifier:
    """Posts embeds to a Discord webhook. Never raises."""

    def __init__(self, webhook_url: str | None, *, timeout: float = 10.0) -> None:
        self._url = webhook_url
        self._timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    def opened(self, symbol: str, detail: str, *, reasoning: str = "") -> None:
        description = detail
        if reasoning:
            description += f"\n\n_{reasoning[:600]}_"
        self._post(_embed(f"{_TAG}  Opened", symbol, description, _GREEN))

    def closed(self, symbol: str, detail: str, *, won: bool | None = None) -> None:
        colour = _GREY if won is None else _GREEN if won else _RED
        self._post(_embed(f"{_TAG}  Closed", symbol, detail, colour))

    def alert(self, message: str) -> None:
        """Something went wrong, or something operationally important happened."""
        self._post(_embed(f"{_TAG}  Alert", "", message, _AMBER))

    def pass_summary(self, *, equity: float, available: float, open_count: int,
                     opened: int, closed: int, considered: int,
                     realized: float, unrealized: float,
                     book: list[tuple[str, float]] | None = None,
                     refusals: list[tuple[str, str]] | None = None,
                     dry_run: bool = False) -> None:
        """What this pass did, including what it refused to do.

        Realised and unrealised are shown separately on purpose. Conflating a
        paper gain on an open position with money actually banked is how a day
        looks better than it was.

        The refusals section is the part worth having. A summary listing only
        trades makes an agent that did nothing look identical to an agent that
        was not running.
        """
        lines = [
            f"### ${equity:,.0f} equity  ·  ${available:,.0f} free",
            f"**{considered}** considered · **{opened}** opened · "
            f"**{closed}** closed · **{open_count}** open",
            f"realised **{_money(realized, signed=True)}** · "
            f"unrealised **{_money(unrealized, signed=True)}**",
        ]

        if book:
            rows = "\n".join(
                f"{'+' if pnl >= 0 else '-'} `{sym}`  {_money(pnl, signed=True)}"
                for sym, pnl in book[:10])
            lines.append("\n**Open book:**\n" + rows)

        title = f"{datetime.now(UTC):%A %d %B, %H:%M} UTC"
        if dry_run:
            title += "  (dry run)"

        self._post(_embed(f"{_TAG}  Pass complete", title, "\n".join(lines), _BLUE))

        # Declines go in their own message, or several. They used to be appended
        # here, capped at eight entries of 110 characters each -- which on a pass
        # where all nine symbols decline meant one symbol vanished entirely and
        # every rationale was cut mid-sentence. That is precisely the content
        # worth reading: a pass that trades nothing is only legible through its
        # reasons. Discord caps an embed description at 4096 characters, so the
        # list is split across as many messages as it needs rather than trimmed
        # to fit one.
        if refusals:
            self._post_declines(refusals)

    def _post_declines(self, refusals: list[tuple[str, str]]) -> None:
        blocks = [f"**{symbol}** — {reason.strip()}" for symbol, reason in refusals]

        for index, chunk in enumerate(_chunked(blocks, _EMBED_LIMIT)):
            heading = ("Considered and declined" if index == 0
                       else f"Considered and declined (continued {index + 1})")
            self._post(_embed(f"{_TAG}  {heading}", "", "\n\n".join(chunk), _GREY))

    def _post(self, payload: dict) -> None:
        if not self._url:
            return
        try:
            request = urllib.request.Request(
                self._url, data=json.dumps(payload).encode("utf-8"),
                headers=_HEADERS, method="POST")
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                if response.status >= 300:
                    logger.warning("discord returned %s", response.status)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.warning("discord notification failed: %s", exc)

    def verify(self) -> tuple[bool, str]:
        """Send a probe and report the real result, for `run.py notify-test`."""
        if not self._url:
            return False, "no webhook configured"
        try:
            request = urllib.request.Request(
                self._url,
                data=json.dumps(_embed(f"{_TAG}  Webhook check", "",
                                       "Notifications are working.", _GREY)).encode("utf-8"),
                headers=_HEADERS, method="POST")
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return response.status < 300, f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            return False, f"HTTP {exc.code}"
        except (urllib.error.URLError, OSError) as exc:
            return False, str(exc)


# Discord's hard cap on an embed description is 4096 characters. Leaving a
# margin means a long rationale can never push a block over the edge and cost
# the whole message.
_EMBED_LIMIT = 3800


def _chunked(blocks: list[str], limit: int) -> list[list[str]]:
    """Group blocks into batches that each fit inside one embed.

    Splits between blocks, never inside one, so a rationale is either shown in
    full or moved whole to the next message. A sentence cut in half is worse
    than a sentence on the following card.

    A single block longer than the limit is the one case where truncation is
    unavoidable, and it is marked so the reader knows something was cut rather
    than wondering why the model stopped mid-thought.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    length = 0

    for block in blocks:
        if len(block) > limit:
            block = block[: limit - 3] + "..."
        # +2 for the blank line joining blocks.
        if current and length + len(block) + 2 > limit:
            batches.append(current)
            current, length = [], 0
        current.append(block)
        length += len(block) + 2

    if current:
        batches.append(current)
    return batches


def _embed(author: str, title: str, description: str, colour: int) -> dict:
    embed = {"description": description[:4000], "color": colour,
             "author": {"name": author}}
    if title:
        embed["title"] = title[:256]
    return {"embeds": [embed]}


def _money(amount: float, *, signed: bool = False) -> str:
    sign = "+" if signed and amount >= 0 else "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.0f}"
