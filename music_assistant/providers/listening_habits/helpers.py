"""Helpers for the Listening Habits provider: name parsing and the durable queue."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import logging

    from music_assistant_models.streamdetails import StreamDetails

# Best-effort device_type from the player's display name -- these are the
# user's room/player names, not something Music Assistant labels, so this is
# pattern-matching rather than a lookup. An unmatched name yields `None`
# rather than a guess, matching the rest of this project: the site then shows
# the device name without a category, which is correct-but-vague instead of
# confidently wrong.
#
# Ordered most-specific first (first match wins) so "Nest Mini" is not caught
# by the bare "google|nest" fallback below it. `fixed` marks a player that
# lives in one place -- for those, whatever text is NOT the model name is
# taken to be the room. Mobile devices (a Mac) are never fixed: they are
# carried around, so they get a device_type but no room.
DEVICE_TYPE_PATTERNS: list[tuple[re.Pattern[str], str, bool]] = [
    # Brand only, matched to end of name so a trailing model ("... WiiM
    # Ultra") is swallowed rather than leaking into the room text.
    (re.compile(r"wiim.*", re.IGNORECASE), "WiiM", True),
    (re.compile(r"homepod.*", re.IGNORECASE), "HomePod", True),
    # Sonos models often appear with no literal "Sonos" in the name at all
    # ("Record Room Era 300"), so match the model words directly.
    (
        re.compile(r"(?:sonos|era\s*\d+|\bbeam\b|\barc\b|\bmove\b|\broam\b).*", re.IGNORECASE),
        "Sonos",
        True,
    ),
    (re.compile(r"(?:google\s*)?nest\s*mini", re.IGNORECASE), "Google Nest Mini", True),
    (re.compile(r"(?:google\s*)?nest\s*hub", re.IGNORECASE), "Google Nest Hub", True),
    (re.compile(r"(?:google\s*)?nest\s*audio", re.IGNORECASE), "Google Nest Audio", True),
    (re.compile(r"google|nest", re.IGNORECASE), "Google Speaker", True),
    (re.compile(r"chromecast", re.IGNORECASE), "Chromecast", True),
    (re.compile(r"frame\s*tv|vizio|\btv\b", re.IGNORECASE), "TV", True),
    (re.compile(r"\bmac\b|macbook", re.IGNORECASE), "Mac", False),
]


def guess_device_type_and_room(display_name: str | None) -> tuple[str | None, str | None]:
    """
    Split a player's display name into (device_type, room).

    "Record Room WiiM Ultra" -> ("WiiM", "Record Room")
    "Solace's MacBook Pro"   -> ("Mac", None)  -- mobile, no fixed room
    "Kitchen"                -> (None, None)   -- caller may fall back to the name
    """
    name = display_name or ""
    for pattern, label, fixed in DEVICE_TYPE_PATTERNS:
        match = pattern.search(name)
        if not match:
            continue
        if not fixed:
            return label, None
        room = (name[: match.start()] + name[match.end() :]).strip(" -_")
        room = re.sub(r"\s+", " ", room).strip()
        return label, room or None
    return None, None


# `DeviceInfo` uses placeholder strings rather than None when a provider does
# not report these, so they have to be treated as absent explicitly.
_UNKNOWN_DEVICE_FIELDS = frozenset({"unknown model", "unknown manufacturer", ""})


def device_type_from_model(manufacturer: str | None, model: str | None) -> tuple[str | None, bool]:
    """
    Derive (device_type, is_fixed) from a player's reported hardware.

    Preferred over parsing the display name: the provider that created the
    player knows what the hardware is, whereas a name like "Kitchen" carries
    no model word at all. Returns (None, True) when nothing is reported.
    """
    parts = [
        value
        for value in (manufacturer, model)
        if value and value.strip().lower() not in _UNKNOWN_DEVICE_FIELDS
    ]
    if not parts:
        return None, True
    text = " ".join(parts)
    for pattern, label, fixed in DEVICE_TYPE_PATTERNS:
        if pattern.search(text):
            return label, fixed
    return None, True


class QualityCache:
    """
    Remembers each track's StreamDetails while it is still the current item.

    Necessary because MEDIA_ITEM_PLAYED usually reports the track that just
    *ended*, fired after the next one already became current -- so reading
    `queue.current_item.streamdetails` inside the handler returns the wrong
    track's audio format. The playback tracker holds the right object when it
    builds the report but does not put it on the event, so the only way to
    line quality up with the play it belongs to is to snapshot it earlier,
    keyed by the uri the report will carry.

    Bounded, because a long-running server would otherwise accumulate one
    entry per track ever played.
    """

    def __init__(self, max_entries: int = 256) -> None:
        """Initialize."""
        self._entries: OrderedDict[str, StreamDetails] = OrderedDict()
        self._max_entries = max_entries

    def remember(self, uri: str | None, streamdetails: StreamDetails | None) -> None:
        """Record the streamdetails currently in effect for `uri`."""
        if not uri or streamdetails is None:
            return
        self._entries[uri] = streamdetails
        self._entries.move_to_end(uri)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def recall(self, uri: str | None) -> StreamDetails | None:
        """Return the streamdetails last seen for `uri`, if any."""
        if not uri:
            return None
        return self._entries.get(uri)


class DurableQueue:
    """
    An append-only JSONL backlog of payloads that failed to reach the server.

    A failed push is never dropped. This mirrors the guarantee the AppDaemon
    app provided (and that MA's own ScrobblerHelper does not -- it logs the
    exception and moves on): the file lives under the server's storage path,
    so a backlog survives the add-on restarting, whether for an update or a
    config change.
    """

    def __init__(self, path: str, logger: logging.Logger) -> None:
        """Initialize."""
        self.path = path
        self.logger = logger
        self._lock = asyncio.Lock()

    async def append(self, payload: dict[str, Any]) -> None:
        """Add a payload to the backlog."""
        async with self._lock:
            await asyncio.to_thread(self._append_sync, payload)

    def _append_sync(self, payload: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    async def drain(self, push: Any) -> int:
        """
        Retry every queued payload, keeping whatever still fails.

        The file is rewritten from what remains rather than truncated up-front,
        so a crash mid-drain loses nothing: the worst case is a payload
        delivered twice, and `client_ref` makes the server idempotent against
        exactly that.
        """
        async with self._lock:
            pending = await asyncio.to_thread(self._read_sync)
            if not pending:
                return 0
            still_failing: list[dict[str, Any]] = []
            delivered = 0
            for payload in pending:
                if await push(payload):
                    delivered += 1
                else:
                    still_failing.append(payload)
            await asyncio.to_thread(self._rewrite_sync, still_failing)
            if delivered:
                self.logger.info(
                    "drained %s queued play(s), %s still pending", delivered, len(still_failing)
                )
            return delivered

    async def depth(self) -> int:
        """
        How many plays are waiting to be delivered.

        Counts lines rather than parsing them: this is called to render a
        status indicator, so a torn final line should not make the whole
        answer fail. It deliberately does NOT take the lock -- a status read
        must never wait behind an in-flight drain, and being one play stale
        matters less than blocking the UI.
        """
        return await asyncio.to_thread(self._depth_sync)

    def _depth_sync(self) -> int:
        if not os.path.exists(self.path):  # noqa: PTH110
            return 0
        with open(self.path, encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def _read_sync(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.path):  # noqa: PTH110
            return []
        entries: list[dict[str, Any]] = []
        with open(self.path, encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    # A torn final line from an interrupted write. Skipping it
                    # loses one play; refusing to parse the file would lose all
                    # of them, so this is the cheaper failure.
                    self.logger.warning("discarding unparseable queue line")
        return entries

    def _rewrite_sync(self, entries: list[dict[str, Any]]) -> None:
        if not entries:
            if os.path.exists(self.path):  # noqa: PTH110
                os.remove(self.path)  # noqa: PTH107
            return
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(entry) + "\n" for entry in entries)
        os.replace(tmp, self.path)  # noqa: PTH105
