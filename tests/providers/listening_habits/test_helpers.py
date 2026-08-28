"""Tests for the Listening Habits provider helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import pytest

from music_assistant.providers.listening_habits.helpers import (
    DurableQueue,
    QualityCache,
    device_type_from_model,
    guess_device_type_and_room,
)

_LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize(
    ("display_name", "expected"),
    [
        # Brand match swallows the trailing model, so it does not leak into the room.
        ("Record Room WiiM Ultra", ("WiiM", "Record Room")),
        # Sonos models routinely carry no literal "Sonos" in the name.
        ("Record Room Era 300", ("Sonos", "Record Room")),
        ("Living Room HomePod", ("HomePod", "Living Room")),
        # Most-specific pattern wins over the bare google|nest fallback.
        ("Kitchen Nest Mini", ("Google Nest Mini", "Kitchen")),
        # Carried devices get a type but never a room.
        ("Solace's MacBook Pro", ("Mac", None)),
        # No model word at all -- caller decides whether to treat it as a room.
        ("Kitchen", (None, None)),
        (None, (None, None)),
    ],
)
def test_guess_device_type_and_room(
    display_name: str | None, expected: tuple[str | None, str | None]
) -> None:
    """Player names resolve to the expected (device_type, room)."""
    assert guess_device_type_and_room(display_name) == expected


def test_quality_cache_is_bounded_and_keyed_by_uri() -> None:
    """The cache keeps the most recent entries and never grows without bound."""
    cache = QualityCache(max_entries=2)
    cache.remember("uri-a", "sd-a")  # type: ignore[arg-type]
    cache.remember("uri-b", "sd-b")  # type: ignore[arg-type]
    cache.remember("uri-c", "sd-c")  # type: ignore[arg-type]

    assert cache.recall("uri-a") is None  # evicted
    assert cache.recall("uri-b") == "sd-b"
    assert cache.recall("uri-c") == "sd-c"
    assert cache.recall(None) is None


async def test_durable_queue_keeps_failures_and_drains_on_recovery(tmp_path: Path) -> None:
    """A failed push is retained on disk until it is actually delivered."""
    path = str(tmp_path / "queue.jsonl")
    queue = DurableQueue(path, _LOGGER)

    await queue.append({"client_ref": "ma:1"})
    await queue.append({"client_ref": "ma:2"})
    assert os.path.exists(path)  # noqa: PTH110

    async def always_fails(_: dict[str, object]) -> bool:
        return False

    assert await queue.drain(always_fails) == 0
    assert len(queue._read_sync()) == 2

    delivered: list[str] = []

    async def succeeds(payload: dict[str, object]) -> bool:
        delivered.append(str(payload["client_ref"]))
        return True

    assert await queue.drain(succeeds) == 2
    assert delivered == ["ma:1", "ma:2"]
    assert not os.path.exists(path)  # noqa: PTH110 -- emptied, so removed


async def test_durable_queue_survives_a_torn_write(tmp_path: Path) -> None:
    """An interrupted final line costs that one play, not the whole backlog."""
    path = str(tmp_path / "queue.jsonl")
    queue = DurableQueue(path, _LOGGER)
    torn = json.dumps({"client_ref": "ma:1"}) + "\n" + '{"client_ref": "ma:2"'
    await asyncio.to_thread(Path(path).write_text, torn, encoding="utf-8")

    assert [entry["client_ref"] for entry in queue._read_sync()] == ["ma:1"]


@pytest.mark.parametrize(
    ("manufacturer", "model", "expected"),
    [
        # The common case this exists for: player named "Kitchen", hardware
        # only knowable from what the provider reported.
        ("Sonos", "Era 300", ("Sonos", True)),
        ("Google", "Nest Mini", ("Google Nest Mini", True)),
        ("WiiM", "WiiM Ultra", ("WiiM", True)),
        # Carried hardware is typed but not placed.
        ("Apple", "MacBook Pro", ("Mac", False)),
        # DeviceInfo defaults to placeholder *strings*, not None, so these
        # must not be treated as real hardware names.
        ("Unknown Manufacturer", "Unknown model", (None, True)),
        ("unknown manufacturer", "UNKNOWN MODEL", (None, True)),
        (None, None, (None, True)),
        ("", "", (None, True)),
        # Reported but unrecognised hardware yields no guess rather than a bad one.
        ("Acme Audio", "Widget 9", (None, True)),
        # Manufacturer alone is enough when the model is a placeholder.
        ("Sonos", "Unknown model", ("Sonos", True)),
    ],
)
def test_device_type_from_model(
    manufacturer: str | None, model: str | None, expected: tuple[str | None, bool]
) -> None:
    """Hardware reported by the provider is preferred over parsing the name."""
    assert device_type_from_model(manufacturer, model) == expected


async def test_durable_queue_depth_counts_without_parsing(tmp_path: Path) -> None:
    """
    Depth is a count, so a torn line still contributes to it.

    _read_sync drops an unparseable line because it cannot be delivered.
    depth() must not: the status indicator is there to say "something is
    stuck", and a corrupt row is exactly that. Reporting 1 while the file
    holds 2 lines would hide the problem the indicator exists to show.
    """
    path = str(tmp_path / "queue.jsonl")
    queue = DurableQueue(path, _LOGGER)

    assert await queue.depth() == 0  # no file yet

    torn = json.dumps({"client_ref": "ma:1"}) + "\n" + '{"client_ref": "ma:2"'
    await asyncio.to_thread(Path(path).write_text, torn, encoding="utf-8")

    assert await queue.depth() == 2
    assert len(queue._read_sync()) == 1


async def test_durable_queue_depth_tracks_appends_and_drains(tmp_path: Path) -> None:
    """Depth follows the backlog up and back down to nothing."""
    path = str(tmp_path / "queue.jsonl")
    queue = DurableQueue(path, _LOGGER)

    await queue.append({"client_ref": "ma:1"})
    await queue.append({"client_ref": "ma:2"})
    assert await queue.depth() == 2

    async def succeeds(_: dict[str, object]) -> bool:
        return True

    await queue.drain(succeeds)
    assert await queue.depth() == 0
