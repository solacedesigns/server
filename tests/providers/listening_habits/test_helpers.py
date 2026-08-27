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
