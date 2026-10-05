"""Tests for one-minute MA song logging, including live radio."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import MediaType, PlaybackState
from music_assistant_models.streamdetails import StreamMetadata

import music_assistant.providers.listening_habits as module
from music_assistant.controllers.streams.constants import STREAMDETAILS_INBAND_TITLE_KEY
from music_assistant.providers.listening_habits import (
    ACTIVE_RADIO_PLAY_KEY,
    LAST_MATCHED_KEY,
    STATION_SLUG_KEY,
    ListeningHabitsProvider,
)
from music_assistant.providers.listening_habits.helpers import QualityCache
from music_assistant.providers.listening_habits.song_logging import SongListen


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Build a provider with fake playback, monotonic time and ingest."""
    clock = SimpleNamespace(now=0.0, serial=0)

    def next_ns() -> int:
        clock.serial += 1
        return int(clock.serial)

    monkeypatch.setattr(
        module,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now,
            time=lambda: 1_800_000_000 + clock.now,
            time_ns=next_ns,
        ),
    )
    media = SimpleNamespace(
        uri="library://track/1", name="Song", artist="Artist", media_type=MediaType.TRACK
    )
    item = SimpleNamespace(media_item=media, queue_item_id="item-1", streamdetails=None)
    queue = SimpleNamespace(current_item=item, state=PlaybackState.PLAYING, elapsed_time=0)
    provider: Any = object.__new__(ListeningHabitsProvider)
    provider.unloading = False
    provider.logger = MagicMock()
    provider._current_plays = {}
    provider._play_refs = {}
    provider._last_counted = {}
    provider._radio_log_inflight = set()
    provider._quality = QualityCache()
    provider._endpoint = ""
    provider._token = ""
    provider._scrobbler_config = SimpleNamespace(
        mass_userids=(), mass_playerids=(), suffix_version=False
    )
    provider._weather_snapshot = AsyncMock(return_value={})
    provider._build_payload = MagicMock(return_value={"artist": "Artist", "title": "Song"})
    provider._push = AsyncMock(return_value=True)
    provider._backlog = SimpleNamespace(drain=AsyncMock(), append=AsyncMock())
    provider.mass = SimpleNamespace(
        player_queues=SimpleNamespace(
            get=lambda queue_id: queue if queue_id == "player" else None,
            queue_data_or_none=lambda _: SimpleNamespace(userid="user"),
        ),
        create_task=asyncio.create_task,
    )
    result = SimpleNamespace(provider=provider, queue=queue, clock=clock)

    def tick(now: float, position: float, state: PlaybackState = PlaybackState.PLAYING) -> None:
        clock.now = now
        queue.elapsed_time = position
        queue.state = state
        provider._on_queue_time_updated(SimpleNamespace(object_id="player", data=position))

    result.tick = tick
    provider._on_queue_updated(SimpleNamespace(object_id="player"))
    return result


def report(
    rig: SimpleNamespace, *, playing: bool = True, completed: bool = False
) -> SimpleNamespace:
    """Return one normal MA track-progress report."""
    return SimpleNamespace(
        media_type=MediaType.TRACK,
        userid="user",
        player_id="player",
        uri=rig.queue.current_item.media_item.uri,
        name="Song",
        artist="Artist",
        fully_played=completed,
        is_playing=playing,
        seconds_played=rig.queue.elapsed_time,
    )


async def settle() -> None:
    """Drain the short fake ingest tasks without sleeping in real time."""
    for _ in range(4):
        await asyncio.sleep(0)


def radio(rig: SimpleNamespace, *, feed: bool = False) -> SimpleNamespace:
    """Switch the same fake queue to one valid radio song."""
    play = {
        "play_id": "airing-1",
        "artist": "Artist",
        "title": "Song",
        "album": "Album",
        "duration_s": 240,
        "played_at": 1_799_999_900,
    }
    data = {}
    if feed:
        data = {
            STATION_SLUG_KEY: "the-current",
            LAST_MATCHED_KEY: "Artist - Song",
            STREAMDETAILS_INBAND_TITLE_KEY: "Artist - Song",
            ACTIVE_RADIO_PLAY_KEY: play,
        }
    details = SimpleNamespace(
        queue_id="player",
        media_type=MediaType.RADIO,
        data=data,
        stream_metadata=StreamMetadata(artist="Artist", title="Song", album="Album", duration=240),
        stream_metadata_update_callback=None,
    )
    rig.queue.current_item = SimpleNamespace(
        queue_item_id="radio-item",
        streamdetails=details,
        media_item=SimpleNamespace(
            uri="library://radio/1", name="Station", media_type=MediaType.RADIO
        ),
    )
    rig.queue.elapsed_time = 0
    rig.provider._on_queue_updated(SimpleNamespace(object_id="player"))
    return details


def test_claim_requires_sixty_observed_seconds_and_is_single_use() -> None:
    """Completion flags are not necessary and cannot claim a second log."""
    listen = SongListen(0, 0, True, 1_800_000_000)
    listen.observe(59, True, 59)
    assert not listen.claim()
    listen.observe(60, True, 60)
    assert listen.claim()
    listen.observe(240, False, 240)
    assert not listen.claim()


def test_pause_and_seek_do_not_create_a_minute() -> None:
    """A moved playhead and time spent paused are not heard music."""
    listen = SongListen(0, 0, True, 1_800_000_000)
    listen.observe(500, True, 1)
    assert listen.heard == 1
    listen.observe(501, False, 2)
    listen.observe(501, False, 200)
    listen.observe(501, True, 201)
    assert listen.heard == 2
    listen.observe(531, True, 231)
    assert listen.heard == 32
    assert not listen.claim()


def test_no_initial_credit_for_attaching_mid_song() -> None:
    """Loading the provider midway through a song does not invent earlier listening."""
    listen = SongListen(180, 0, True, 1_800_000_000)
    assert not listen.due
    listen.observe(239, True, 59)
    assert not listen.due
    listen.observe(240, True, 60)
    assert listen.due


async def test_track_logs_at_minute_without_waiting_for_completion(rig: SimpleNamespace) -> None:
    """One-minute promotion retains its ref for later completion reports."""
    rig.tick(30, 30)
    await rig.provider._on_media_item_played(SimpleNamespace(data=report(rig)))
    assert rig.provider._push.await_count == 0
    ref = rig.provider._current_plays["player"]["client_ref"]
    rig.tick(59, 59)
    await settle()
    assert rig.provider._push.await_count == 0
    rig.tick(60, 60)
    await settle()
    assert rig.provider._push.await_count == 1
    assert rig.provider._build_payload.call_args.kwargs["client_ref"] == ref
    assert rig.provider._build_payload.call_args.kwargs["played_at"].timestamp() == 1_800_000_000
    rig.tick(230, 230)
    await rig.provider._on_media_item_played(SimpleNamespace(data=report(rig, completed=True)))
    rig.tick(240, 240, PlaybackState.IDLE)
    await rig.provider._on_media_item_played(
        SimpleNamespace(data=report(rig, playing=False, completed=True))
    )
    assert rig.provider._push.await_count == 1


async def test_short_completed_song_does_not_bypass_minute(rig: SimpleNamespace) -> None:
    """The explicitly requested minute is not replaced with a completion fallback."""
    rig.tick(25, 25)
    await rig.provider._on_media_item_played(SimpleNamespace(data=report(rig, completed=True)))
    rig.tick(30, 30, PlaybackState.IDLE)
    await rig.provider._on_media_item_played(
        SimpleNamespace(data=report(rig, playing=False, completed=True))
    )
    assert rig.provider._push.await_count == 0


async def test_failed_song_uses_durable_backlog_once(rig: SimpleNamespace) -> None:
    """A failed first push queues its payload without creating another completion row."""
    rig.provider._push.return_value = False
    rig.tick(30, 30)
    await rig.provider._on_media_item_played(SimpleNamespace(data=report(rig)))
    rig.tick(60, 60)
    await settle()
    rig.tick(240, 240)
    await rig.provider._on_media_item_played(SimpleNamespace(data=report(rig, completed=True)))
    assert rig.provider._push.await_count == 1
    assert rig.provider._backlog.append.await_count == 1


async def test_repeat_one_gets_a_fresh_ref_and_minute(rig: SimpleNamespace) -> None:
    """A real repeat of the same URI can log once again, not immediately."""
    rig.tick(30, 30)
    await rig.provider._on_media_item_played(SimpleNamespace(data=report(rig)))
    rig.tick(60, 60)
    await settle()
    first = rig.provider._current_plays["player"]["client_ref"]
    rig.tick(61, 0)
    second = rig.provider._current_plays["player"]["client_ref"]
    assert first != second
    rig.tick(91, 30)
    await rig.provider._on_media_item_played(SimpleNamespace(data=report(rig)))
    assert rig.provider._push.await_count == 1
    rig.tick(121, 60)
    await settle()
    assert rig.provider._push.await_count == 2


@pytest.mark.parametrize("feed", [False, True])
async def test_radio_song_waits_a_minute_and_logs_once(rig: SimpleNamespace, feed: bool) -> None:
    """Both provider-owned metadata and exact station-feed matches obey sixty seconds."""
    radio(rig, feed=feed)
    rig.tick(59, 59)
    await settle()
    assert rig.provider._push.await_count == 0
    rig.tick(60, 60)
    await settle()
    assert rig.provider._push.await_count == 1
    rig.tick(120, 120)
    await settle()
    assert rig.provider._push.await_count == 1
    kwargs = rig.provider._build_payload.call_args.kwargs
    assert kwargs["title"] == "Song"
    if feed:
        assert kwargs["client_ref"] == "thecurrent:airing-1"


async def test_new_radio_song_cannot_inherit_prior_song_time(rig: SimpleNamespace) -> None:
    """A title change starts a new minute even though the station URI is unchanged."""
    details = radio(rig)
    rig.tick(59, 59)
    details.stream_metadata = StreamMetadata(artist="Other", title="Next")
    rig.tick(60, 60)
    rig.tick(119, 119)
    await settle()
    assert rig.provider._push.await_count == 0
    rig.tick(120, 120)
    await settle()
    assert rig.provider._push.await_count == 1
    assert rig.provider._build_payload.call_args.kwargs["title"] == "Next"


async def test_radio_talk_break_or_unresolved_icy_clears_timer(rig: SimpleNamespace) -> None:
    """A previous feed match must not survive a new unresolved in-band title."""
    details = radio(rig, feed=True)
    rig.tick(59, 59)
    details.data[STREAMDETAILS_INBAND_TITLE_KEY] = "Talk break"
    rig.tick(60, 60)
    rig.tick(180, 180)
    await settle()
    assert rig.provider._push.await_count == 0


async def test_radio_pause_does_not_count_toward_minute(rig: SimpleNamespace) -> None:
    """A long radio pause needs actual playback to finish the minute."""
    radio(rig)
    rig.tick(30, 30, PlaybackState.PAUSED)
    rig.tick(300, 30, PlaybackState.PAUSED)
    rig.tick(301, 30)
    rig.tick(330, 59)
    await settle()
    assert rig.provider._push.await_count == 0
    rig.tick(331, 60)
    await settle()
    assert rig.provider._push.await_count == 1


@pytest.mark.parametrize("feed", [False, True])
async def test_filtered_radio_does_not_loop_or_log(rig: SimpleNamespace, feed: bool) -> None:
    """Excluded radio players cannot enter recursive qualification tasks."""
    radio(rig, feed=feed)
    rig.provider._scrobbler_config.mass_playerids = ("another-player",)
    rig.tick(60, 60)
    await settle()
    assert rig.provider._push.await_count == 0


@pytest.mark.parametrize("filter_name", ["mass_playerids", "mass_userids"])
async def test_track_filter_changed_before_minute_is_respected(
    rig: SimpleNamespace, filter_name: str
) -> None:
    """A cached report cannot bypass a filter disabled before qualification."""
    rig.tick(30, 30)
    await rig.provider._on_media_item_played(SimpleNamespace(data=report(rig)))
    setattr(rig.provider._scrobbler_config, filter_name, ("someone-else",))
    rig.tick(60, 60)
    await settle()
    assert rig.provider._push.await_count == 0


async def test_overlapping_track_events_claim_before_network_work(rig: SimpleNamespace) -> None:
    """Concurrent time and progress events still create one ingest attempt."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def held_weather() -> dict[str, Any]:
        started.set()
        await release.wait()
        return {}

    rig.provider._weather_snapshot.side_effect = held_weather
    rig.tick(30, 30)
    await rig.provider._on_media_item_played(SimpleNamespace(data=report(rig)))
    rig.tick(60, 60)
    await started.wait()
    rig.tick(61, 61)
    await rig.provider._on_media_item_played(SimpleNamespace(data=report(rig)))
    release.set()
    await settle()
    assert rig.provider._push.await_count == 1


@pytest.mark.parametrize("feed", [False, True])
async def test_overlapping_radio_events_claim_before_network_work(
    rig: SimpleNamespace, feed: bool
) -> None:
    """Repeated radio ticks during slow enrichment cannot create duplicate rows."""
    radio(rig, feed=feed)
    started = asyncio.Event()
    release = asyncio.Event()

    async def held_weather() -> dict[str, Any]:
        started.set()
        await release.wait()
        return {}

    rig.provider._weather_snapshot.side_effect = held_weather
    rig.tick(60, 60)
    await started.wait()
    rig.tick(61, 61)
    await settle()
    release.set()
    await settle()
    assert rig.provider._push.await_count == 1
