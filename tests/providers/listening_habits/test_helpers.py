"""Tests for the Listening Habits provider helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import MediaType, ProviderIconVariant

from music_assistant.providers.listening_habits import (
    AMBIENT_SESSION_RESUME_GRACE_S,
    AMBIENT_SESSION_UPDATE_THRESHOLDS_S,
    ListeningHabitsProvider,
    SUPPORTED_MEDIA_TYPES,
)
from music_assistant.providers.listening_habits.helpers import (
    DurableQueue,
    QualityCache,
    device_type_from_model,
    guess_device_type_and_room,
    match_play,
    named_show,
    on_air_station_slug,
    on_air_url,
    station_playlist_url,
)
from music_assistant.providers.listening_habits.weather import (
    snapshot_from_hass_state,
    snapshot_from_open_meteo,
)

_LOGGER = logging.getLogger(__name__)


def test_podcast_episodes_are_supported_listens() -> None:
    """Completed podcast episodes reach the Listening Habits ingest path."""
    assert MediaType.PODCAST_EPISODE in SUPPORTED_MEDIA_TYPES


def test_audiobooks_are_supported_listens() -> None:
    """Completed audiobooks reach the Listening Habits ingest path."""
    assert MediaType.AUDIOBOOK in SUPPORTED_MEDIA_TYPES


def _playback_report(
    *, fully_played: bool, is_playing: bool, uri: str = "track://one"
) -> Any:
    """Build the small report surface exercised by the completion guard."""
    return type(
        "PlaybackReport",
        (),
        {
            "player_id": "multi-room-group",
            "uri": uri,
            "fully_played": fully_played,
            "is_playing": is_playing,
        },
    )()


def test_fully_played_group_report_logs_before_queue_advance() -> None:
    """A multi-room queue may never send a later stopped/advanced report."""
    provider = object.__new__(ListeningHabitsProvider)
    provider._last_counted = {}

    assert provider._should_log(_playback_report(fully_played=True, is_playing=True))
    assert not provider._should_log(_playback_report(fully_played=True, is_playing=False))


def test_restarted_track_rearms_completion_guard() -> None:
    """Repeat-one can count the same URI again after a non-complete report."""
    provider = object.__new__(ListeningHabitsProvider)
    provider._last_counted = {}

    assert provider._should_log(_playback_report(fully_played=True, is_playing=True))
    assert not provider._should_log(_playback_report(fully_played=False, is_playing=True))
    assert provider._should_log(_playback_report(fully_played=True, is_playing=True))


async def test_ambient_session_is_inserted_refreshed_and_finalized() -> None:
    """One ambient row appears at ten minutes and receives duration updates."""
    provider = object.__new__(ListeningHabitsProvider)
    provider._ambient_sessions = {}
    provider._last_counted = {}
    provider._weather_snapshot = AsyncMock(return_value={"weather_temperature_c": 20})
    provider._build_payload = MagicMock(return_value={})
    provider._push = AsyncMock(return_value=True)
    provider._backlog = SimpleNamespace(drain=AsyncMock(), append=AsyncMock())
    provider.mass = SimpleNamespace(
        player_queues=SimpleNamespace(get=lambda _: None),
        get_provider_icon_data=MagicMock(return_value="data:image/svg+xml;base64,cmFpbg=="),
        metadata=SimpleNamespace(
            get_image_url=MagicMock(
                return_value="http://music-assistant/imageproxy/rain-mood?size=0&fmt=svg"
            )
        ),
    )
    report = SimpleNamespace(
        player_id="player-one",
        uri="ambient_sounds://sound_effect/rain",
        name="Rain",
        seconds_played=599,
        is_playing=True,
    )

    await provider._handle_ambient_report(report)
    provider._push.assert_not_awaited()

    report.seconds_played = 600
    await provider._handle_ambient_report(report)
    provider._push.assert_awaited_once_with(
        {
            "artwork_url": "http://music-assistant/imageproxy/rain-mood?size=0&fmt=svg",
            "weather_temperature_c": 20,
        }
    )

    for update_number, threshold in enumerate(
        AMBIENT_SESSION_UPDATE_THRESHOLDS_S, start=2
    ):
        report.seconds_played = threshold - 1
        await provider._handle_ambient_report(report)
        assert provider._push.await_count == update_number - 1

        report.seconds_played = threshold
        await provider._handle_ambient_report(report)
        assert provider._push.await_count == update_number

    report.seconds_played = AMBIENT_SESSION_UPDATE_THRESHOLDS_S[-1] + 60
    report.is_playing = False
    await provider._handle_ambient_report(report)

    expected_durations = [600, *AMBIENT_SESSION_UPDATE_THRESHOLDS_S]
    expected_durations.append(AMBIENT_SESSION_UPDATE_THRESHOLDS_S[-1] + 60)
    assert [
        call.kwargs["duration_s"] for call in provider._build_payload.call_args_list
    ] == expected_durations
    played_at_values = {
        call.kwargs["played_at"] for call in provider._build_payload.call_args_list
    }
    assert len(played_at_values) == 1
    for call in provider._build_payload.call_args_list:
        assert call.kwargs["source_type"] == "ambient"
        assert call.kwargs["artist"] == "Ambient Sounds"
        assert call.kwargs["client_ref_prefix"] == "ma-ambient"
    provider.mass.get_provider_icon_data.assert_called_with(
        "ambient_sounds", ProviderIconVariant.DARK
    )
    expected_pushes = len(expected_durations)
    assert provider._push.await_count == expected_pushes
    assert provider._backlog.drain.await_count == expected_pushes


async def test_short_ambient_session_is_discarded() -> None:
    """Ambient sessions below the ten-minute threshold do not become listens."""
    provider = object.__new__(ListeningHabitsProvider)
    provider._ambient_sessions = {}
    provider._last_counted = {}
    provider._weather_snapshot = AsyncMock(return_value={})
    provider._push = AsyncMock(return_value=True)
    provider.mass = SimpleNamespace(player_queues=SimpleNamespace(get=lambda _: None))
    provider.logger = MagicMock()
    report = SimpleNamespace(
        player_id="player-one",
        uri="ambient_sounds://sound_effect/rain",
        name="Rain",
        seconds_played=599,
        is_playing=False,
    )

    await provider._handle_ambient_report(report)

    provider._push.assert_not_awaited()


async def test_ambient_stop_and_quick_resume_keeps_session_and_duration() -> None:
    """Stop -> Play inside the grace period updates the original ambient row."""
    provider = object.__new__(ListeningHabitsProvider)
    provider._ambient_sessions = {}
    provider._last_counted = {}
    provider._weather_snapshot = AsyncMock(return_value={})
    provider._build_payload = MagicMock(return_value={"artwork_url": "https://cover"})
    provider._push = AsyncMock(return_value=True)
    provider._backlog = SimpleNamespace(drain=AsyncMock(), append=AsyncMock())
    provider.mass = SimpleNamespace(player_queues=SimpleNamespace(get=lambda _: None))
    report = SimpleNamespace(
        player_id="player-one",
        uri="rain_mood://sound_effect/rain",
        name="Rain",
        seconds_played=600,
        is_playing=True,
    )

    await provider._handle_ambient_report(report)
    original_start = provider._ambient_sessions[(report.player_id, report.uri)]["started_at"]

    report.seconds_played = 650
    report.is_playing = False
    await provider._handle_ambient_report(report)

    session = provider._ambient_sessions[(report.player_id, report.uri)]
    session["stopped_at"] -= timedelta(seconds=AMBIENT_SESSION_RESUME_GRACE_S - 1)
    report.seconds_played = 10
    report.is_playing = True
    await provider._handle_ambient_report(report)

    assert session["started_at"] == original_start
    assert session["elapsed"] == 660
    assert "stopped_at" not in session
    assert provider._last_counted == {}


async def test_audiobook_chapter_crossing_logs_separate_entry() -> None:
    """Crossing a chapter end publishes that chapter under its book."""
    provider = object.__new__(ListeningHabitsProvider)
    provider._audiobook_sessions = {}
    provider._build_payload = MagicMock(return_value={})
    provider._weather_snapshot = AsyncMock(return_value={"weather_temperature_c": 20})
    provider._push = AsyncMock(return_value=True)
    provider._backlog = SimpleNamespace(drain=AsyncMock(), append=AsyncMock())
    chapter = SimpleNamespace(position=1, name="Opening Credits", start=0, end=19)
    audiobook = SimpleNamespace(metadata=SimpleNamespace(chapters=[chapter]))
    provider.mass = SimpleNamespace(
        music=SimpleNamespace(get_item_by_uri=AsyncMock(return_value=audiobook))
    )
    report = SimpleNamespace(
        player_id="web-player",
        uri="audible://audiobook/book-one",
        name="Do What You Want",
        artist="Bad Religion / Jim Ruland",
        seconds_played=8,
        duration=3600,
    )

    await provider._log_completed_audiobook_chapters(report)
    report.seconds_played = 20
    await provider._log_completed_audiobook_chapters(report)

    provider._build_payload.assert_called_once_with(
        report,
        duration_s=19,
        source_type="audiobook",
        artist="Bad Religion / Jim Ruland",
        title="Opening Credits",
        album="Do What You Want",
        client_ref_prefix="ma-audiobook-chapter-1",
    )
    provider._push.assert_awaited_once_with({"weather_temperature_c": 20})


async def test_podcast_session_logs_at_ten_minutes() -> None:
    """Podcast progress creates one session row after ten listened minutes."""
    provider = object.__new__(ListeningHabitsProvider)
    provider._podcast_sessions = {}
    provider._last_counted = {}
    provider._weather_snapshot = AsyncMock(return_value={})
    provider._save_podcast_sessions = AsyncMock()
    provider._publish_podcast = AsyncMock()
    provider.mass = SimpleNamespace(player_queues=SimpleNamespace(get=lambda _: None))
    report = SimpleNamespace(
        player_id="web-player",
        userid="listener",
        uri="apple_podcasts://podcast_episode/one",
        seconds_played=0,
        fully_played=False,
        is_playing=True,
        duration=3600,
    )

    await provider._handle_podcast_report(report)
    for position in range(100, 601, 100):
        report.seconds_played = position
        await provider._handle_podcast_report(report)

    provider._publish_podcast.assert_awaited_once()
    assert provider._publish_podcast.await_args.args[2] == 600


async def test_podcast_publish_distinguishes_episode_length_from_progress() -> None:
    """Podcast payload carries total length, heard position and completion separately."""
    provider = object.__new__(ListeningHabitsProvider)
    provider._build_payload = Mock(return_value={})
    provider._push = AsyncMock(return_value=True)
    provider._backlog = SimpleNamespace(drain=AsyncMock())
    report = SimpleNamespace(duration=3179)
    session = {
        "client_ref": "ma-podcast-session:listener:one",
        "started_at": datetime(2026, 9, 3, 15, 1, tzinfo=UTC),
        "weather": {},
    }

    await provider._publish_podcast(report, session, 979)

    call = provider._build_payload.call_args
    assert call.args == (report,)
    assert call.kwargs["duration_s"] == 3179
    assert call.kwargs["source_type"] == "podcast"
    assert call.kwargs["client_ref_prefix"] == "ma-podcast-session"
    assert call.kwargs["played_at"] is session["started_at"]
    payload = provider._push.await_args.args[0]
    assert payload["podcast_position_s"] == 979
    assert payload["podcast_completed"] is False
    assert abs(payload["podcast_checked_at"] - int(datetime.now(tz=UTC).timestamp())) < 2


async def test_completed_podcast_publish_marks_full_progress() -> None:
    """The terminal report explicitly marks the same podcast row complete."""
    provider = object.__new__(ListeningHabitsProvider)
    provider._build_payload = Mock(return_value={})
    provider._push = AsyncMock(return_value=True)
    provider._backlog = SimpleNamespace(drain=AsyncMock())
    report = SimpleNamespace(duration=3179)
    session = {
        "client_ref": "ma-podcast-session:listener:one",
        "started_at": datetime(2026, 9, 3, 15, 1, tzinfo=UTC),
        "weather": {},
    }

    await provider._publish_podcast(report, session, 3179, completed=True)

    payload = provider._push.await_args.args[0]
    assert payload["podcast_position_s"] == 3179
    assert payload["podcast_completed"] is True


async def test_unfinished_podcast_session_survives_restart(tmp_path: Path) -> None:
    """Podcast identity and progress are restored from durable provider state."""
    provider = object.__new__(ListeningHabitsProvider)
    provider._podcast_state_lock = asyncio.Lock()
    provider._podcast_state_path = str(tmp_path / "podcast-sessions.json")
    provider._podcast_sessions = {
        ("listener", "podcast://episode/one"): {
            "started_at": datetime(2026, 9, 3, 8, 0, tzinfo=UTC),
            "position": 1800,
            "observed_at": 0.0,
            "listened": 900,
            "reported": 600,
            "persisted_listened": 900,
            "client_ref": "ma-podcast-session:listener:1788440400",
            "weather": {"weather_temperature_c": 20},
        }
    }
    await provider._save_podcast_sessions()

    restored = object.__new__(ListeningHabitsProvider)
    restored._podcast_state_path = provider._podcast_state_path
    restored._podcast_sessions = {}
    restored._load_podcast_sessions()

    session = restored._podcast_sessions[("listener", "podcast://episode/one")]
    assert session["listened"] == 900
    assert session["reported"] == 600
    assert session["client_ref"] == "ma-podcast-session:listener:1788440400"


def test_hass_weather_state_maps_to_ingest_schema_and_converts_units() -> None:
    snapshot = snapshot_from_hass_state(
        {
            "state": "lightning-rainy",
            "last_updated": "2026-09-02T12:00:00Z",
            "attributes": {
                "temperature": 68,
                "temperature_unit": "°F",
                "apparent_temperature": 66.2,
                "cloud_coverage": 87,
                "wind_speed": 10,
                "wind_speed_unit": "mph",
            },
        }
    )
    assert snapshot == {
        "weather_observed_at": 1788350400,
        "weather_temperature_c": 20.0,
        "weather_apparent_temperature_c": 19.0,
        "weather_condition": "Lightning and rain",
        "weather_precipitation": "Lightning and rain",
        "weather_symbol": "cloud.bolt.rain.fill",
        "weather_cloud_cover_pct": 87,
        "weather_wind_kph": 16.09,
    }


def test_open_meteo_current_maps_to_ingest_schema() -> None:
    snapshot = snapshot_from_open_meteo(
        {"current": {"time": "2026-09-02T07:00", "temperature_2m": 14.5,
         "apparent_temperature": 13.2, "weather_code": 61, "cloud_cover": 92,
         "wind_speed_10m": 8.4}}
    )
    assert snapshot is not None
    assert snapshot["weather_condition"] == "Rainy"
    assert snapshot["weather_precipitation"] == "Rain"
    assert snapshot["weather_symbol"] == "cloud.rain.fill"


def test_hass_compact_condition_key_gets_a_readable_label() -> None:
    snapshot = snapshot_from_hass_state(
        {"state": "partlycloudy", "attributes": {"temperature": 29}}
    )
    assert snapshot is not None
    assert snapshot["weather_condition"] == "Partly cloudy"


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


@pytest.mark.parametrize(
    ("station_name", "expected"),
    [
        ("The Current", "the-current"),
        # A provider is free to decorate the name; the frequency and the
        # punctuation must not decide whether the DJ shows up.
        ("89.3 The Current", "the-current"),
        ("thecurrent", "the-current"),
        ("THE CURRENT", "the-current"),
        ("The Current Morning Show", "the-current"),
        ("KEXP", None),
        ("", None),
        (None, None),
    ],
)
def test_on_air_station_slug(station_name: str | None, expected: str | None) -> None:
    """A station name a player reported resolves to a log-server slug."""
    assert on_air_station_slug(station_name) == expected


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        (
            "http://192.168.0.115:8130/api/ingest",
            "http://192.168.0.115:8130/api/on-air?station=the-current",
        ),
        # Trailing slashes are a normal thing to paste into a config form.
        (
            "https://example.org/api/listens/",
            "https://example.org/api/on-air?station=the-current",
        ),
        # The sub-path mount is the case that rules out just taking the
        # origin: /lhs has to survive.
        (
            "https://example.org/lhs/api/ingest",
            "https://example.org/lhs/api/on-air?station=the-current",
        ),
        # No /api/ segment at all -- treat the whole thing as the root
        # rather than guessing which part of the path to discard.
        (
            "https://example.org/hooks/log",
            "https://example.org/hooks/log/api/on-air?station=the-current",
        ),
    ],
)
def test_on_air_url(endpoint: str, expected: str) -> None:
    """The on-air route is built as a sibling of the configured ingest URL."""
    assert on_air_url(endpoint, "the-current") == expected


@pytest.mark.parametrize(
    ("show_name", "expected"),
    [
        ("Teenage Kicks", "Teenage Kicks"),
        ("United States of Americana", "United States of Americana"),
        # The station's ordinary rotation. Every hour that is not a specialty
        # show carries this, and it is noise under the station's own name.
        ("The Current Music", None),
        ("the current music", None),
        ("  The Current Music  ", None),
        ("", None),
        (None, None),
    ],
)
def test_named_show(show_name: str | None, expected: str | None) -> None:
    """A generic rotation label is not a show worth displaying."""
    assert named_show(show_name) == expected


# Two plays as the log server returns them, shortened to the fields that matter
# to matching. Real values: these are the strings the feed actually served on
# 2026-08-28.
PLAYS: list[dict[str, object]] = [
    {
        "artist": "Noah Kahan",
        "title": "The Great Divide",
        "album": "Cape Elizabeth",
        "artwork_url": "https://albumart.publicradio.org/a.jpg",
        "duration_s": 213,
    },
    {
        "artist": "The Smashing Pumpkins",
        "title": "Tonight, Tonight",
        "album": "Mellon Collie and the Infinite Sadness",
        "artwork_url": "https://albumart.publicradio.org/b.jpg",
        "duration_s": 255,
    },
]


@pytest.mark.parametrize(
    ("icy_title", "expected_album"),
    [
        # The shape The Current actually sends: "Title-Artist", no spaces.
        # Splitting on the hyphen is not possible -- "The Great Divide" and
        # plenty of artist names carry their own.
        ("The Great Divide-Noah Kahan", "Cape Elizabeth"),
        # The feed writes "Tonight, Tonight"; the stream drops the comma.
        # Neither spelling is wrong and the difference means nothing.
        ("Tonight Tonight-The Smashing Pumpkins", "Mellon Collie and the Infinite Sadness"),
        # Ordering is an observation about this station, not a promise.
        ("Noah Kahan - The Great Divide", "Cape Elizabeth"),
        # What the stream sends during a talk break. No song is playing, so
        # matching nothing is the correct answer rather than a failure.
        ("The Current-", None),
        # A song the feed has not logged yet -- the stream runs ahead of it at
        # the start of a play.
        ("Some Song-Nobody", None),
        ("", None),
        (None, None),
    ],
)
def test_match_play(icy_title: str | None, expected_album: str | None) -> None:
    """An ICY title is matched to a logged play by identity, not by splitting."""
    match = match_play(icy_title, PLAYS)
    assert (match["album"] if match else None) == expected_album


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        (
            "https://host/api/ingest",
            "https://host/api/station-playlist?station=the-current&limit=12",
        ),
        # A server mounted under a sub-path keeps it.
        (
            "https://host/lhs/api/ingest",
            "https://host/lhs/api/station-playlist?station=the-current&limit=12",
        ),
    ],
)
def test_station_playlist_url(endpoint: str, expected: str) -> None:
    """The playlist route is built as a sibling of the configured ingest URL."""
    assert station_playlist_url(endpoint, "the-current") == expected
