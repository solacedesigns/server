"""Tests for the LRCLIB metadata provider."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientError
from music_assistant_models.enums import MediaType
from music_assistant_models.errors import MusicAssistantError
from music_assistant_models.media_items import ItemMapping, Track, UniqueList

from music_assistant.constants import LYRICS_LOOKUP_PROVIDER
from music_assistant.helpers.throttle_retry import ThrottlerManager
from music_assistant.providers.lrclib import SUPPORTED_FEATURES, LrclibProvider
from tests.common import use_real_create_task


@pytest.fixture
def provider() -> LrclibProvider:
    """Return an LrclibProvider with mocked dependencies and a cold cache."""
    mass = AsyncMock()
    mass.http_session = MagicMock()
    # force a cache miss so the wrapped fetch always runs
    mass.cache.get_with_freshness = AsyncMock(return_value=(None, False, False))
    use_real_create_task(mass)
    manifest = MagicMock()
    manifest.domain = "lrclib"
    config = MagicMock()
    config.get_value.return_value = "GLOBAL"
    prov = LrclibProvider(mass, manifest, config, SUPPORTED_FEATURES)
    prov.api_url = "https://lrclib.net/api"
    # retry_attempts=1 keeps the transient-error test fast (no backoff sleeps)
    prov.throttler = ThrottlerManager(rate_limit=1, period=1, retry_attempts=1)
    return prov


def _response_cm(*, response: MagicMock | None = None, exc: Exception | None = None) -> MagicMock:
    """Build a fake async context manager mimicking aiohttp's session.get()."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=response, side_effect=exc)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


async def test_get_data_returns_none_when_not_found(provider: LrclibProvider) -> None:
    """A 404 means there are genuinely no lyrics, so None is returned (and cached)."""
    response = MagicMock()
    response.status = 404
    response.raise_for_status = MagicMock()
    provider.mass.http_session.get = MagicMock(  # type: ignore[method-assign]
        return_value=_response_cm(response=response)
    )

    assert await provider._get_data(track_name="Song", artist_name="Artist") is None


async def test_get_data_propagates_transient_error(provider: LrclibProvider) -> None:
    """A network failure surfaces as a MusicAssistantError instead of being cached as 'no lyrics'."""
    provider.mass.http_session.get = MagicMock(  # type: ignore[method-assign]
        return_value=_response_cm(exc=ClientError("network down"))
    )

    with pytest.raises(MusicAssistantError):
        await provider._get_data(track_name="Song", artist_name="Artist")


def _track(provider: str, duration: int) -> Track:
    """Build a Track the way the lyrics lookup sees one, real or standing in for a stream."""
    return Track(
        item_id="1",
        provider=provider,
        name="Song",
        duration=duration,
        provider_mappings=set(),
        artists=UniqueList(
            [
                ItemMapping(
                    media_type=MediaType.ARTIST, item_id="a", provider=provider, name="Artist"
                )
            ]
        ),
    )


def _fake_get_data(
    provider: LrclibProvider, *results: dict[str, str] | None
) -> list[dict[str, Any]]:
    """Answer each _get_data call with the next result, recording the params it was asked with."""
    calls: list[dict[str, Any]] = []

    async def _get_data(**params: Any) -> dict[str, str] | None:
        calls.append(params)
        return results[len(calls) - 1]

    provider._get_data = _get_data  # type: ignore[method-assign]
    return calls


async def test_stream_track_retries_without_duration(provider: LrclibProvider) -> None:
    """A station's duration is scheduling data, so a miss on it is retried without it."""
    calls = _fake_get_data(provider, None, {"plainLyrics": "words"})

    metadata = await provider.get_track_metadata(_track(LYRICS_LOOKUP_PROVIDER, 200))

    assert [c.get("duration") for c in calls] == [200, None]
    assert metadata is not None
    assert metadata.lyrics == "words"


async def test_library_track_does_not_retry(provider: LrclibProvider) -> None:
    """A library track is tagged from the recording, so its duration stays part of the search."""
    calls = _fake_get_data(provider, None)

    assert await provider.get_track_metadata(_track("filesystem_local", 200)) is None
    assert len(calls) == 1


async def test_stream_track_without_duration_still_searches(provider: LrclibProvider) -> None:
    """Some stations report no duration at all; artist and title alone are still worth asking."""
    calls = _fake_get_data(provider, {"syncedLyrics": "[00:01.00] words"})

    metadata = await provider.get_track_metadata(_track(LYRICS_LOOKUP_PROVIDER, 0))

    assert len(calls) == 1
    assert "duration" not in calls[0]
    assert metadata is not None
    assert metadata.lrc_lyrics == "[00:01.00] words"


async def test_library_track_without_duration_is_skipped(provider: LrclibProvider) -> None:
    """An untimed library track is a tagging gap, not a stream, and is still skipped."""
    calls = _fake_get_data(provider, {"plainLyrics": "words"})

    assert await provider.get_track_metadata(_track("filesystem_local", 0)) is None
    assert calls == []
