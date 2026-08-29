"""Tests for `get_lyrics_by_name`, the lyrics lookup for a track with no media item."""

from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import ProviderFeature, ProviderType
from music_assistant_models.media_items import MediaItemMetadata, Track

from music_assistant.controllers.metadata import MetaDataController
from music_assistant.controllers.metadata.controller import LYRICS_LOOKUP_PROVIDER


def _controller() -> MetaDataController:
    """Create a bare MetaDataController without running __init__."""
    return MetaDataController.__new__(MetaDataController)


def _capturing_controller() -> tuple[MetaDataController, list[Track]]:
    """Return a controller that records the track handed to the lyrics lookup."""
    ctrl = _controller()
    seen: list[Track] = []

    async def _lookup(track: Track) -> tuple[str | None, str | None]:
        seen.append(track)
        return None, None

    ctrl._get_track_lyrics = _lookup  # type: ignore[method-assign]
    return ctrl, seen


class _FakeLyricsProvider:
    """Metadata provider that answers every lookup with the same lyrics."""

    priority = 0
    name = "fake"
    supported_features: ClassVar[set[ProviderFeature]] = {ProviderFeature.LYRICS}

    def __init__(self, lyrics: str | None = None, lrc_lyrics: str | None = None) -> None:
        self.lyrics = lyrics
        self.lrc_lyrics = lrc_lyrics
        self.seen: list[Track] = []

    async def get_track_metadata(self, track: Track) -> MediaItemMetadata:
        """Return the canned lyrics and remember what was asked for."""
        self.seen.append(track)
        metadata = MediaItemMetadata()
        metadata.lyrics = self.lyrics
        metadata.lrc_lyrics = self.lrc_lyrics
        return metadata


def _controller_with(provider: _FakeLyricsProvider) -> MetaDataController:
    """Return a controller whose only metadata provider is the given one."""
    ctrl = _controller()

    def _get_providers(provider_type: ProviderType) -> list[Any]:
        return [provider] if provider_type == ProviderType.METADATA else []

    ctrl.mass = SimpleNamespace(  # type: ignore[assignment]
        # no provider is named LYRICS_LOOKUP_PROVIDER, which is the point of it
        get_provider=MagicMock(return_value=None),
        get_providers=_get_providers,
    )
    return ctrl


class TestStandInTrack:
    """The stand-in Track carries exactly what the lyrics providers read off it."""

    @pytest.mark.asyncio
    async def test_all_four_fields_reach_the_lookup(self) -> None:
        """Title, artist, album and duration all land on the stand-in."""
        ctrl, seen = _capturing_controller()
        await ctrl.get_lyrics_by_name(
            title="Walking on a Dream",
            artist="Empire of the Sun",
            album="Walking On a Dream",
            duration=199,
        )
        track = seen[0]
        assert track.name == "Walking on a Dream"
        assert track.artists[0].name == "Empire of the Sun"
        assert track.album is not None
        assert track.album.name == "Walking On a Dream"
        assert track.duration == 199

    @pytest.mark.asyncio
    async def test_album_is_omitted_rather_than_blank(self) -> None:
        """An unspecified album leaves the stand-in's album unset."""
        # lrclib sends album_name through as a search filter, so a blank stand-in
        # album would narrow the search rather than leave it open.
        ctrl, seen = _capturing_controller()
        await ctrl.get_lyrics_by_name(title="Kinky", artist="bby")
        assert seen[0].album is None

    @pytest.mark.asyncio
    async def test_missing_duration_becomes_zero(self) -> None:
        """An unspecified duration becomes the falsy int the providers expect."""
        # Track.duration is an int; lrclib reads `track.duration or 0` and skips
        # the lookup on a falsy one, which is the correct outcome here.
        ctrl, seen = _capturing_controller()
        await ctrl.get_lyrics_by_name(title="Kinky", artist="bby")
        assert seen[0].duration == 0

    @pytest.mark.asyncio
    async def test_provider_matches_nothing_loaded(self) -> None:
        """The stand-in's provider is a sentinel that resolves to no loaded provider."""
        # The sentinel keeps _get_track_lyrics off both the library branch and the
        # "ask the item's own provider" branch, neither of which can serve a stream.
        ctrl, seen = _capturing_controller()
        await ctrl.get_lyrics_by_name(title="Kinky", artist="bby")
        assert seen[0].provider == LYRICS_LOOKUP_PROVIDER
        assert seen[0].provider != "library"


class TestLookupResult:
    """A name-only lookup reaches the metadata providers and normalizes on the way out."""

    @pytest.mark.asyncio
    async def test_plain_lyrics_are_returned(self) -> None:
        """Plain lyrics from a metadata provider come back unchanged."""
        provider = _FakeLyricsProvider(lyrics="First line\nSecond line")
        ctrl = _controller_with(provider)
        lyrics, lrc_lyrics = await ctrl.get_lyrics_by_name(
            title="A Little Pain", artist="Margo Price", duration=214
        )
        assert lyrics == "First line\nSecond line"
        assert lrc_lyrics is None
        assert provider.seen[0].name == "A Little Pain"

    @pytest.mark.asyncio
    async def test_synced_lyrics_are_normalized(self) -> None:
        """LRC id tags are stripped from synced lyrics on the way out."""
        # get_track_lyrics strips LRC id tags on the way out; a name-only lookup is
        # never stored in the library db, so this is the only pass it gets.
        provider = _FakeLyricsProvider(
            lrc_lyrics="[ar:Semisonic]\n[00:12.30]First line\n[00:15.00]Second line"
        )
        ctrl = _controller_with(provider)
        _, lrc_lyrics = await ctrl.get_lyrics_by_name(
            title="The Rope", artist="Semisonic", duration=232
        )
        assert lrc_lyrics == "[00:12.30]First line\n[00:15.00]Second line"

    @pytest.mark.asyncio
    async def test_no_lyrics_found_returns_a_pair_of_nones(self) -> None:
        """A lookup that finds nothing returns (None, None)."""
        ctrl = _controller_with(_FakeLyricsProvider())
        assert await ctrl.get_lyrics_by_name(title="Kinky", artist="bby") == (None, None)
