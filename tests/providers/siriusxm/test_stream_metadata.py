"""Test SiriusXM channel updates producing stream metadata with artwork."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

from music_assistant_models.enums import ContentType, MediaType, StreamType
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.streamdetails import StreamDetails
from sxm.models import (
    XMAlbum,
    XMArt,
    XMArtist,
    XMChannel,
    XMCut,
    XMCutMarker,
    XMImage,
    XMLiveChannel,
    XMSong,
)

from music_assistant.providers.siriusxm import SiriusXMProvider

CHANNEL_ID = "9420"
ALBUM_ART_URL = "https://example.com/album-art.jpg"
CHANNEL_LOGO_URL = "https://example.com/channel-logo.png"
SIRIUSXM_THUMB_URL = "https://albumart.siriusxm.com/albumart/2000/example_t.jpg"
SIRIUSXM_LARGE_URL = "https://albumart.siriusxm.com/albumart/2000/example_m.jpg"


def _make_channel() -> XMChannel:
    return XMChannel(
        guid="guid",
        id=CHANNEL_ID,
        name="Test Channel",
        streaming_name="Test Channel",
        sort_order=1,
        short_description="short",
        medium_description="medium",
        url="https://example.com/channel",
        is_available=True,
        is_favorite=False,
        is_mature=False,
        channel_number=42,
        images=[
            XMImage(
                name=None,
                url="https://example.com/other.png",
                art_type="IMAGE",
                width=100,
                height=100,
            ),
            XMImage(name=None, url=CHANNEL_LOGO_URL, art_type="IMAGE", width=300, height=300),
        ],
        categories=[],
    )


def _make_live_channel(cut: XMCut) -> XMLiveChannel:
    now = datetime.now(UTC)
    cut_marker = XMCutMarker(
        guid="cut-guid",
        time=now - timedelta(minutes=1),
        time_seconds=int((now - timedelta(minutes=1)).timestamp()),
        duration=timedelta(minutes=3),
        cut=cut,
    )
    return XMLiveChannel(
        id=CHANNEL_ID,
        hls_infos=[],
        custom_hls_infos=[],
        episode_markers=[],
        cut_markers=[cut_marker],
    )


def _make_provider(high_res_status: int = 404) -> Any:
    provider = MagicMock()
    provider._current_stream_details = StreamDetails(
        item_id=CHANNEL_ID,
        provider="siriusxm",
        audio_format=AudioFormat(content_type=ContentType.AAC),
        stream_type=StreamType.HLS,
        media_type=MediaType.RADIO,
        path="dummy",
    )
    provider._channels_by_id = {CHANNEL_ID: _make_channel()}
    provider._artwork_url_cache = {}
    provider._pending_artwork_urls = set()

    class HeadResponse:
        status = high_res_status

        async def __aenter__(self) -> HeadResponse:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    provider.mass.http_session.head.return_value = HeadResponse()
    provider.mass.create_task.side_effect = lambda task: asyncio.run(task)
    return provider


def _run_channel_updated(provider: Any, live_channel: XMLiveChannel) -> None:
    with patch.object(XMLiveChannel, "from_dict", return_value=live_channel):
        SiriusXMProvider._channel_updated(provider, {})


def test_channel_update_sets_song_album_art() -> None:
    """The current song's album art is used as stream metadata image."""
    provider = _make_provider()
    song = XMSong(
        title="Test Song",
        artists=[XMArtist(name="Test Artist")],
        album=XMAlbum(
            title="Test Album",
            arts=[XMArt(name=None, url=ALBUM_ART_URL, art_type="IMAGE")],
        ),
    )
    _run_channel_updated(provider, _make_live_channel(song))

    metadata = provider._current_stream_details.stream_metadata
    assert metadata is not None
    assert metadata.title == "Test Song"
    assert metadata.artist == "Test Artist"
    assert metadata.image_url == ALBUM_ART_URL


def test_channel_update_falls_back_to_channel_logo() -> None:
    """Without album art the channel logo is used as stream metadata image."""
    provider = _make_provider()
    song = XMSong(title="Test Song", artists=[XMArtist(name="Test Artist")], album=None)
    _run_channel_updated(provider, _make_live_channel(song))

    metadata = provider._current_stream_details.stream_metadata
    assert metadata is not None
    assert metadata.image_url == CHANNEL_LOGO_URL


def test_channel_update_non_song_cut_uses_channel_logo() -> None:
    """A cut that is not a song has no album art, so the channel logo is used."""
    provider = _make_provider()
    cut = XMCut(title="Talk Segment", artists=[XMArtist(name="Test Host")])
    _run_channel_updated(provider, _make_live_channel(cut))

    metadata = provider._current_stream_details.stream_metadata
    assert metadata is not None
    assert metadata.title == "Talk Segment"
    assert metadata.artist == "Test Host"
    assert metadata.image_url == CHANNEL_LOGO_URL


def test_channel_update_ignores_other_channel() -> None:
    """An update for another channel does not touch the current stream metadata."""
    provider = _make_provider()
    song = XMSong(title="Test Song", artists=[XMArtist(name="Test Artist")], album=None)
    live_channel = _make_live_channel(song)
    live_channel.id = "other-channel"
    _run_channel_updated(provider, live_channel)

    assert provider._current_stream_details.stream_metadata is None


def test_channel_update_uses_verified_larger_siriusxm_art() -> None:
    """A working larger SiriusXM album-art variant replaces its thumbnail."""
    provider = _make_provider(high_res_status=200)
    song = XMSong(
        title="Test Song",
        artists=[XMArtist(name="Test Artist")],
        album=XMAlbum(
            title="Test Album",
            arts=[XMArt(name=None, url=SIRIUSXM_THUMB_URL, art_type="IMAGE")],
        ),
    )

    _run_channel_updated(provider, _make_live_channel(song))

    metadata = provider._current_stream_details.stream_metadata
    assert metadata is not None
    assert metadata.image_url == SIRIUSXM_LARGE_URL
    provider.mass.http_session.head.assert_called_once_with(SIRIUSXM_LARGE_URL, timeout=5)


def test_channel_update_keeps_thumbnail_when_larger_siriusxm_art_is_missing() -> None:
    """A missing larger variant leaves the original artwork usable."""
    provider = _make_provider(high_res_status=404)
    song = XMSong(
        title="Test Song",
        artists=[XMArtist(name="Test Artist")],
        album=XMAlbum(
            title="Test Album",
            arts=[XMArt(name=None, url=SIRIUSXM_THUMB_URL, art_type="IMAGE")],
        ),
    )

    _run_channel_updated(provider, _make_live_channel(song))

    metadata = provider._current_stream_details.stream_metadata
    assert metadata is not None
    assert metadata.image_url == SIRIUSXM_THUMB_URL
    assert provider._artwork_url_cache[SIRIUSXM_LARGE_URL] is False
