"""Explicit provider selection cannot silently widen to another service."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ContentType, MediaType, QueueOption
from music_assistant_models.errors import InvalidDataError, MediaNotFoundError
from music_assistant_models.media_items import Album, ProviderMapping, SearchResults
from music_assistant_models.queue_item import QueueItem

from music_assistant.controllers.music import MusicController
from music_assistant.controllers.music.media.albums import AlbumsController
from music_assistant.controllers.player_queues.provider_selection import (
    ALLOWED_PROVIDERS_ATTRIBUTE,
    allowed_providers,
    restrict_queue_item,
)
from music_assistant.controllers.streams.audio import StreamsAudio
from music_assistant.controllers.streams.audio_buffer import AudioBuffer
from music_assistant.models.music_provider import ProviderStreamLimitError
from tests.controllers.player_queues.test_atomic_replace import (
    NEW_TRACKS,
    _controller,
    _item_ids,
    _load_playing_queue,
    _playlist,
    _track,
)
from tests.controllers.streams.test_stream_capacity import (
    BUSY_INSTANCE,
    FALLBACK_INSTANCE,
    _limit_error,
    _mapping,
    _mass,
    _music_provider,
    _queue_item,
    _streamdetails,
)


def test_restriction_is_a_copy_and_survives_cache_restore_and_handoff() -> None:
    """A queue-local restriction does not overwrite library mappings."""
    queue_item = _queue_item(_mapping(BUSY_INSTANCE), _mapping(FALLBACK_INSTANCE))
    original = queue_item.media_item
    restrict_queue_item(queue_item, [BUSY_INSTANCE])
    assert original is not None
    assert len(original.provider_mappings) == 2
    assert queue_item.media_item is not None
    assert len(queue_item.media_item.provider_mappings) == 1
    restored = QueueItem.from_cache(queue_item.to_cache())
    restored.queue_id = "handoff-target"
    assert allowed_providers(restored) == {BUSY_INSTANCE}


@pytest.mark.parametrize("value", [None, [], "qobuz", [""], [42]])
def test_invalid_stored_restriction_is_not_automatic(value: object) -> None:
    """Corrupt restrictions fail closed rather than playing an excluded source."""
    queue_item = _queue_item(_mapping(BUSY_INSTANCE))
    queue_item.extra_attributes[ALLOWED_PROVIDERS_ATTRIBUTE] = value  # type: ignore[assignment]
    with pytest.raises(MediaNotFoundError):
        allowed_providers(queue_item)


def test_missing_mapping_fails_and_automatic_keeps_all_candidates() -> None:
    """Restrictions do not manufacture a provider mapping from a library ID."""
    queue_item = _queue_item(_mapping(BUSY_INSTANCE))
    with pytest.raises(MediaNotFoundError, match="No other service was substituted"):
        restrict_queue_item(queue_item, [FALLBACK_INSTANCE])
    assert allowed_providers(queue_item) is None


async def test_lower_quality_selected_provider_wins_without_trying_the_excluded_service() -> None:
    """An excluded higher-quality mapping never gets a stream-details request."""
    selected = _music_provider(BUSY_INSTANCE, has_slot=True)
    excluded = _music_provider(FALLBACK_INSTANCE, has_slot=True)
    queue_item = _queue_item(_mapping(BUSY_INSTANCE), _mapping(FALLBACK_INSTANCE, ContentType.FLAC))
    queue_item.extra_attributes[ALLOWED_PROVIDERS_ATTRIBUTE] = json.dumps([BUSY_INSTANCE])
    audio = StreamsAudio(_mass({BUSY_INSTANCE: selected, FALLBACK_INSTANCE: excluded}))
    result = await audio.get_stream_details(queue_item)
    assert result.provider == BUSY_INSTANCE
    excluded.get_stream_details.assert_not_called()


async def test_cached_excluded_stream_is_not_reused() -> None:
    """Seeking or restoring must not reuse details from a previously allowed source."""
    selected = _music_provider(BUSY_INSTANCE, has_slot=True)
    excluded = _music_provider(FALLBACK_INSTANCE, has_slot=True)
    queue_item = _queue_item(_mapping(BUSY_INSTANCE), _mapping(FALLBACK_INSTANCE))
    queue_item.streamdetails = _streamdetails(FALLBACK_INSTANCE)
    queue_item.extra_attributes[ALLOWED_PROVIDERS_ATTRIBUTE] = json.dumps([BUSY_INSTANCE])
    audio = StreamsAudio(_mass({BUSY_INSTANCE: selected, FALLBACK_INSTANCE: excluded}))
    result = await audio.get_stream_details(queue_item)
    assert result.provider == BUSY_INSTANCE
    excluded.get_stream_details.assert_not_called()


def test_same_catalog_other_account_is_also_excluded() -> None:
    """The restriction is by exact instance, not merely by the service domain."""
    selected = _music_provider(BUSY_INSTANCE, has_slot=True)
    excluded = _music_provider(FALLBACK_INSTANCE, has_slot=True)
    audio = StreamsAudio(_mass({BUSY_INSTANCE: selected, FALLBACK_INSTANCE: excluded}))
    candidates = audio._get_streamdetail_candidates(
        [_mapping(BUSY_INSTANCE)], [], set(), {BUSY_INSTANCE}
    )
    assert [provider.instance_id for _, provider in candidates] == [BUSY_INSTANCE]


async def test_failure_on_selected_service_does_not_request_another_service() -> None:
    """A provider failure is surfaced instead of silently substituting a stream."""
    selected = _music_provider(BUSY_INSTANCE, has_slot=True)
    selected.get_stream_details = AsyncMock(side_effect=MediaNotFoundError("Unavailable"))
    excluded = _music_provider(FALLBACK_INSTANCE, has_slot=True)
    queue_item = _queue_item(_mapping(BUSY_INSTANCE), _mapping(FALLBACK_INSTANCE))
    queue_item.extra_attributes[ALLOWED_PROVIDERS_ATTRIBUTE] = json.dumps([BUSY_INSTANCE])
    audio = StreamsAudio(_mass({BUSY_INSTANCE: selected, FALLBACK_INSTANCE: excluded}))
    with pytest.raises(MediaNotFoundError):
        await audio.get_stream_details(queue_item)
    excluded.get_stream_details.assert_not_called()


async def test_stream_capacity_cannot_trigger_cross_provider_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep concurrency limits intact; never escape them by widening a strict request."""
    selected = _music_provider(BUSY_INSTANCE, has_slot=False)
    excluded = _music_provider(FALLBACK_INSTANCE, has_slot=True)
    queue_item = _queue_item(_mapping(BUSY_INSTANCE), _mapping(FALLBACK_INSTANCE))
    queue_item.media_item = _track("test")
    queue_item.media_item.provider_mappings = {_mapping(BUSY_INSTANCE), _mapping(FALLBACK_INSTANCE)}
    queue_item.streamdetails = _streamdetails(BUSY_INSTANCE)
    queue_item.extra_attributes[ALLOWED_PROVIDERS_ATTRIBUTE] = json.dumps([BUSY_INSTANCE])
    audio = StreamsAudio(_mass({BUSY_INSTANCE: selected, FALLBACK_INSTANCE: excluded}))
    discovery = AsyncMock(return_value={FALLBACK_INSTANCE})
    monkeypatch.setattr(audio, "_discover_alternative_provider_mappings", discovery)
    monkeypatch.setattr(audio, "_has_alternative_match_providers", lambda _: True)
    monkeypatch.setattr(
        AudioBuffer, "get_buffer", AsyncMock(side_effect=_limit_error(BUSY_INSTANCE))
    )
    with pytest.raises(ProviderStreamLimitError):
        await audio.get_audio_buffer(queue_item, reason="streaming", capacity_wait_timeout=0.1)
    discovery.assert_not_called()
    excluded.get_stream_details.assert_not_called()


async def test_provider_only_album_does_not_merge_library_track_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the selected provider's complete ordered album, not a merged library expansion."""
    ctrl = AlbumsController.__new__(AlbumsController)
    library_tracks = AsyncMock()
    monkeypatch.setattr(ctrl, "get_library_item_by_prov_id", AsyncMock(return_value=MagicMock()))
    monkeypatch.setattr(ctrl, "get_library_album_tracks", library_tracks)
    selected_tracks = [_track("first"), _track("second")]
    provider_tracks = AsyncMock(return_value=selected_tracks)
    monkeypatch.setattr(ctrl, "_get_provider_album_tracks", provider_tracks)
    provider_item = AsyncMock(
        return_value=Album(
            item_id="album",
            provider="test",
            name="Album",
            provider_mappings={
                ProviderMapping(item_id="album", provider_domain="test", provider_instance="test")
            },
        )
    )
    monkeypatch.setattr(ctrl, "get_provider_item", provider_item)
    result = await ctrl.tracks("album", "test", provider_only=True)
    assert [track.item_id for track in result] == ["first", "second"]
    provider_tracks.assert_awaited_once_with("album", "test")
    library_tracks.assert_not_called()


async def test_finite_request_sets_restriction_on_every_song() -> None:
    """All tracks in the expansion carry the restriction, not only the first one."""
    ctrl = _controller()
    ctrl.mass.get_providers.return_value = [MagicMock(instance_id="test")]
    ctrl.mass.music.get_controller.return_value.get_provider_item = AsyncMock(
        return_value=_playlist()
    )
    await ctrl.play_media(
        "q1", _playlist(), QueueOption.REPLACE, allowed_provider_instances=["test"]
    )
    assert _item_ids(ctrl) == NEW_TRACKS
    assert all(allowed_providers(item) == {"test"} for item in ctrl._queue_data["q1"].items)


@pytest.mark.parametrize("instances", [[], ["missing"]])
async def test_invalid_request_preserves_playing_queue(instances: list[str]) -> None:
    """An invalid restriction fails before replacing the user's existing queue."""
    ctrl = _controller()
    ctrl.mass.get_providers.return_value = [MagicMock(instance_id="test")]
    playing = _load_playing_queue(ctrl)
    with pytest.raises(InvalidDataError):
        await ctrl.play_media(
            "q1", _track("new"), QueueOption.REPLACE, allowed_provider_instances=instances
        )
    assert ctrl._queue_data["q1"].items == playing


async def test_one_unavailable_song_does_not_replace_queue_with_partial_album() -> None:
    """Validate every expanded song before enqueuing any of a restricted album."""
    ctrl = _controller()
    ctrl.mass.get_providers.return_value = [MagicMock(instance_id="test")]
    playing = _load_playing_queue(ctrl)
    unavailable = _track("excluded")
    ctrl.mass.music.get_controller.return_value.get_provider_item = AsyncMock(
        return_value=_playlist()
    )
    unavailable.provider_mappings = {
        ProviderMapping(item_id="other", provider_domain="other", provider_instance="other")
    }
    ctrl._media_resolver._resolve_media_items = AsyncMock(return_value=[_track("new"), unavailable])
    with pytest.raises(MediaNotFoundError):
        await ctrl.play_media(
            "q1", _playlist(), QueueOption.REPLACE, allowed_provider_instances=["test"]
        )
    assert ctrl._queue_data["q1"].items == playing


async def test_provider_search_does_not_hide_saved_songs_when_library_is_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider-only search must retain matches already present in the library."""
    music = MusicController.__new__(MusicController)
    music.mass = MagicMock()
    music.mass.cache.get = AsyncMock(return_value=None)
    music.mass.cache.set = AsyncMock()
    music.logger = MagicMock()
    monkeypatch.setattr(music, "get_unique_providers", MagicMock(return_value=["test"]))
    monkeypatch.setattr(music, "get_active_provider_instances", MagicMock(return_value=["test"]))
    music.mass.get_providers_supporting_feature.return_value = []
    music.mass.get_provider.return_value = MagicMock(instance_id="test", domain="test")
    monkeypatch.setattr(music, "_search_shareable_url", AsyncMock(return_value=None))
    monkeypatch.setattr(
        music, "search_library", AsyncMock(return_value=SearchResults(tracks=[_track("saved")]))
    )
    provider_search = AsyncMock(return_value=SearchResults(tracks=[_track("saved")]))
    monkeypatch.setattr(music, "_search_provider", provider_search)
    result = await music.search("saved", media_types=[MediaType.TRACK], providers=["test"])
    assert len(result.tracks) == 1
    assert provider_search.await_args is not None
    assert provider_search.await_args.kwargs["skip_item_ids"] == set()
