"""Tests for the read-only Vinyl display bridge."""

from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from music_assistant.providers.listening_habits.vinyl_display import (
    DEFAULT_VINYL_PLAYER_ID,
    VinylDisplayBridge,
    vinyl_snapshot,
)


class VinylSnapshotTests(unittest.TestCase):
    """Reject unconfirmed, stale and cross-player recognition data."""

    def setUp(self) -> None:
        """Build a live, explicitly tagged Vinyl response."""
        self.payload = {
            "state": "playing",
            "source_type": "vinyl",
            "stale": False,
            "title": "Bobby",
            "artist": "Phoebe Bridgers",
            "album": "Lost Weekend",
            "artwork_url": "https://example.com/cover.jpg",
            "player_id": None,
        }

    def test_recognized_metadata(self) -> None:
        """Keep all recognized fields, with no queue mutation or stream URL."""
        result = vinyl_snapshot(self.payload, DEFAULT_VINYL_PLAYER_ID)
        assert result is not None
        assert result["album"] == "Lost Weekend"
        assert result["player_id"] == DEFAULT_VINYL_PLAYER_ID
        assert set(result) == {
            "player_id",
            "state",
            "source_type",
            "title",
            "artist",
            "album",
            "artwork_url",
        }

    def test_only_explicitly_fresh_vinyl(self) -> None:
        """Physical Phono alone does not turn ordinary listening into Vinyl."""
        for overrides in (
            {"source_type": "shazam"},
            {"source_type": "radio"},
            {"state": "stopped"},
            {"stale": True},
            {"stale": None},
            {"fallback_stale": True},
            {"player_id": "other-player"},
        ):
            with self.subTest(overrides=overrides):
                assert (
                    vinyl_snapshot({**self.payload, **overrides}, DEFAULT_VINYL_PLAYER_ID) is None
                )
        assert vinyl_snapshot(None, DEFAULT_VINYL_PLAYER_ID) is None

    def test_waiting_and_invalid_artwork(self) -> None:
        """Never invent artwork or reuse an old cover while waiting."""
        for artwork in (None, "file:///secrets", "https://[invalid", "https:relative"):
            result = vinyl_snapshot(
                {
                    **self.payload,
                    "title": None,
                    "artist": None,
                    "album": None,
                    "artwork_url": artwork,
                },
                DEFAULT_VINYL_PLAYER_ID,
            )
            assert result is not None
            assert result["title"] == ""
            assert result["artwork_url"] is None


class VinylBridgeTests(unittest.IsolatedAsyncioTestCase):
    """Check physical-player eligibility and request races."""

    def setUp(self) -> None:
        """Build a provider stub without any player-control methods."""
        self.player = SimpleNamespace(
            available=True, powered=True, active_source="phono", current_media=None
        )
        self.provider = SimpleNamespace(
            config=MagicMock(),
            _endpoint="https://example.com/api/ingest",
            _token="test-only",
            mass=SimpleNamespace(
                players=MagicMock(spec=["get_player_state"]), http_session=MagicMock()
            ),
            logger=logging.getLogger(__name__),
        )
        self.provider.config.get_value.return_value = DEFAULT_VINYL_PLAYER_ID
        self.provider.mass.players.get_player_state.return_value = self.player
        self.bridge = VinylDisplayBridge(self.provider)

    async def test_rejects_other_players_and_streaming(self) -> None:
        """Do not fetch metadata for unrelated players or a native stream."""
        assert await self.bridge.get_state("kitchen") is None
        self.player.active_source = "ma-queue"
        assert await self.bridge.get_state(DEFAULT_VINYL_PLAYER_ID) is None
        self.provider.mass.http_session.get.assert_not_called()

    async def test_fresh_read_and_shared_cache(self) -> None:
        """Concurrent views share a short cache instead of duplicating reads."""
        result = {"title": "Bobby"}
        self.bridge._read_snapshot = AsyncMock(return_value=result)
        assert await self.bridge.get_state(DEFAULT_VINYL_PLAYER_ID) == result
        assert await self.bridge.get_state(DEFAULT_VINYL_PLAYER_ID) == result
        self.bridge._read_snapshot.assert_awaited_once()

    async def test_drops_reply_after_source_switch(self) -> None:
        """A late Phono lookup cannot override the new streaming source."""

        async def read(_player_id: str) -> dict[str, str]:
            self.player.active_source = "dlna"
            return {"title": "Bobby"}

        self.bridge._read_snapshot = read
        assert await self.bridge.get_state(DEFAULT_VINYL_PLAYER_ID) is None

    async def test_failed_http_read_has_no_history_fallback(self) -> None:
        """Network errors return no overlay, not a remembered record."""
        self.provider.mass.http_session.get.side_effect = OSError("offline")
        assert await self.bridge.get_state(DEFAULT_VINYL_PLAYER_ID) is None

    async def test_reads_authenticated_live_feed_and_filters_response(self) -> None:
        """Use the integration's existing token and publish only Vinyl metadata."""
        payload = {
            "state": "playing",
            "source_type": "vinyl",
            "stale": False,
            "title": "Bobby",
            "artist": "Phoebe Bridgers",
            "album": "Lost Weekend",
            "artwork_url": "https://example.com/cover.jpg",
        }
        response = MagicMock(status=200, json=AsyncMock(return_value=payload))
        context = self.provider.mass.http_session.get.return_value
        context.__aenter__.return_value = response
        result = await self.bridge.get_state(DEFAULT_VINYL_PLAYER_ID)
        assert result is not None
        assert result["title"] == "Bobby"
        self.provider.mass.http_session.get.assert_called_once_with(
            "https://example.com/api/now-playing",
            headers={"Authorization": "Bearer test-only"},
            timeout=3,
        )


if __name__ == "__main__":
    unittest.main()
