"""Read-only Vinyl presentation data; never modifies a player or queue."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from .helpers import api_root

if TYPE_CHECKING:
    from . import ListeningHabitsProvider

VINYL_PLAYER_CONFIG = "vinyl_player_id"
DEFAULT_VINYL_PLAYER_ID = "wiim_uuid:FF98F7F4-FC5E-ACD3-2A59-9315FF98F7F4"
VINYL_CACHE_SECONDS = 1.0
VINYL_TIMEOUT_SECONDS = 3
PHONO_SOURCES = frozenset({"phono", "phono_in", "phono in"})


def vinyl_snapshot(payload: Any, player_id: str) -> dict[str, Any] | None:
    """Extract only explicitly live Vinyl metadata from the canonical feed."""
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("state") != "playing"
        or payload.get("source_type") != "vinyl"
        or payload.get("stale") is not False
        or payload.get("fallback_stale")
        or (payload.get("player_id") and payload["player_id"] != player_id)
    ):
        return None
    artwork = payload.get("artwork_url")
    try:
        parsed_artwork = urlsplit(artwork) if isinstance(artwork, str) else None
    except ValueError:
        parsed_artwork = None
    if (
        not parsed_artwork
        or parsed_artwork.scheme not in {"http", "https"}
        or not parsed_artwork.netloc
    ):
        artwork = None
    return {
        "player_id": player_id,
        "state": "playing",
        "source_type": "vinyl",
        "title": str(payload.get("title") or ""),
        "artist": str(payload.get("artist") or ""),
        "album": str(payload.get("album") or ""),
        "artwork_url": artwork,
    }


class VinylDisplayBridge:
    """Expose canonical Vinyl metadata only for its configured physical player."""

    def __init__(self, provider: ListeningHabitsProvider) -> None:
        """Initialize a short-lived, shared read cache."""
        self.provider = provider
        self._cache: tuple[float, dict[str, Any] | None] | None = None
        self._lock = asyncio.Lock()

    async def get_state(self, player_id: str) -> dict[str, Any] | None:
        """
        Read Vinyl display data without claiming ownership or changing playback.

        :param player_id: The player whose Now Playing screen is being displayed.
        """
        provider = self.provider
        target = provider.config.get_value(VINYL_PLAYER_CONFIG, DEFAULT_VINYL_PLAYER_ID)
        if player_id != target or not self._is_phono(player_id):
            return None
        if not provider._endpoint or not provider._token:
            return None
        async with self._lock:
            now = time.monotonic()
            if self._cache and now - self._cache[0] < VINYL_CACHE_SECONDS:
                result = self._cache[1]
            else:
                result = await self._read_snapshot(player_id)
                self._cache = (time.monotonic(), result)
        # A source switch can happen while the network request is outstanding.
        return result if self._is_phono(player_id) else None

    def _is_phono(self, player_id: str) -> bool:
        player = self.provider.mass.players.get(player_id)
        if player is None or not player.available or player.powered is False:
            return False
        media = player.current_media
        source = (
            str(player.active_source or (media.source_id if media else "") or "").strip().lower()
        )
        return source in PHONO_SOURCES

    async def _read_snapshot(self, player_id: str) -> dict[str, Any] | None:
        """Read the live feed, returning no metadata on any failure."""
        provider = self.provider
        try:
            async with provider.mass.http_session.get(
                f"{api_root(provider._endpoint)}/api/now-playing",
                headers={"Authorization": f"Bearer {provider._token}"},
                timeout=VINYL_TIMEOUT_SECONDS,
            ) as response:
                if response.status != 200:
                    return None
                return vinyl_snapshot(await response.json(), player_id)
        except Exception as err:
            provider.logger.debug("Vinyl display metadata unavailable: %s", err)
            return None
