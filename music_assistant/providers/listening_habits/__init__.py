"""
Push every play to a self-hosted Listening Habits log.

This is the in-process successor to an AppDaemon app that watched Home
Assistant's `media_player.*` entities and pushed from outside. Moving it
inside Music Assistant removes the three workarounds that approach needed:

- No artwork settle delay. HA updates `media_title` before it finishes
  resolving the artwork proxy URL, so reading the entity on a track change
  reliably caught the *previous* track's cover; the app slept 3 seconds and
  re-read to dodge that. `MEDIA_ITEM_PLAYED` carries an already-resolved
  `image_url`, so there is nothing to wait for.
- No second WebSocket connection per track for audio quality.
- No parsing a provider out of a media_content_id: `StreamDetails.provider`
  says outright which service the audio came from.

What it deliberately does not reuse is `ScrobblerHelper`. That helper catches
a submission failure, logs it, and moves on -- the play is gone. Losing plays
is the one thing this log will not do, so the event handling is written here
against a durable on-disk backlog instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import EventType, MediaType, ProviderFeature

from music_assistant.helpers.scrobbler import ScrobblerConfig
from music_assistant.models.plugin import PluginProvider

from .helpers import (
    DurableQueue,
    QualityCache,
    device_type_from_model,
    guess_device_type_and_room,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from music_assistant_models.config_entries import ConfigEntry, ProviderConfig
    from music_assistant_models.event import MassEvent
    from music_assistant_models.playback_progress_report import MediaItemPlaybackProgressReport
    from music_assistant_models.provider import ProviderManifest
    from music_assistant_models.streamdetails import StreamDetails

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

CONF_ENDPOINT = "endpoint"
CONF_TOKEN = "_token"
CONF_HOME_PLACE = "home_place_name"

# How often the backlog is retried regardless of new plays, so a queue that
# built up overnight still drains even if nothing plays for a while.
RETRY_INTERVAL_S = 300
PUSH_TIMEOUT_S = 15

SUPPORTED_FEATURES: set[ProviderFeature] = set()

# Radio is logged as well as tracks -- a station play is a listen. Anything
# else (audiobook, podcast) is out of scope for this log for now.
SUPPORTED_MEDIA_TYPES = frozenset({MediaType.TRACK, MediaType.RADIO})


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return ListeningHabitsProvider(mass, manifest, config, SUPPORTED_FEATURES)


class ListeningHabitsProvider(PluginProvider):
    """Plugin provider that logs every play to a Listening Habits server."""

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
        supported_features: set[ProviderFeature],
    ) -> None:
        """Initialize."""
        super().__init__(mass, manifest, config, supported_features)
        self._on_unload: list[Callable[[], None]] = []
        self._quality = QualityCache()
        self._retry_task: asyncio.Task[None] | None = None

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return Config entries to configure this provider."""
        # Only entries that resolve without user input belong here: options are
        # validated against an empty value set when the instance is created, so
        # a required entry with no default makes the provider impossible to
        # load. The endpoint, token and home place are collected by the setup
        # flow into setup_data instead, and read back via get_setup_value.
        return tuple(await ScrobblerConfig.get_shared_config_entries(self.mass, None))

    async def handle_async_init(self) -> None:
        """Read configuration and prepare the backlog."""
        self._endpoint = str(self.get_setup_value(CONF_ENDPOINT) or "").strip()
        self._token = str(self.get_setup_value(CONF_TOKEN) or "").strip()
        self._home_place = str(self.get_setup_value(CONF_HOME_PLACE) or "").strip() or None
        self._scrobbler_config = ScrobblerConfig.create_from_config(self.config)
        self._backlog = DurableQueue(
            os.path.join(self.mass.storage_path, "listening_habits_queue.jsonl"),
            self.logger,
        )

    async def loaded_in_mass(self) -> None:
        """Subscribe to playback events once the provider is live."""
        await super().loaded_in_mass()
        self._on_unload.append(self.mass.subscribe(self._on_queue_updated, EventType.QUEUE_UPDATED))
        self._on_unload.append(
            self.mass.subscribe(self._on_media_item_played, EventType.MEDIA_ITEM_PLAYED)
        )
        self._retry_task = self.mass.create_task(self._retry_loop())

    async def unload(self, is_removed: bool = False) -> None:
        """Unsubscribe and stop the retry loop."""
        for unsub in self._on_unload:
            unsub()
        self._on_unload.clear()
        if self._retry_task and not self._retry_task.done():
            self._retry_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._retry_task

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------

    def _on_queue_updated(self, event: MassEvent) -> None:
        """
        Snapshot the current item's streamdetails while it is still current.

        See QualityCache: by the time MEDIA_ITEM_PLAYED fires for a track,
        the queue has usually already moved on to the next one.
        """
        queue = self.mass.player_queues.get(event.object_id)
        if queue is None or (item := queue.current_item) is None:
            return
        if item.media_item and item.streamdetails:
            self._quality.remember(item.media_item.uri, item.streamdetails)

    async def _on_media_item_played(self, event: MassEvent) -> None:
        """Handle a finished media item by logging it."""
        report: MediaItemPlaybackProgressReport = event.data

        if report.media_type not in SUPPORTED_MEDIA_TYPES:
            return
        if not report.fully_played:
            return
        cfg = self._scrobbler_config
        if cfg.mass_userids and report.userid not in cfg.mass_userids:
            return
        if cfg.mass_playerids and report.player_id not in cfg.mass_playerids:
            return

        payload = self._build_payload(report)
        if await self._push(payload):
            # Opportunistic drain: a push that just succeeded is good evidence
            # the server is reachable again.
            await self._backlog.drain(self._push)
        else:
            await self._backlog.append(payload)

    async def _retry_loop(self) -> None:
        """Drain the backlog periodically, so it empties even with nothing playing."""
        while True:
            await asyncio.sleep(RETRY_INTERVAL_S)
            try:
                await self._backlog.drain(self._push)
            except Exception:
                self.logger.exception("backlog drain failed")

    # ------------------------------------------------------------------
    # payload
    # ------------------------------------------------------------------

    def _build_payload(self, report: MediaItemPlaybackProgressReport) -> dict[str, Any]:
        """Map a playback report onto the log server's schema."""
        played_at = datetime.now(tz=UTC).astimezone()
        streamdetails: StreamDetails | None = self._quality.recall(report.uri)
        audio_format = streamdetails.audio_format if streamdetails else None
        # MA's own notion of lossless, rather than a hardcoded codec list.
        is_lossless = bool(audio_format and audio_format.content_type.is_lossless())
        # Live stream metadata -- the show/DJ data a Home Assistant entity
        # simply does not carry, so these were always null before.
        stream_meta = streamdetails.stream_metadata if streamdetails else None

        device_type, room, device_name = self._describe_player(report.player_id)
        title = report.name
        if self._scrobbler_config.suffix_version and report.version:
            title = f"{title} ({report.version})"

        return {
            "played_at": int(played_at.timestamp()),
            "played_at_local": played_at.isoformat(timespec="seconds"),
            "tz_offset_minutes": int((played_at.utcoffset() or timedelta(0)).total_seconds() // 60),
            "artist": report.artist,
            "title": title,
            "album": report.album,
            "duration_s": report.duration,
            "artwork_url": report.image_url,
            "source_type": "radio" if report.media_type is MediaType.RADIO else "streaming",
            # The actual streaming service, not the string "Music Assistant" --
            # this is what collapsed every row to one provider before.
            "source_provider": streamdetails.provider if streamdetails else None,
            "source_name": stream_meta.album if stream_meta else None,
            "source_app": "Music Assistant",
            "source_uri": report.uri,
            "show_name": stream_meta.title if stream_meta else None,
            "dj_name": stream_meta.artist if stream_meta else None,
            "device_type": device_type,
            "device": device_name,
            "player": device_name,
            "room": room,
            # A fixed player proves its own location; a carried one does not.
            # place_source "player" keeps that provenance honest, and matters
            # practically: the geocode pass only fills rows whose place_source
            # is null/geofence/geocoded, so this is never silently overwritten.
            "place": self._home_place if room else None,
            "place_type": "home" if room else None,
            "place_source": "player" if room else None,
            "audio_format": audio_format.content_type.value.upper() if audio_format else None,
            "bitrate_kbps": audio_format.bit_rate if audio_format else None,
            # bit_depth is only meaningful for lossless; a lossy codec's
            # nominal depth is an artifact of decoding, not of the source.
            "bit_depth": audio_format.bit_depth if audio_format and is_lossless else None,
            "sample_rate_hz": audio_format.sample_rate if audio_format else None,
            "is_lossless": is_lossless,
            "mbid": report.mbid,
            "artist_mbids": report.artist_mbids,
            "album_mbid": report.album_mbid,
            # Namespaced distinctly from the apps' "lh:" and the poller's
            # per-station refs, so a retry from here is never mistaken for a
            # play some other ingest already reported.
            "client_ref": f"ma:{report.player_id}:{int(played_at.timestamp())}",
            "ingest_method": "music_assistant",
        }

    def _describe_player(self, player_id: str | None) -> tuple[str | None, str | None, str | None]:
        """Return (device_type, room, display_name) for the player that played this."""
        if not player_id:
            return None, None, None
        # The report's `player_id` is the *queue* id (the playback tracker
        # passes `queue.queue_id`), which is the player's own id for a normal
        # queue but worth resolving through the queue rather than assuming.
        queue = self.mass.player_queues.get(player_id)
        player = self.mass.players.get_player(queue.queue_id if queue else player_id)
        if player is None:
            return None, None, None
        name = player.display_name
        device_type, room = guess_device_type_and_room(name)
        if device_type is not None:
            return device_type, room, name
        # The name carried no model word -- most players here are named for
        # where they are ("Kitchen", "Hallway"), so the whole name is the room
        # and the hardware has to come from the provider that created the
        # player rather than from the text.
        info = player.device_info
        device_type, fixed = device_type_from_model(
            info.manufacturer if info else None,
            info.model if info else None,
        )
        # A device that gets carried around has no meaningful room, even
        # though the name would otherwise read like one.
        return device_type, (name or None) if fixed else None, name

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------

    async def _push(self, payload: dict[str, Any]) -> bool:
        """POST one payload. Returns True only if the server accepted it."""
        try:
            async with self.mass.http_session.post(
                self._endpoint,
                json=payload,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=PUSH_TIMEOUT_S,
            ) as response:
                if response.status < 300:
                    self.logger.debug("logged %s - %s", payload.get("artist"), payload.get("title"))
                    return True
                # 4xx is the server refusing this payload, not a transport
                # problem -- retrying it forever would wedge the backlog behind
                # a row that will never be accepted.
                if 400 <= response.status < 500 and response.status not in (408, 429):
                    self.logger.error(
                        "server rejected play (%s), dropping: %s",
                        response.status,
                        payload.get("client_ref"),
                    )
                    return True
                self.logger.warning("push failed with status %s, queued", response.status)
        except Exception as err:
            self.logger.warning("push failed (%s), queued", err)
        return False
