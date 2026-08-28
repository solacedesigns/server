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
    named_show,
    on_air_station_slug,
    on_air_url,
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

# The on-air lookup is a scrape of the station's schedule page two hops away,
# so it is cached rather than made once per asking client. A block's own
# ends_at bounds the cache; these are the floor and ceiling around it. The
# floor stops a block that is about to end from being re-fetched on every
# poll; the ceiling keeps a long block (an overnight rotation is hours) from
# going stale if the schedule is changed mid-flight.
ON_AIR_MIN_CACHE_S = 60
ON_AIR_MAX_CACHE_S = 600
ON_AIR_TIMEOUT_S = 10

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
        # player_id -> uri of the last play counted for it; see _should_log.
        self._last_counted: dict[str, str] = {}
        # What the Now Playing indicator reports. Held in memory only: it
        # describes this run of the provider, and a restart genuinely has
        # nothing to say about pushes it did not make. The backlog on disk is
        # the part that must survive, and it does.
        self._unregister_api: Callable[[], None] | None = None
        self._unregister_on_air: Callable[[], None] | None = None
        self._logged_total = 0
        self._failed_total = 0
        self._last_result: str | None = None
        self._last_error: str | None = None
        self._last_logged: dict[str, Any] | None = None
        # slug -> (expires_at, payload). Payload may be None: "nothing is
        # scheduled right now" is a real answer and worth not re-asking for.
        self._on_air_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}

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
        # required_scope=None -> any authenticated user. This only reads back
        # what we already logged, and the frontend calls it to render a chip.
        self._unregister_api = self.mass.register_api_command(
            "listening_habits/status", self.get_status
        )
        self._unregister_on_air = self.mass.register_api_command(
            "listening_habits/on_air", self.get_on_air
        )

    async def unload(self, is_removed: bool = False) -> None:
        """Unsubscribe and stop the retry loop."""
        for unsub in self._on_unload:
            unsub()
        self._on_unload.clear()
        if self._unregister_api is not None:
            self._unregister_api()
            self._unregister_api = None
        if self._unregister_on_air is not None:
            self._unregister_on_air()
            self._unregister_on_air = None
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

    def _should_log(self, report: MediaItemPlaybackProgressReport) -> bool:
        """
        Return whether this report is a completed play we have not already logged.

        MEDIA_ITEM_PLAYED is signalled on *every* progress report, outside the
        `_should_mark_played` guard MA applies to its own playlog -- so a
        consumer that does not repeat that logic double-counts. A track is
        reported once as the current item as soon as it comes within 10s of the
        end (`is_playing` still true), and again when the queue advances past
        it; at end-of-queue the final track is reported twice over. Only the
        report where playback has actually stopped is the real completion.

        Keyed on uri because the report carries no queue_item_id, unlike the
        guard upstream.
        """
        key = report.player_id or ""
        if report.fully_played and not report.is_playing:
            if self._last_counted.get(key) == report.uri:
                return False
            self._last_counted[key] = report.uri
            return True
        # A not-fully-played report for the same item means it started over
        # (repeat one, or a manual seek back), so re-arm to count it again.
        if not report.fully_played and self._last_counted.get(key) == report.uri:
            del self._last_counted[key]
        return False

    async def _on_media_item_played(self, event: MassEvent) -> None:
        """Handle a finished media item by logging it."""
        report: MediaItemPlaybackProgressReport = event.data
        # This fires once per finished item, not once per progress tick, so an
        # unconditional line here is cheap -- and without it a play that got
        # filtered out is indistinguishable from an event that never arrived,
        # which is the hardest kind of silence to debug.
        self.logger.debug(
            "played event: %s - %s (%s, fully_played=%s, is_playing=%s, player=%s)",
            report.artist,
            report.name,
            report.media_type,
            report.fully_played,
            report.is_playing,
            report.player_id,
        )

        if report.media_type not in SUPPORTED_MEDIA_TYPES:
            self.logger.debug("skipped: unsupported media type %s", report.media_type)
            return
        if not self._should_log(report):
            self.logger.debug(
                "skipped: %s (fully_played=%s, is_playing=%s)",
                report.uri,
                report.fully_played,
                report.is_playing,
            )
            return
        cfg = self._scrobbler_config
        if cfg.mass_userids and report.userid not in cfg.mass_userids:
            self.logger.debug("skipped: user %s not in configured users", report.userid)
            return
        if cfg.mass_playerids and report.player_id not in cfg.mass_playerids:
            self.logger.debug("skipped: player %s not in configured players", report.player_id)
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
            "source_provider": self._describe_source_provider(
                streamdetails.provider if streamdetails else None
            ),
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

    def _describe_source_provider(self, instance_id: str | None) -> str | None:
        """Return a readable name for the streaming service, e.g. "Tidal"."""
        if not instance_id:
            return None
        # `StreamDetails.provider` is an *instance* id -- the domain plus a
        # suffix MA generated when that account was added, as in
        # "tidal--PoeusMTs". Unique and stable while the account lives, but it
        # reads like noise in a listening log, and the suffix is regenerated if
        # the account is ever removed and re-added, which would quietly split
        # one service into two in any group-by over the history.
        provider = self.mass.get_provider(instance_id, return_unavailable=True)
        if provider is None:
            # An account that has since been deleted no longer resolves. Keep
            # the raw id: unreadable beats losing the only record of where the
            # audio came from.
            return instance_id
        # Honours a name set in MA's provider settings, and disambiguates a
        # second account of the same service as "Tidal [2]".
        return provider.name

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
                    self._note_result("ok")
                    self._last_logged = {
                        "artist": payload.get("artist"),
                        "title": payload.get("title"),
                        "at": datetime.now(UTC).timestamp(),
                    }
                    return True
                # An auth failure is a *configuration* error, not a bad
                # payload: the play is perfectly good and will be accepted as
                # soon as the token is corrected. Dropping it here is how a
                # single mistyped token silently destroys every listen it
                # touches, so these go to the backlog instead.
                if response.status in (401, 403):
                    self.logger.error(
                        "server refused our token (%s), queued -- check the provider token "
                        "matches LHS_TOKEN on the log server",
                        response.status,
                    )
                    self._note_result("refused", f"token rejected ({response.status})")
                    return False
                # Any other 4xx is the server refusing this payload, not a
                # transport problem -- retrying it forever would wedge the
                # backlog behind a row that will never be accepted.
                if 400 <= response.status < 500 and response.status not in (408, 429):
                    self.logger.error(
                        "server rejected play (%s: %s), dropping: %s",
                        response.status,
                        (await response.text())[:200],
                        payload.get("client_ref"),
                    )
                    self._note_result("rejected", f"server rejected the play ({response.status})")
                    return True
                self.logger.warning("push failed with status %s, queued", response.status)
                self._note_result("error", f"HTTP {response.status}")
        except Exception as err:
            self.logger.warning("push failed (%s), queued", err)
            self._note_result("error", str(err))
        return False

    def _note_result(self, result: str, error: str | None = None) -> None:
        """
        Remember how the last push went, for the Now Playing indicator.

        "rejected" counts as a failure here even though _push returns True for
        it. True there means "stop retrying this payload", which is a backlog
        decision; from the listener's point of view the play was still lost,
        and an indicator that called that success would be lying.
        """
        self._last_result = result
        self._last_error = error
        if result == "ok":
            self._logged_total += 1
        else:
            self._failed_total += 1

    async def get_status(self) -> dict[str, Any]:
        """
        Report ingest health, for the frontend's Now Playing indicator.

        Deliberately says nothing about the *current* track. The provider logs
        on completion, so mid-play there is nothing yet to report about it --
        what a listener can actually use here is whether the pipe is open and
        whether anything is stuck behind it.

        Never returns the token, only whether one is set.
        """
        return {
            "configured": bool(self._endpoint and self._token),
            "endpoint": self._endpoint or None,
            "healthy": self._last_result in (None, "ok"),
            "backlog": await self._backlog.depth(),
            "logged_total": self._logged_total,
            "failed_total": self._failed_total,
            "last_result": self._last_result,
            "last_error": self._last_error,
            "last_logged": self._last_logged,
        }

    async def get_on_air(self, station: str | None = None) -> dict[str, Any] | None:
        """
        Report who is presenting on a live station right now, or None.

        Exists because MA has nowhere to get this. A station's show and DJ are
        not in the stream: ICY metadata carries the song and nothing else, and
        StreamMetadata.description -- the field built for exactly this -- is
        populated by only a couple of providers, none of which serve this
        station. The log server already scrapes the station's schedule page to
        attribute logged plays, so this asks it rather than growing a second
        scraper here that could disagree with the first.

        `station` is the station name as the player reports it, not a slug;
        resolving it is this provider's job, not the caller's.
        """
        slug = on_air_station_slug(station)
        if not slug or not self._endpoint or not self._token:
            return None
        now = datetime.now(UTC).timestamp()
        if (cached := self._on_air_cache.get(slug)) and cached[0] > now:
            return cached[1]
        try:
            async with self.mass.http_session.get(
                on_air_url(self._endpoint, slug),
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=ON_AIR_TIMEOUT_S,
            ) as response:
                if response.status >= 400:
                    self.logger.debug("on-air lookup for %s returned %s", slug, response.status)
                    return None
                block = self._as_on_air(await response.json())
        except Exception as err:
            # Nobody's listening is harmed by not knowing who the DJ is, so a
            # failure here is debug-level and simply uncached: the next poll
            # gets a fresh try rather than inheriting a shrugged-off answer.
            self.logger.debug("on-air lookup for %s failed: %s", slug, err)
            return None
        self._on_air_cache[slug] = (now + self._on_air_ttl(block, now), block)
        return block

    @staticmethod
    def _on_air_ttl(block: dict[str, Any] | None, now: float) -> float:
        """
        How long an on-air answer stays good for, from the block's own end time.

        A block knows when it is over, which beats any fixed interval: an hour
        into a three-hour show there is provably nothing to re-ask, and thirty
        seconds before the handover there provably is.
        """
        ends_at = (block or {}).get("ends_at")
        if not ends_at:
            return ON_AIR_MIN_CACHE_S
        try:
            remaining = datetime.fromisoformat(str(ends_at)).timestamp() - now
        except ValueError:
            return ON_AIR_MIN_CACHE_S
        return min(max(remaining, ON_AIR_MIN_CACHE_S), ON_AIR_MAX_CACHE_S)

    @staticmethod
    def _as_on_air(block: Any) -> dict[str, Any] | None:
        """
        Reshape a log-server on-air block for display, or None if it says nothing.

        A block with neither a host nor a nameable show is indistinguishable
        from no answer at all, so it is reported as none rather than as an
        object of nulls the caller then has to test field by field.
        """
        if not isinstance(block, dict):
            return None
        show = named_show(block.get("show_name"))
        host = str(block.get("host_name") or "").strip() or None
        if not show and not host:
            return None
        return {
            "station": block.get("station"),
            "show_name": show,
            "host_name": host,
            "hosts": block.get("hosts") or [],
            "starts_at": block.get("starts_at"),
            "ends_at": block.get("ends_at"),
        }
