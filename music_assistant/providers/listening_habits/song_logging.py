"""Playback-qualified song logging, independent of MA's completion flag."""

from __future__ import annotations

from dataclasses import dataclass

SONG_LOG_THRESHOLD_S = 60


@dataclass
class SongListen:
    """Accumulate observed playback without counting pauses or skipped positions."""

    position: float
    observed_at: float
    playing: bool
    started_at: float
    heard: float = 0
    claimed: bool = False

    def observe(self, position: float, playing: bool, now: float) -> None:
        """Credit only position movement bounded by time spent actually playing."""
        wall_elapsed = max(0.0, now - self.observed_at)
        advance = position - self.position
        if self.playing and advance > 0:
            self.heard += min(advance, wall_elapsed)
        self.position = position
        self.observed_at = now
        self.playing = playing

    @property
    def due(self) -> bool:
        """Return whether this unclaimed song has reached one minute."""
        return not self.claimed and self.heard >= SONG_LOG_THRESHOLD_S

    def claim(self) -> bool:
        """Reserve exactly one ingest attempt before asynchronous network work."""
        if not self.due:
            return False
        self.claimed = True
        return True
