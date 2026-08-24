"""Tests that one unreadable playlist file does not take down the whole listing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from music_assistant.providers.builtin import BuiltinProvider
from music_assistant.providers.builtin.constants import BUILTIN_PLAYLISTS


def _make_provider(playlists_dir: Path) -> BuiltinProvider:
    """Create a minimal BuiltinProvider that reads playlists from the given directory."""
    prov = object.__new__(BuiltinProvider)
    prov.mass = MagicMock()
    prov.logger = MagicMock()
    prov.manifest = MagicMock(domain="builtin")
    prov.config = MagicMock(instance_id="builtin")
    prov.config.get_value = MagicMock(return_value=True)
    prov._playlists_dir = str(playlists_dir)
    prov._playlist_lock = asyncio.Lock()
    prov.report_skipped_sync_item = MagicMock()
    return prov


@pytest.mark.asyncio
async def test_unreadable_playlist_file_does_not_hide_the_builtin_playlists(
    tmp_path: Path,
) -> None:
    """
    A playlist file that can not be decoded must skip only that playlist.

    The builtin playlists are listed after the user's own ones, so an error raised while
    reading a user playlist would otherwise keep every builtin playlist out of the library.
    """
    (tmp_path / "Good.m3u").write_text("#EXTM3U\n#PLAYLIST:Good\n", encoding="utf-8")
    # a file written by an external tool in a non-utf-8 encoding
    (tmp_path / "Bad.m3u").write_bytes(b"#EXTM3U\n#PLAYLIST:Caf\xe9 Rock\n")
    prov = _make_provider(tmp_path)

    names = [playlist.item_id async for playlist in prov.get_library_playlists()]

    assert "Good" in names
    assert "Bad" not in names
    assert set(BUILTIN_PLAYLISTS).issubset(names)
    prov.report_skipped_sync_item.assert_called_once()
