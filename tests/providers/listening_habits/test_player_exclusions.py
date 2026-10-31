"""Tests for Listening Habits player logging exclusions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from music_assistant_models.enums import ConfigEntryType, PlayerType

from music_assistant.providers.listening_habits import (
    CONF_EXCLUDED_PLAYERS,
    ListeningHabitsProvider,
)


@pytest.mark.asyncio
async def test_player_config_defaults_to_logging_all_players() -> None:
    """The picker is an empty opt-out list, not an allow-list."""
    provider = object.__new__(ListeningHabitsProvider)
    provider.mass = MagicMock()
    provider.mass.webserver.auth.list_users = AsyncMock(return_value=[])
    provider.mass.players.all_players.return_value = [
        SimpleNamespace(player_id="kitchen", display_name="Kitchen", type=PlayerType.PLAYER),
        SimpleNamespace(player_id="display", display_name="Display", type=PlayerType.DISPLAY),
    ]
    provider.mass.get_provider.return_value = None
    provider.logger = Mock()

    entries = await provider.get_config_entries()
    by_key = {entry.key: entry for entry in entries}

    assert "scrobble_players" not in by_key
    excluded_players = by_key[CONF_EXCLUDED_PLAYERS]
    assert excluded_players.type == ConfigEntryType.STRING
    assert excluded_players.multi_value is True
    assert excluded_players.default_value == []
    assert [option.value for option in excluded_players.options] == ["kitchen"]


def test_only_checked_player_is_excluded() -> None:
    """An empty exclusion list keeps every player; a selected ID opts out."""
    provider = object.__new__(ListeningHabitsProvider)
    provider._excluded_player_ids = set()

    assert not provider._is_player_excluded("kitchen")
    assert not provider._is_player_excluded(None)

    provider._excluded_player_ids = {"kitchen"}
    assert provider._is_player_excluded("kitchen")
    assert not provider._is_player_excluded("bedroom")
