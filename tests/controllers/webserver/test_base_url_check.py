"""Tests for the guard that keeps an unusable configured Base URL off the network."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientConnectorError
from music_assistant_models.enums import ConfigEntryType
from music_assistant_models.errors import InvalidDataError

from music_assistant.constants import CONF_VALUE_AUTO
from music_assistant.controllers.webserver.controller import (
    CONF_BASE_URL,
    WebserverController,
    _points_at_this_host,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from music_assistant_models.config_entries import CoreConfig

AUTO_URL = "http://192.168.1.5:8095"
SERVER_ID = "1234567890"
HOST_ADDRESSES = ("192.168.1.5", "fd00::5")


class _FakeResponse:
    """Stand-in for the aiohttp response of a /info request."""

    def __init__(self, status: int, payload: Any) -> None:
        self.status = status
        self._payload = payload

    async def json(self, content_type: str | None = None) -> Any:
        """Return the decoded payload, raising when it is not valid JSON."""
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _fake_session(outcome: _FakeResponse | Exception, probed: list[str]) -> MagicMock:
    """
    Build an HTTP session double that answers every GET with the given outcome.

    :param outcome: The response to hand out, or the error to raise on connecting.
    :param probed: List that each requested URL is appended to.
    """

    @asynccontextmanager
    async def _get(url: str, **kwargs: Any) -> AsyncIterator[_FakeResponse]:  # noqa: ARG001
        probed.append(url)
        if isinstance(outcome, Exception):
            raise outcome
        yield outcome

    session = MagicMock()
    session.get = MagicMock(side_effect=_get)
    return session


def _connection_error() -> ClientConnectorError:
    """Return the error aiohttp raises when nothing is listening."""
    return ClientConnectorError(MagicMock(), OSError(111, "Connection refused"))


@pytest.fixture
def mock_mass() -> MagicMock:
    """Create a mock Music Assistant instance carrying this server's identity."""
    mass = MagicMock()
    mass.config.get_raw_core_config_value.return_value = "GLOBAL"
    mass.server_id = SERVER_ID
    mass.providers = []
    mass.discovery.republish_mass_service = AsyncMock()
    return mass


def _make_webserver(mock_mass: MagicMock, base_url: str) -> WebserverController:
    """
    Create a controller carrying the state that setup() resolves.

    :param mock_mass: Mocked Music Assistant instance to build the controller on.
    :param base_url: Value for the base_url config option.
    """
    webserver = WebserverController(mock_mass)
    config = MagicMock()
    config.get_value.side_effect = lambda key, default=None: {
        CONF_BASE_URL: base_url,
    }.get(key, default)
    webserver.config = cast("CoreConfig", config)
    webserver.publish_port = 8095
    webserver.publish_ip = "192.168.1.5"
    webserver.bind_ip = None
    webserver._auto_base_url = AUTO_URL
    webserver._advertised_base_url = webserver.base_url
    return webserver


async def _run_check(
    webserver: WebserverController, outcome: _FakeResponse | Exception
) -> list[str]:
    """
    Run the base URL check against a single canned probe outcome.

    :param webserver: The prepared controller to run the check on.
    :param outcome: The response the probe receives, or the error it runs into.
    :return: The URLs that were probed.
    """
    probed: list[str] = []
    cast("MagicMock", webserver.mass).http_session_no_ssl = _fake_session(outcome, probed)
    with patch(
        "music_assistant.controllers.webserver.controller.get_ip_addresses",
        AsyncMock(return_value=HOST_ADDRESSES),
    ):
        await webserver._check_configured_base_url()
    return probed


async def _base_url_alerted(webserver: WebserverController) -> bool:
    """
    Return whether the settings UI renders the alert about an unusable Base URL.

    :param webserver: The controller to read the config entries from.
    """
    with patch(
        "music_assistant.controllers.webserver.controller.get_ip_addresses",
        AsyncMock(return_value=HOST_ADDRESSES),
    ):
        entries = await webserver._build_config_entries()
    return any(
        entry.key == "base_url_unreachable_warn"
        and entry.type == ConfigEntryType.ALERT
        and not entry.hidden
        for entry in entries
    )


async def test_auto_setting_is_never_probed(mock_mass: MagicMock) -> None:
    """The auto URL is derived from what the server binds to, so there is nothing to check."""
    webserver = _make_webserver(mock_mass, CONF_VALUE_AUTO)

    probed = await _run_check(webserver, _FakeResponse(200, {"server_id": SERVER_ID}))

    assert probed == []
    assert webserver.base_url == AUTO_URL


async def test_url_that_reaches_this_server_is_kept(mock_mass: MagicMock) -> None:
    """A base URL that hands back our own server info is exactly what should be advertised."""
    webserver = _make_webserver(mock_mass, "https://ma.example.com")

    probed = await _run_check(webserver, _FakeResponse(200, {"server_id": SERVER_ID}))

    assert probed == ["https://ma.example.com/info"]
    assert webserver.base_url == "https://ma.example.com"
    assert not await _base_url_alerted(webserver)
    mock_mass.discovery.republish_mass_service.assert_not_awaited()


async def test_dead_port_on_our_own_address_falls_back_to_the_auto_url(
    mock_mass: MagicMock,
) -> None:
    """A port stripped off our own IP cannot reach us, and must not be handed to clients."""
    webserver = _make_webserver(mock_mass, "http://192.168.1.5")

    await _run_check(webserver, _connection_error())

    assert webserver.base_url == AUTO_URL
    assert await _base_url_alerted(webserver)
    # clients that already adopted the broken URL only recover once it is re-broadcast
    mock_mass.discovery.republish_mass_service.assert_awaited_once()


async def test_a_different_server_falls_back_to_the_auto_url(mock_mass: MagicMock) -> None:
    """Something answering /info that is not this server is just as unusable."""
    webserver = _make_webserver(mock_mass, "http://192.168.1.9:8095")

    await _run_check(webserver, _FakeResponse(200, {"server_id": "some-other-server"}))

    assert webserver.base_url == AUTO_URL


async def test_a_non_music_assistant_response_falls_back_to_the_auto_url(
    mock_mass: MagicMock,
) -> None:
    """Another application on that address answers, but never with our server info."""
    webserver = _make_webserver(mock_mass, "http://192.168.1.5")

    await _run_check(webserver, _FakeResponse(200, ValueError("not json")))

    assert webserver.base_url == AUTO_URL


async def test_an_unreachable_hostname_is_left_alone(mock_mass: MagicMock) -> None:
    """A reverse proxy may be reachable for clients while it is not for us."""
    webserver = _make_webserver(mock_mass, "https://ma.example.com")

    await _run_check(webserver, _connection_error())

    assert webserver.base_url == "https://ma.example.com"
    assert not await _base_url_alerted(webserver)


async def test_an_unreachable_foreign_address_is_left_alone(mock_mass: MagicMock) -> None:
    """An address this host does not hold may still be routable from the network."""
    webserver = _make_webserver(mock_mass, "http://192.168.1.9")

    await _run_check(webserver, _connection_error())

    assert webserver.base_url == "http://192.168.1.9"


async def test_a_gated_info_endpoint_is_left_alone(mock_mass: MagicMock) -> None:
    """A proxy keeping /info behind its own authentication is not proof of a broken URL."""
    webserver = _make_webserver(mock_mass, "http://192.168.1.5")

    await _run_check(webserver, _FakeResponse(401, None))

    assert webserver.base_url == "http://192.168.1.5"


async def test_a_recovered_url_is_advertised_again(mock_mass: MagicMock) -> None:
    """A reverse proxy that only comes up later is adopted without needing a restart."""
    webserver = _make_webserver(mock_mass, "http://192.168.1.5")
    await _run_check(webserver, _connection_error())
    assert webserver.base_url == AUTO_URL
    mock_mass.discovery.republish_mass_service.reset_mock()

    await _run_check(webserver, _FakeResponse(200, {"server_id": SERVER_ID}))

    assert webserver.base_url == "http://192.168.1.5"
    assert not await _base_url_alerted(webserver)
    mock_mass.discovery.republish_mass_service.assert_awaited_once()


async def test_a_changed_url_is_rebroadcast(mock_mass: MagicMock) -> None:
    """The setting takes effect without a reload, so the mDNS record must follow it."""
    webserver = _make_webserver(mock_mass, "https://ma.example.com")
    await _run_check(webserver, _FakeResponse(200, {"server_id": SERVER_ID}))
    mock_mass.discovery.republish_mass_service.assert_not_awaited()

    cast("MagicMock", webserver.config).get_value.side_effect = lambda key, default=None: {
        CONF_BASE_URL: "https://music.example.com",
    }.get(key, default)
    await _run_check(webserver, _FakeResponse(200, {"server_id": SERVER_ID}))

    assert webserver.base_url == "https://music.example.com"
    mock_mass.discovery.republish_mass_service.assert_awaited_once()


@pytest.mark.parametrize(
    "value",
    ["192.168.1.5:8095", "ftp://192.168.1.5", "http://", "not a url", ""],
)
def test_a_base_url_that_is_not_an_absolute_http_url_is_rejected(
    mock_mass: MagicMock, value: str
) -> None:
    """A value that can never address the webserver is refused at the point it is entered."""
    webserver = _make_webserver(mock_mass, CONF_VALUE_AUTO)

    with pytest.raises(InvalidDataError) as err:
        webserver._validate_base_url(value)

    assert err.value.translation_key == "invalid_base_url"
    assert err.value.translation_owner == "core.webserver"
    assert err.value.translation_args == [value]


@pytest.mark.parametrize(
    "value",
    [CONF_VALUE_AUTO, "http://192.168.1.5:8095", "https://ma.example.com", "http://[fd00::5]:8095"],
)
def test_a_usable_base_url_is_accepted(mock_mass: MagicMock, value: str) -> None:
    """Both the auto default and a full URL pass validation."""
    webserver = _make_webserver(mock_mass, CONF_VALUE_AUTO)

    webserver._validate_base_url(value)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://192.168.1.5", True),
        ("http://192.168.1.5:8095", True),
        ("http://[fd00::5]:8095", True),
        ("http://127.0.0.1:8095", True),
        ("http://[::1]:8095", True),
        # a hostname resolves differently here than it may for a client on the network
        ("https://ma.example.com", False),
        # an address this host does not hold says nothing about what the network can reach
        ("http://192.168.1.9:8095", False),
        ("not a url", False),
    ],
)
def test_points_at_this_host(url: str, expected: bool) -> None:
    """Only a literal address of this host makes a failed probe conclusive."""
    assert _points_at_this_host(url, HOST_ADDRESSES) is expected
