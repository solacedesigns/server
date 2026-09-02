"""Setup flow for the Listening Habits provider."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType

from music_assistant.models.setup_flow import SetupFlowError
from music_assistant.providers.listening_habits import (
    CONF_ENDPOINT,
    CONF_HOME_PLACE,
    CONF_TOKEN,
    CONF_WEATHER_ENTITY,
    PUSH_TIMEOUT_S,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType

    from music_assistant.models.setup_flow import SetupSession

_ENTRIES = (
    ConfigEntry(key=CONF_ENDPOINT, type=ConfigEntryType.STRING, required=True),
    ConfigEntry(key=CONF_TOKEN, type=ConfigEntryType.SECURE_STRING, required=True),
    ConfigEntry(key=CONF_HOME_PLACE, type=ConfigEntryType.STRING, required=False, advanced=True),
    ConfigEntry(key=CONF_WEATHER_ENTITY, type=ConfigEntryType.STRING, required=False, advanced=True),
)


async def _verify_credentials(
    session: SetupSession, setup_data: dict[str, ConfigValueType]
) -> None:
    """
    Reject an unreachable endpoint or a wrong token here, while a human is watching.

    The log server exposes an auth-only probe precisely because a bad token is
    otherwise invisible: plays are refused one by one at 403 long after setup
    said "All set!". Checking it costs one request and turns a silent, ongoing
    data loss into a message on the form that caused it.
    """
    endpoint = str(setup_data.get(CONF_ENDPOINT) or "").strip()
    token = str(setup_data.get(CONF_TOKEN) or "").strip()
    if not endpoint:
        raise SetupFlowError("An ingest endpoint is required.")
    probe = endpoint.rstrip("/") + "/test"
    try:
        async with session.mass.http_session.post(
            probe,
            headers={"Authorization": f"Bearer {token}"},
            timeout=PUSH_TIMEOUT_S,
        ) as response:
            if response.status in (401, 403):
                raise SetupFlowError(
                    "The log server refused this token. It must match LHS_TOKEN on the server."
                )
            if response.status >= 400:
                raise SetupFlowError(f"The log server answered {response.status} for {probe}.")
    except SetupFlowError:
        raise
    except Exception as err:
        raise SetupFlowError(f"Could not reach {probe}: {err}") from err


async def run_setup(session: SetupSession) -> None:
    """Run the setup flow: collect the endpoint and token, then create the provider."""
    errors: dict[str, str] | None = None
    setup_data = dict(session.context.setup_data)
    while True:
        entries = [
            replace(entry, value=setup_data.get(entry.key, entry.value)) for entry in _ENTRIES
        ]
        submitted = await session.form(entries, step_id="user", errors=errors, last_step=True)
        setup_data.update(submitted)
        try:
            await _verify_credentials(session, setup_data)
            await session.finish(setup_data)
            return
        except SetupFlowError as err:
            errors = {"base": err.translation_key or str(err)}
