"""Explicit provider restrictions retained on each queue item."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING

from music_assistant_models.errors import MediaNotFoundError

if TYPE_CHECKING:
    from music_assistant_models.queue_item import QueueItem

ALLOWED_PROVIDERS_ATTRIBUTE = "allowed_provider_instances"


def allowed_providers(queue_item: QueueItem) -> set[str] | None:
    """Return the item's explicit restriction, or None for automatic selection."""
    if ALLOWED_PROVIDERS_ATTRIBUTE not in queue_item.extra_attributes:
        return None
    value = queue_item.extra_attributes[ALLOWED_PROVIDERS_ATTRIBUTE]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as err:
            raise MediaNotFoundError("The queue item's provider restriction is invalid.") from err
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x for x in value):
        raise MediaNotFoundError("The queue item's provider restriction is invalid.")
    return set(value)


def restrict_queue_item(queue_item: QueueItem, instances: list[str]) -> None:
    """
    Restrict a queue item without modifying the library's shared media object.

    :param queue_item: Newly resolved queue item.
    :param instances: Exact allowed provider instance identifiers.
    """
    if not instances or queue_item.media_item is None:
        raise MediaNotFoundError("Choose at least one provider for playback.")
    mappings = {
        mapping
        for mapping in queue_item.media_item.provider_mappings
        if mapping.available and mapping.provider_instance in instances
    }
    if not mappings:
        raise MediaNotFoundError(
            f"{queue_item.name} is not available from the selected provider(s). "
            "No other service was substituted."
        )
    queue_item.media_item = deepcopy(queue_item.media_item)
    queue_item.media_item.provider_mappings = mappings
    queue_item.extra_attributes[ALLOWED_PROVIDERS_ATTRIBUTE] = json.dumps(sorted(set(instances)))
