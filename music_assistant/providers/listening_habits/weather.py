"""Weather snapshots for Listening Habits ingest payloads."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


PRECIPITATION_TERMS = ("rain", "drizzle", "snow", "sleet", "hail", "lightning")

SYMBOLS = {
    "clear-night": "moon.stars.fill",
    "cloudy": "cloud.fill",
    "exceptional": "exclamationmark.triangle.fill",
    "fog": "cloud.fog.fill",
    "hail": "cloud.hail.fill",
    "lightning": "cloud.bolt.fill",
    "lightning-rainy": "cloud.bolt.rain.fill",
    "partlycloudy": "cloud.sun.fill",
    "pouring": "cloud.heavyrain.fill",
    "rainy": "cloud.rain.fill",
    "snowy": "cloud.snow.fill",
    "snowy-rainy": "cloud.sleet.fill",
    "sunny": "sun.max.fill",
    "windy": "wind",
    "windy-variant": "wind",
}


def snapshot_from_hass_state(state: dict[str, Any]) -> dict[str, Any] | None:
    """Translate a Home Assistant weather state to the shared ingest schema."""
    if str(state.get("state") or "").lower() in {"", "unknown", "unavailable"}:
        return None
    attributes = state.get("attributes")
    if not isinstance(attributes, dict):
        return None
    temperature = _number(attributes.get("temperature"))
    if temperature is None:
        return None
    temperature_c = _temperature_c(
        temperature, str(attributes.get("temperature_unit") or "°C")
    )
    apparent = _number(attributes.get("apparent_temperature"))
    if apparent is not None:
        apparent = _temperature_c(apparent, str(attributes.get("temperature_unit") or "°C"))
    condition_key = str(state.get("state") or "").strip().lower()
    condition = condition_key.replace("-", " ").capitalize() or None
    precipitation = (
        condition if any(term in condition_key for term in PRECIPITATION_TERMS) else None
    )
    wind = _number(attributes.get("wind_speed"))
    wind_kph = _speed_kph(wind, str(attributes.get("wind_speed_unit") or "km/h"))
    observed = _timestamp(state.get("last_updated") or state.get("last_changed"))
    return {
        "weather_observed_at": observed,
        "weather_temperature_c": round(temperature_c, 2),
        "weather_apparent_temperature_c": round(apparent, 2) if apparent is not None else None,
        "weather_condition": condition,
        "weather_precipitation": precipitation,
        "weather_symbol": SYMBOLS.get(condition_key, "cloud.fill"),
        "weather_cloud_cover_pct": _integer(attributes.get("cloud_coverage")),
        "weather_wind_kph": round(wind_kph, 2) if wind_kph is not None else None,
    }


def snapshot_from_open_meteo(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Translate an Open-Meteo current observation to the shared ingest schema."""
    current = payload.get("current")
    if not isinstance(current, dict):
        return None
    temperature = _number(current.get("temperature_2m"))
    if temperature is None:
        return None
    code = _integer(current.get("weather_code"))
    condition, precipitation, symbol = _open_meteo_condition(code)
    return {
        "weather_observed_at": _timestamp(current.get("time")),
        "weather_temperature_c": temperature,
        "weather_apparent_temperature_c": _number(current.get("apparent_temperature")),
        "weather_condition": condition,
        "weather_precipitation": precipitation,
        "weather_symbol": symbol,
        "weather_cloud_cover_pct": _integer(current.get("cloud_cover")),
        "weather_wind_kph": _number(current.get("wind_speed_10m")),
    }


def _open_meteo_condition(code: int | None) -> tuple[str, str | None, str]:
    if code == 0:
        return "Sunny", None, "sun.max.fill"
    if code in (1, 2):
        return "Partly cloudy", None, "cloud.sun.fill"
    if code == 3:
        return "Overcast", None, "cloud.fill"
    if code in (45, 48):
        return "Foggy", None, "cloud.fog.fill"
    if code is not None and 51 <= code <= 57:
        return "Drizzle", "Drizzle", "cloud.drizzle.fill"
    if code is not None and 61 <= code <= 67:
        return "Rainy", "Rain", "cloud.rain.fill"
    if code is not None and 71 <= code <= 77:
        return "Snowy", "Snow", "cloud.snow.fill"
    if code is not None and 80 <= code <= 82:
        return "Rain showers", "Rain showers", "cloud.heavyrain.fill"
    if code is not None and 85 <= code <= 86:
        return "Snow showers", "Snow showers", "cloud.snow.fill"
    if code is not None and 95 <= code <= 99:
        return "Thunderstorms", "Thunderstorms", "cloud.bolt.rain.fill"
    return "Unknown", None, "cloud.fill"


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return round(number) if number is not None else None


def _temperature_c(value: float, unit: str) -> float:
    return (value - 32) * 5 / 9 if "F" in unit.upper() else value


def _speed_kph(value: float | None, unit: str) -> float | None:
    if value is None:
        return None
    normalized = unit.lower().replace(" ", "")
    if normalized in {"mph", "mi/h"}:
        return value * 1.609344
    if normalized in {"m/s", "mps"}:
        return value * 3.6
    return value


def _timestamp(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return int(parsed.timestamp())
        except ValueError:
            pass
    return int(datetime.now(UTC).timestamp())
