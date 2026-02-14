"""
Runtime configuration validation for the trading bot.

Validates bot_config.json values at startup to catch misconfiguration
before any money is at risk.
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# (key, expected_type, min_value, max_value, description)
_NUMERIC_RULES: List[Tuple[str, type, float, float, str]] = [
    ("MT5_PORT", int, 1, 65535, "MT5 Bridge port"),
    ("MAGIC", int, 1, None, "Magic number"),
    ("MIN_ACCEPTED", (int, float), 0.0, 1.0, "Minimum adjusted probability threshold"),
    ("STD_FACTOR", (int, float), 0.0, 10.0, "Uncertainty penalty factor"),
    ("MC_DROPOUT_SAMPLES", int, 1, 500, "MC Dropout samples"),
    ("WINDOW_SIZE", int, 10, 500, "Look-back window size"),
    ("RISK_PER_TRADE_PCT", (int, float), 0.01, 5.0, "Risk per trade (%)"),
    ("DEFAULT_LOT_SIZE", (int, float), 0.001, 10.0, "Default lot size"),
    ("MAX_LOT_SIZE", (int, float), 0.01, 100.0, "Maximum lot size"),
    ("ATR_SL_MULTIPLIER", (int, float), 0.5, 10.0, "ATR stop-loss multiplier"),
    ("ATR_TP_MULTIPLIER", (int, float), 0.5, 20.0, "ATR take-profit multiplier"),
    ("MAX_DRAWDOWN_PCT", (int, float), 1.0, 50.0, "Max drawdown (%)"),
    ("CORRELATION_THRESHOLD", (int, float), 0.0, 1.0, "Correlation filter threshold"),
    ("CORRELATION_LOOKBACK", int, 10, 500, "Correlation lookback days"),
    ("BREAKEVEN_AFTER_R", (int, float), 0.1, 10.0, "Breakeven after N×R"),
    ("TRAILING_ATR_MULTIPLIER", (int, float), 0.1, 10.0, "Trailing stop ATR multiplier"),
    ("MAX_SPREAD_ATR_RATIO", (int, float), 0.0, 1.0, "Max spread/ATR ratio"),
    ("FRIDAY_CLOSE_HOUR_UTC", int, 0, 23, "Friday close hour (UTC)"),
    ("ATR_PERCENTILE_LOW", (int, float), 0, 50, "ATR percentile low filter"),
    ("ATR_PERCENTILE_HIGH", (int, float), 50, 100, "ATR percentile high filter"),
    ("ATR_PERCENTILE_LOOKBACK", int, 20, 1000, "ATR percentile lookback"),
    ("RETRAIN_INTERVAL_DAYS", int, 1, 365, "Retrain interval (days)"),
    ("CHECK_INTERVAL_SECONDS", int, 10, 86400, "Cycle check interval (seconds)"),
    ("DAILY_REFRESH_HOUR", int, 0, 23, "Daily refresh hour (UTC)"),
    ("POSITION_REVIEW_HOURS", (int, float), 0, 720, "Position review hours"),
]

_BOOLEAN_KEYS = [
    "TRAILING_STOP_ENABLED",
    "CLOSE_BEFORE_WEEKEND",
    "VOLATILITY_FILTER_ENABLED",
    "RETRAIN_ENABLED",
    "NOTIFY_ON_TRADE",
    "NOTIFY_ON_DRAWDOWN",
    "NOTIFY_ON_ERROR",
]

_STRING_KEYS = [
    "MT5_HOST",
    "MODEL_PATH",
    "SCALER_PATH",
    "FEATURES_PATH",
    "SYMBOLS_PATH",
    "STATE_FILE",
]

_REQUIRED_KEYS = [
    "MT5_HOST", "MT5_PORT", "MAGIC",
    "MODEL_PATH", "SCALER_PATH", "FEATURES_PATH", "SYMBOLS_PATH",
    "MIN_ACCEPTED", "WINDOW_SIZE", "MC_DROPOUT_SAMPLES",
    "RISK_PER_TRADE_PCT",
    "ATR_SL_MULTIPLIER", "ATR_TP_MULTIPLIER", "MAX_DRAWDOWN_PCT",
]


def validate_config(config: Dict) -> List[str]:
    """
    Validate bot configuration and return a list of error messages.
    Returns an empty list if all checks pass.
    """
    errors: List[str] = []

    # Check required keys exist
    for key in _REQUIRED_KEYS:
        if key not in config:
            errors.append(f"Missing required config key: {key}")

    # Validate numeric ranges
    for key, expected_type, min_val, max_val, desc in _NUMERIC_RULES:
        if key not in config:
            continue
        val = config[key]
        if not isinstance(val, expected_type):
            errors.append(f"{key} ({desc}): expected {expected_type.__name__ if isinstance(expected_type, type) else 'number'}, got {type(val).__name__} = {val!r}")
            continue
        if min_val is not None and val < min_val:
            errors.append(f"{key} ({desc}): {val} < minimum {min_val}")
        if max_val is not None and val > max_val:
            errors.append(f"{key} ({desc}): {val} > maximum {max_val}")

    # Validate boolean keys
    for key in _BOOLEAN_KEYS:
        if key in config and not isinstance(config[key], bool):
            errors.append(f"{key}: expected bool, got {type(config[key]).__name__} = {config[key]!r}")

    # Validate string keys
    for key in _STRING_KEYS:
        if key in config and not isinstance(config[key], str):
            errors.append(f"{key}: expected str, got {type(config[key]).__name__} = {config[key]!r}")

    # Cross-field validations
    if config.get("ATR_SL_MULTIPLIER", 0) >= config.get("ATR_TP_MULTIPLIER", 0):
        errors.append(
            f"ATR_SL_MULTIPLIER ({config.get('ATR_SL_MULTIPLIER')}) should be < "
            f"ATR_TP_MULTIPLIER ({config.get('ATR_TP_MULTIPLIER')}) for positive reward/risk"
        )

    if config.get("ATR_PERCENTILE_LOW", 0) >= config.get("ATR_PERCENTILE_HIGH", 100):
        errors.append(
            f"ATR_PERCENTILE_LOW ({config.get('ATR_PERCENTILE_LOW')}) must be < "
            f"ATR_PERCENTILE_HIGH ({config.get('ATR_PERCENTILE_HIGH')})"
        )

    # Validate HORIZON_WEIGHTS if present
    hw = config.get("HORIZON_WEIGHTS")
    if hw is not None:
        if not isinstance(hw, dict):
            errors.append(f"HORIZON_WEIGHTS: expected dict, got {type(hw).__name__}")
        else:
            for k, v in hw.items():
                if not isinstance(v, (int, float)) or v < 0:
                    errors.append(f"HORIZON_WEIGHTS[{k}]: expected positive number, got {v!r}")

    # Validate per-instrument-type slot allocation
    slots = config.get("SLOTS_PER_TYPE")
    if slots is not None:
        if not isinstance(slots, dict):
            errors.append(f"SLOTS_PER_TYPE: expected dict, got {type(slots).__name__}")
        else:
            for k, v in slots.items():
                if not isinstance(v, int) or v < 0:
                    errors.append(f"SLOTS_PER_TYPE[{k}]: expected non-negative int, got {v!r}")

    risk_per_type = config.get("RISK_PER_TYPE")
    if risk_per_type is not None:
        if not isinstance(risk_per_type, dict):
            errors.append(f"RISK_PER_TYPE: expected dict, got {type(risk_per_type).__name__}")
        else:
            for k, v in risk_per_type.items():
                if not isinstance(v, (int, float)) or v <= 0 or v > 5.0:
                    errors.append(f"RISK_PER_TYPE[{k}]: expected 0 < value <= 5.0, got {v!r}")

    gms = config.get("GLOBAL_MAX_SLOTS")
    if gms is not None:
        if not isinstance(gms, int) or gms < 1 or gms > 20:
            errors.append(f"GLOBAL_MAX_SLOTS: expected int 1-20, got {gms!r}")

    gmr = config.get("GLOBAL_MAX_RISK_PCT")
    if gmr is not None:
        if not isinstance(gmr, (int, float)) or gmr <= 0 or gmr > 25.0:
            errors.append(f"GLOBAL_MAX_RISK_PCT: expected 0 < value <= 25.0, got {gmr!r}")

    return errors


def validate_config_or_die(config: Dict) -> None:
    """Validate config and raise ValueError if any errors found."""
    errors = validate_config(config)
    if errors:
        msg = "Bot configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(msg)
        raise ValueError(msg)
    logger.info("Configuration validation passed")
