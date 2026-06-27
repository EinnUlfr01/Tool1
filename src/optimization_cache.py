from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.adjusted_predictor import AFTER_ALL_ROLE_MULTIPLIERS
from src.conditional_predictor import DEFAULT_COOLDOWN_CONFIG
from src.conditional_predictor import RULE_DISPLAY_EPSILON
from src.data_loader import DATA_PATH
from src.parameter_optimizer import (
    COOLDOWN_CANDIDATES,
    PRIOR_STRENGTH_CANDIDATES,
    OptimizationResult,
)
from src.seasonal_rules import (
    OUT_OF_SEASON_PENALTY,
    SEASONAL_ALLOWED_MONTHS,
    SEASONAL_COMPONENT_MONTHS,
    SEASONAL_MONTH_BOOST,
)


CACHE_PATH = Path(".cache/optimization_result.json")
LOGIC_VERSION = "primary3-seasonal-eligibility-v1"


@dataclass
class OptimizationCacheState:
    status: str
    cache: dict[str, Any] | None
    reason: str

    @property
    def is_valid(self) -> bool:
        return self.status == "cached" and self.cache is not None


def load_optimization_cache(
    csv_path: str | Path = DATA_PATH,
    cache_path: str | Path = CACHE_PATH,
) -> OptimizationCacheState:
    cache_file = Path(cache_path)
    expected_metadata = build_cache_metadata(csv_path)

    if not cache_file.exists():
        return OptimizationCacheState("default", None, "Optimization cache is missing.")

    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return OptimizationCacheState("stale", None, f"Optimization cache is unreadable: {exc}")

    for key, expected_value in expected_metadata.items():
        if cache.get(key) != expected_value:
            return OptimizationCacheState(
                "stale",
                cache,
                f"Optimization cache is stale because {key} changed.",
            )

    best_params = cache.get("best_params", {})
    if "prior_strength" not in best_params or "cooldown_config" not in best_params:
        return OptimizationCacheState(
            "stale",
            cache,
            "Optimization cache is missing best_params.",
        )

    return OptimizationCacheState("cached", cache, "Using cached optimization params.")


def save_optimization_cache(
    optimization_result: OptimizationResult,
    csv_path: str | Path = DATA_PATH,
    cache_path: str | Path = CACHE_PATH,
) -> dict[str, Any]:
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    cache = {
        **build_cache_metadata(csv_path),
        "best_params": {
            "prior_strength": optimization_result.best_prior_strength,
            "cooldown_config": optimization_result.best_cooldown_config,
        },
        "score": optimization_result.best_final_backtest_score,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    cache_file.write_text(
        json.dumps(cache, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return cache


def get_cached_best_params(cache: dict[str, Any]) -> tuple[float, dict[str, float]]:
    best_params = cache.get("best_params", {})
    prior_strength = float(best_params.get("prior_strength", 3))
    cooldown_config = best_params.get("cooldown_config", DEFAULT_COOLDOWN_CONFIG)
    return prior_strength, dict(cooldown_config)


def build_cache_metadata(csv_path: str | Path = DATA_PATH) -> dict[str, str]:
    return {
        "csv_hash": hash_file(Path(csv_path)),
        "rules_hash": hash_json(build_rules_payload()),
        "logic_version": LOGIC_VERSION,
        "search_space_hash": hash_json(build_search_space_payload()),
    }


def build_rules_payload() -> dict[str, Any]:
    return {
        "after_all_role_multipliers": AFTER_ALL_ROLE_MULTIPLIERS,
        "default_cooldown_config": DEFAULT_COOLDOWN_CONFIG,
        "seasonal_allowed_months": {
            key: sorted(value) for key, value in SEASONAL_ALLOWED_MONTHS.items()
        },
        "seasonal_component_months": {
            key: sorted(value) for key, value in SEASONAL_COMPONENT_MONTHS.items()
        },
        "seasonal_month_boost": SEASONAL_MONTH_BOOST,
        "out_of_season_penalty": OUT_OF_SEASON_PENALTY,
        "rule_display_epsilon": RULE_DISPLAY_EPSILON,
    }


def build_search_space_payload() -> dict[str, Any]:
    return {
        "prior_strength_candidates": PRIOR_STRENGTH_CANDIDATES,
        "cooldown_candidates": COOLDOWN_CANDIDATES,
    }


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(payload: dict[str, Any]) -> str:
    raw_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
