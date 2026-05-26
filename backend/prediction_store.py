# ==========================================
# File: backend/prediction_store.py
# Role: User-scoped prediction history management.
# Input Data: User IDs, Prediction records.
# Output Data: History lists, User context, Storage keys.
# Dependencies: hashlib, json, pathlib, sqlite_store
# Notes: Bridges the gap between the API and the SQLite/JSON persistence layer.
# ==========================================

"""
Disk-backed, user-scoped prediction history.

The application only has a local session layer today, so this store keys each
prediction ledger by the signed-in frontend identity that arrives in the
`X-User-Id` request header.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Iterable

from .pipeline_models import PredictionStorageProfile, PredictionUserContext
from .schemas import HistoryEntry, HistoryResponse, PredictionRequest, StoredPredictionRecord

from .sqlite_store import (
    clear_user_history,
    get_user_history,
    get_user_history_count,
    get_user_history_summary,
    persist_prediction,
)

log = logging.getLogger(__name__)

PREDICTION_STORE_ROOT = Path(__file__).resolve().parent / "Predictions" / "users"
PREDICTION_HISTORY_MAX = 1000
_prediction_store_lock = Lock()
_USER_ID_SANITIZER = re.compile(r"[^a-z0-9._-]+")


def build_prediction_user_context(user_id: str | None) -> PredictionUserContext:
    """Convert an incoming user id into a stable, filesystem-safe context."""

    raw_user_id = (user_id or "anonymous").strip().lower() or "anonymous"
    slug = _USER_ID_SANITIZER.sub("-", raw_user_id).strip("-.") or "anonymous"
    digest = hashlib.sha1(raw_user_id.encode("utf-8")).hexdigest()[:10]
    return PredictionUserContext(user_id=raw_user_id, storage_key=f"{slug}-{digest}")


def _user_dir(context: PredictionUserContext) -> Path:
    return PREDICTION_STORE_ROOT / context.storage_key


def _user_history_path(context: PredictionUserContext) -> Path:
    return _user_dir(context) / "predictions.json"


def _user_profile_path(context: PredictionUserContext) -> Path:
    return _user_dir(context) / "profile.json"


def _read_history_records(path: Path) -> list[StoredPredictionRecord]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to read prediction history at %s: %s", path, exc)
        return []

    if not isinstance(payload, list):
        return []

    records: list[StoredPredictionRecord] = []
    for item in payload:
        try:
            records.append(StoredPredictionRecord.model_validate(item))
        except Exception as exc:
            log.warning("Skipping invalid stored prediction record: %s", exc)
    return records


def _read_profile(path: Path) -> PredictionStorageProfile | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return PredictionStorageProfile.model_validate(payload)
    except Exception as exc:
        log.warning("Failed to read prediction profile at %s: %s", path, exc)
        return None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _dump_records(records: Iterable[StoredPredictionRecord]) -> list[dict]:
    return [record.model_dump(mode="json") for record in records]


def append_prediction_record(
    context: PredictionUserContext,
    request_payload: PredictionRequest,
    prediction_payload: dict,
) -> StoredPredictionRecord:
    """Persist a validated prediction record for one user."""

    timestamp = datetime.now(timezone.utc).isoformat()
    record = StoredPredictionRecord.model_validate(
        {
            **prediction_payload,
            "ts": timestamp,
            "user_id": context.user_id,
            "storage_key": context.storage_key,
            "request": request_payload.model_dump(),
        }
    )

    history_path = _user_history_path(context)
    profile_path = _user_profile_path(context)

    with _prediction_store_lock:
        existing = _read_history_records(history_path)
        next_records = [record, *existing][:PREDICTION_HISTORY_MAX]
        existing_profile = _read_profile(profile_path)
        total_predictions_all_time = (
            existing_profile.total_predictions_all_time + 1 if existing_profile else len(existing) + 1
        )

        _write_json(history_path, _dump_records(next_records))
        profile = PredictionStorageProfile(
            user_id=context.user_id,
            storage_key=context.storage_key,
            updated_at_utc=timestamp,
            retained_predictions=len(next_records),
            total_predictions_all_time=total_predictions_all_time,
        )
        _write_json(
            profile_path,
            profile.model_dump(mode="json"),
        )

    persist_prediction(context, record.model_dump(mode="json"))
    return record


def get_prediction_history(context: PredictionUserContext, limit: int = 100) -> HistoryResponse:
    """Load one user's prediction history from disk."""

    bounded_limit = max(1, min(int(limit or 100), PREDICTION_HISTORY_MAX))
    history_path = _user_history_path(context)

    sqlite_entries = get_user_history(context, limit=bounded_limit)
    if sqlite_entries:
        entries = [HistoryEntry.model_validate(entry) for entry in sqlite_entries]
        return HistoryResponse(entries=entries, total=len(sqlite_entries), user_id=context.user_id)

    with _prediction_store_lock:
        records = _read_history_records(history_path)

    entries = [HistoryEntry.model_validate(record.model_dump()) for record in records[:bounded_limit]]
    return HistoryResponse(entries=entries, total=len(records), user_id=context.user_id)


def get_prediction_history_count(context: PredictionUserContext) -> int:
    """Count stored predictions for one user without exposing the whole ledger."""

    try:
        return int(get_user_history_count(context))
    except Exception:
        log.exception("Failed to count SQLite-backed prediction history; falling back to JSON.")

    history_path = _user_history_path(context)
    with _prediction_store_lock:
        return len(_read_history_records(history_path))


def clear_prediction_history(context: PredictionUserContext) -> dict[str, object]:
    """Clear one user's SQLite history and JSON fallback ledger."""

    deleted_sqlite = 0
    try:
        deleted_sqlite = clear_user_history(context)
    except Exception:
        log.exception("Failed to clear SQLite-backed prediction history; continuing with JSON fallback.")

    timestamp = datetime.now(timezone.utc).isoformat()
    history_path = _user_history_path(context)
    profile_path = _user_profile_path(context)
    with _prediction_store_lock:
        json_count = len(_read_history_records(history_path))
        _write_json(history_path, [])
        existing_profile = _read_profile(profile_path)
        profile = PredictionStorageProfile(
            user_id=context.user_id,
            storage_key=context.storage_key,
            updated_at_utc=timestamp,
            retained_predictions=0,
            total_predictions_all_time=(
                existing_profile.total_predictions_all_time if existing_profile else 0
            ),
        )
        _write_json(profile_path, profile.model_dump(mode="json"))

    return {
        "deleted": max(deleted_sqlite, json_count),
        "sqlite_deleted": deleted_sqlite,
        "json_deleted": json_count,
        "user_id": context.user_id,
        "storage_key": context.storage_key,
        "cleared_at": timestamp,
    }


def get_prediction_history_summary(context: PredictionUserContext) -> dict[str, object]:
    """Return per-user prediction metrics with SQLite as the primary source."""

    try:
        summary = get_user_history_summary(context)
        if int(summary.get("total_predictions") or 0) > 0:
            return summary
    except Exception:
        log.exception("Failed to summarize SQLite-backed prediction history; falling back to JSON.")

    with _prediction_store_lock:
        records = _read_history_records(_user_history_path(context))

    latest_prediction_at = records[0].ts if records else None
    confidences = []
    for record in records:
        try:
            confidences.append(
                max(float(record.home_win_probability), float(record.away_win_probability))
            )
        except Exception:
            continue

    avg_confidence = (sum(confidences) / len(confidences)) if confidences else None
    return {
        "total_predictions": len(records),
        "resolved_games": 0,
        "win_rate": None,
        "avg_abs_spread_error": None,
        "avg_confidence": avg_confidence,
        "latest_prediction_at": latest_prediction_at,
        "last_score_sync_at": None,
    }
