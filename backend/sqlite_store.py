# ==========================================
# File: backend/sqlite_store.py
# Role: SQLite persistence for game scores and user predictions.
# Input Data: Score entries, Prediction records.
# Output Data: SQL query results, DB updates.
# Dependencies: sqlite3, pathlib, threading
# Notes: Low-level DB access; ensures schema consistency and thread-safe writes.
# ==========================================

from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional

from .pipeline_models import PredictionUserContext
# Comment 1: Import core score sync utilities from backend/scripts/score_sync.py after reorganization.
from backend.scripts.score_sync import build_score_game_id, normalize_team_code

DB_PATH = Path(__file__).resolve().parent / "predictions.db"
_DB_LOCK = Lock()


def _ensure_db() -> None:
    # Comment 2: Guarantee that local SQLite database and required tables (game_scores, user_predictions) exist.
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS game_scores (
                game_id TEXT PRIMARY KEY,
                season INTEGER,
                week INTEGER,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                status TEXT,
                updated_at TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                storage_key TEXT,
                game_id TEXT,
                season INTEGER,
                week INTEGER,
                home_team TEXT,
                away_team TEXT,
                home_win_probability REAL,
                away_win_probability REAL,
                predicted_home_score REAL,
                predicted_away_score REAL,
                confidence REAL,
                prediction_source TEXT,
                win_classifier_used INTEGER DEFAULT 0,
                ts TEXT,
                final_home_score INTEGER,
                final_away_score INTEGER,
                updated_at TEXT,
                FOREIGN KEY(game_id) REFERENCES game_scores(game_id)
            );
            """
        )
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(user_predictions)").fetchall()
        }
        if "win_classifier_used" not in columns:
            conn.execute(
                "ALTER TABLE user_predictions ADD COLUMN win_classifier_used INTEGER DEFAULT 0"
            )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_predictions_storage_key ON user_predictions(storage_key);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_predictions_game_id ON user_predictions(game_id);")


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _conn_context():
    _ensure_db()
    with _get_connection() as conn:
        yield conn


def _to_int_or_none(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _normalize_score_entry(raw_entry: Dict[str, object]) -> Optional[Dict[str, object]]:
    season = _to_int_or_none(raw_entry.get("season"))
    week = _to_int_or_none(raw_entry.get("week"))
    home_score = _to_int_or_none(raw_entry.get("home_score"))
    away_score = _to_int_or_none(raw_entry.get("away_score"))
    home_team = normalize_team_code(raw_entry.get("home_team"))
    away_team = normalize_team_code(raw_entry.get("away_team"))

    if None in {season, week, home_score, away_score} or not home_team or not away_team:
        return None

    updated_at = raw_entry.get("updated_at") or datetime.now(timezone.utc).isoformat()
    return {
        "game_id": build_score_game_id(season, week, home_team, away_team),
        "season": season,
        "week": week,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "status": raw_entry.get("status") or "final",
        "updated_at": updated_at,
    }


def persist_prediction(context: PredictionUserContext, payload: Dict[str, object]) -> None:
    """Store a user's prediction snapshot and seed it with any final score we already know."""
    # Comment 3: Deduplicate user prediction record based on game identifier and commit to database.
    game_id = payload.get("game_id")
    if not game_id:
        return

    final_home_score: Optional[int] = None
    final_away_score: Optional[int] = None
    season = payload.get("season")
    week = payload.get("week")
    home_team = normalize_team_code(payload.get("home_team"))
    away_team = normalize_team_code(payload.get("away_team"))
    canonical_game_id = build_score_game_id(season, week, home_team, away_team) or game_id
    with _conn_context() as conn:
        row = conn.execute(
            """
            SELECT home_score, away_score
            FROM game_scores
            WHERE game_id = ?
               OR (season = ? AND week = ? AND home_team = ? AND away_team = ?)
            ORDER BY CASE WHEN game_id = ? THEN 0 ELSE 1 END
            LIMIT 1
            """,
            (canonical_game_id, season, week, home_team, away_team, canonical_game_id),
        ).fetchone()
        if row:
            final_home_score = row["home_score"]
            final_away_score = row["away_score"]

        ts = payload.get("ts") or datetime.now(timezone.utc).isoformat()
        conn.execute(
            "DELETE FROM user_predictions WHERE storage_key = ? AND game_id = ?",
            (context.storage_key, game_id),
        )
        conn.execute(
            """
            INSERT INTO user_predictions (
                user_id,
                storage_key,
                game_id,
                season,
                week,
                home_team,
                away_team,
                home_win_probability,
                away_win_probability,
                predicted_home_score,
                predicted_away_score,
                confidence,
                prediction_source,
                win_classifier_used,
                ts,
                final_home_score,
                final_away_score,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                context.user_id,
                context.storage_key,
                payload["game_id"],
                payload.get("season"),
                payload.get("week"),
                payload.get("home_team"),
                payload.get("away_team"),
                payload.get("home_win_probability"),
                payload.get("away_win_probability"),
                payload.get("home_score"),
                payload.get("away_score"),
                max(payload.get("home_win_probability", 0), payload.get("away_win_probability", 0)),
                payload.get("prediction_source"),
                1 if bool(payload.get("win_classifier_used")) else 0,
                ts,
                final_home_score,
                final_away_score,
                ts,
            ),
        )


def upsert_game_scores(entries: Iterable[Dict[str, object]]) -> None:
    """Bulk insert/update game results and refresh dependent predictions."""

    with _conn_context() as conn:
        for raw_entry in entries:
            if not isinstance(raw_entry, dict):
                continue
            entry = _normalize_score_entry(raw_entry)
            if entry is None:
                continue
            conn.execute(
                """
                INSERT INTO game_scores (
                    game_id,
                    season,
                    week,
                    home_team,
                    away_team,
                    home_score,
                    away_score,
                    status,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    home_score = excluded.home_score,
                    away_score = excluded.away_score,
                    status = excluded.status,
                    updated_at = excluded.updated_at;
                """,
                (
                    entry["game_id"],
                    entry.get("season"),
                    entry.get("week"),
                    entry.get("home_team"),
                    entry.get("away_team"),
                    entry.get("home_score"),
                    entry.get("away_score"),
                    entry.get("status"),
                    entry.get("updated_at") or datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.execute(
                """
                UPDATE user_predictions
                SET final_home_score = ?,
                    final_away_score = ?,
                    updated_at = ?
                WHERE game_id = ?
                   OR (
                        season = ?
                    AND week = ?
                    AND home_team = ?
                    AND away_team = ?
                   )
                """,
                (
                    entry.get("home_score"),
                    entry.get("away_score"),
                    entry.get("updated_at") or datetime.now(timezone.utc).isoformat(),
                    entry["game_id"],
                    entry.get("season"),
                    entry.get("week"),
                    entry.get("home_team"),
                    entry.get("away_team"),
                ),
            )


def get_user_history(
    context: PredictionUserContext, limit: int = 100
) -> List[Dict[str, object]]:
    """Return the most recent predictions, augmented with actual scores if available."""

    bounded = max(1, min(limit or 100, 1000))
    with _conn_context() as conn:
        rows = conn.execute(
            """
            SELECT
                up.*,
                gs.home_score AS actual_home_score,
                gs.away_score AS actual_away_score,
                gs.status AS game_status,
                gs.updated_at AS score_updated_at
            FROM user_predictions up
            LEFT JOIN game_scores gs
              ON up.game_id = gs.game_id
              OR (
                    up.season = gs.season
                AND up.week = gs.week
                AND up.home_team = gs.home_team
                AND up.away_team = gs.away_team
              )
            WHERE up.storage_key = ?
            ORDER BY up.ts DESC
            LIMIT ?
            """,
            (context.storage_key, bounded),
        ).fetchall()

    result: List[Dict[str, object]] = []
    for row in rows:
        predicted_home = row["predicted_home_score"]
        predicted_away = row["predicted_away_score"]
        point_diff = (
            (predicted_home or 0) - (predicted_away or 0)
            if predicted_home is not None and predicted_away is not None
            else None
        )
        entry = {
            "user_id": row["user_id"],
            "storage_key": row["storage_key"],
            "ts": row["ts"],
            "game_id": row["game_id"],
            "season": row["season"],
            "week": row["week"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_score": predicted_home,
            "away_score": predicted_away,
            "point_diff": point_diff,
            "home_win_probability": row["home_win_probability"],
            "away_win_probability": row["away_win_probability"],
            "prediction_source": row["prediction_source"],
            "win_classifier_used": bool(row["win_classifier_used"]) if row["win_classifier_used"] is not None else False,
            "final_home_score": row["final_home_score"] if row["final_home_score"] is not None else row["actual_home_score"],
            "final_away_score": row["final_away_score"] if row["final_away_score"] is not None else row["actual_away_score"],
            "game_status": row["game_status"],
            "score_updated_at": row["score_updated_at"],
        }
        result.append(entry)

    return result


def get_user_history_count(context: PredictionUserContext) -> int:
    """Return the total number of predictions stored for one user."""

    with _conn_context() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total FROM user_predictions WHERE storage_key = ?",
            (context.storage_key,),
        ).fetchone()

    return int(row["total"]) if row and row["total"] is not None else 0


def clear_user_history(context: PredictionUserContext) -> int:
    """Delete persisted prediction rows for one user and return the row count."""

    with _conn_context() as conn:
        before = conn.execute(
            "SELECT COUNT(*) AS total FROM user_predictions WHERE storage_key = ?",
            (context.storage_key,),
        ).fetchone()
        conn.execute(
            "DELETE FROM user_predictions WHERE storage_key = ?",
            (context.storage_key,),
        )

    return int(before["total"]) if before and before["total"] is not None else 0


def get_user_history_summary(context: PredictionUserContext) -> Dict[str, object]:
    """Return aggregate accuracy and freshness metrics for one user's predictions."""

    with _conn_context() as conn:
        rows = conn.execute(
            """
            SELECT
                up.ts,
                up.home_win_probability,
                up.away_win_probability,
                up.predicted_home_score,
                up.predicted_away_score,
                COALESCE(up.final_home_score, gs.home_score) AS final_home_score,
                COALESCE(up.final_away_score, gs.away_score) AS final_away_score,
                gs.updated_at AS score_updated_at
            FROM user_predictions up
            LEFT JOIN game_scores gs
              ON up.game_id = gs.game_id
              OR (
                    up.season = gs.season
                AND up.week = gs.week
                AND up.home_team = gs.home_team
                AND up.away_team = gs.away_team
              )
            WHERE up.storage_key = ?
            ORDER BY up.ts DESC
            """,
            (context.storage_key,),
        ).fetchall()

    total_predictions = len(rows)
    latest_prediction_at: Optional[str] = None
    last_score_sync_at: Optional[str] = None
    resolved_games = 0
    correct_predictions = 0
    spread_errors: List[float] = []
    confidences: List[float] = []

    for row in rows:
        ts = row["ts"]
        if latest_prediction_at is None and ts:
            latest_prediction_at = str(ts)

        score_updated_at = row["score_updated_at"]
        if score_updated_at and (last_score_sync_at is None or str(score_updated_at) > last_score_sync_at):
            last_score_sync_at = str(score_updated_at)

        home_prob = row["home_win_probability"]
        away_prob = row["away_win_probability"]
        if home_prob is not None or away_prob is not None:
            confidences.append(float(max(home_prob or 0.0, away_prob or 0.0)))

        actual_home = row["final_home_score"]
        actual_away = row["final_away_score"]
        predicted_home = row["predicted_home_score"]
        predicted_away = row["predicted_away_score"]
        if actual_home is None or actual_away is None:
            continue

        resolved_games += 1

        predicted_home_wins = float(home_prob or 0.0) >= float(away_prob or 0.0)
        actual_home_wins = int(actual_home) > int(actual_away)
        if predicted_home_wins == actual_home_wins:
            correct_predictions += 1

        if predicted_home is not None and predicted_away is not None:
            predicted_diff = float(predicted_home) - float(predicted_away)
            actual_diff = float(actual_home) - float(actual_away)
            spread_errors.append(abs(predicted_diff - actual_diff))

    win_rate = (correct_predictions / resolved_games) if resolved_games > 0 else None
    avg_abs_spread_error = (
        sum(spread_errors) / len(spread_errors) if spread_errors else None
    )
    avg_confidence = (sum(confidences) / len(confidences)) if confidences else None

    return {
        "total_predictions": total_predictions,
        "resolved_games": resolved_games,
        "win_rate": win_rate,
        "avg_abs_spread_error": avg_abs_spread_error,
        "avg_confidence": avg_confidence,
        "latest_prediction_at": latest_prediction_at,
        "last_score_sync_at": last_score_sync_at,
    }


def get_game_scores(season: Optional[int] = None, week: Optional[int] = None) -> List[Dict[str, object]]:
    """Return scoreboard results with optional season/week filtering."""

    with _conn_context() as conn:
        query = "SELECT * FROM game_scores"
        params: list[object] = []
        clauses: list[str] = []
        if season is not None:
            clauses.append("season = ?")
            params.append(season)
        if week is not None:
            clauses.append("week = ?")
            params.append(week)
        if clauses:
            query = f"{query} WHERE {' AND '.join(clauses)}"
        query = f"{query} ORDER BY season DESC, week DESC"
        rows = conn.execute(query, params).fetchall()

    results: List[Dict[str, object]] = []
    for row in rows:
        results.append(
            {
                "game_id": row["game_id"],
                "season": row["season"],
                "week": row["week"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": row["home_score"],
                "away_score": row["away_score"],
                "status": row["status"],
                "updated_at": row["updated_at"],
            }
        )
    return results
