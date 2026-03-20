from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional

from .pipeline_models import PredictionUserContext

DB_PATH = Path(__file__).resolve().parent / "predictions.db"
_DB_LOCK = Lock()


def _ensure_db() -> None:
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


def persist_prediction(context: PredictionUserContext, payload: Dict[str, object]) -> None:
    """Store a user's prediction snapshot and seed it with any final score we already know."""

    game_id = payload.get("game_id")
    if not game_id:
        return

    final_home_score: Optional[int] = None
    final_away_score: Optional[int] = None
    with _conn_context() as conn:
        row = conn.execute(
            "SELECT home_score, away_score FROM game_scores WHERE game_id = ?",
            (game_id,),
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
        for entry in entries:
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
                """,
                (
                    entry.get("home_score"),
                    entry.get("away_score"),
                    entry.get("updated_at") or datetime.now(timezone.utc).isoformat(),
                    entry["game_id"],
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
                gs.status AS game_status
            FROM user_predictions up
            LEFT JOIN game_scores gs ON up.game_id = gs.game_id
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
            "final_home_score": row["final_home_score"],
            "final_away_score": row["final_away_score"],
            "game_status": row["game_status"],
        }
        result.append(entry)

    return result


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
