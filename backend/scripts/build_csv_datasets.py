#!/usr/bin/env python3
"""
File: backend/scripts/build_csv_datasets.py
Purpose: Production entrypoint for dataset builds with validation and versioning outputs.

This script wraps the canonical builder in `backend/build_csv_datasetsv3.py` and adds:
  1) deterministic output naming + latest alias
  2) schema validation and explicit failure exit codes
  3) run metadata + dataset versioning report artifacts
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Ensure both repo root and backend root are importable no matter where CLI is launched.
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import backend.build_csv_datasetsv3 as builder_v3  # noqa: E402
from backend.utils.ops_reporting import file_sha256, write_dataset_version_report  # noqa: E402


REQUIRED_COLUMNS: List[str] = [
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "home_points_for",
    "away_points_for",
    "home_win",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build NFL game_features dataset with production-grade validation and reporting."
    )
    parser.add_argument("--start", type=int, default=2018, help="Start season (inclusive).")
    parser.add_argument("--end", type=int, default=2026, help="End season (inclusive).")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str((BACKEND_DIR / "data").resolve()),
        help="Output directory for dataset CSV files.",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str((BACKEND_DIR / "reports").resolve()),
        help="Report root directory for dataset versioning outputs.",
    )
    parser.add_argument(
        "--encode",
        choices=["onehot", "none"],
        default="onehot",
        help="Team encoding mode passed through to v3 builder.",
    )
    parser.add_argument(
        "--save-dominance-matrix",
        action="store_true",
        help="Also export dominance_matrix.csv.",
    )
    parser.add_argument(
        "--no-calibration-rows",
        action="store_true",
        help="Do not append blank calibration rows.",
    )
    parser.add_argument(
        "--dominance-log",
        type=str,
        default=None,
        help="Optional path for pairwise dominance text output.",
    )
    parser.add_argument(
        "--legacy-root-copy",
        action="store_true",
        help="Also write a legacy root-level copy for backwards compatibility.",
    )
    parser.add_argument(
        "--enable-heavy-stats",
        action="store_true",
        help=(
            "Enable expensive PBP/team/player stat loaders from v3 builder. "
            "Disabled by default for reliability in constrained environments."
        ),
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail if expected training columns are missing from output.",
    )
    parser.add_argument(
        "--skip-version-report",
        action="store_true",
        help="Skip generating backend/reports/versioning outputs.",
    )
    return parser.parse_args()


def _setup_logging(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "build_csv_datasets_entry.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


def _validate_output(df: pd.DataFrame, strict_validation: bool) -> None:
    if df.empty:
        raise RuntimeError("Dataset builder returned an empty DataFrame.")

    if strict_validation:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"Strict validation failed. Missing required dataset columns: {missing}"
            )


def _disable_heavy_loaders() -> None:
    """Prevent expensive loaders from crashing builds on constrained hosts."""

    def _empty_team_game_metrics(_: Path) -> pd.DataFrame:
        return pd.DataFrame(columns=["season", "week", "game_id", "team"])

    def _empty_team_weekly_stats(_: List[int]) -> pd.DataFrame:
        return pd.DataFrame(columns=["season", "week", "team"])

    def _empty_player_stats(_: List[int]) -> pd.DataFrame:
        return pd.DataFrame(columns=["season", "week", "game_id", "team"])

    builder_v3.load_team_game_metrics = _empty_team_game_metrics
    builder_v3.load_team_weekly_stats = _empty_team_weekly_stats
    builder_v3.load_player_game_stats = _empty_player_stats


def _read_csv_with_conflict_cleanup(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("<<<<<<<") or stripped.startswith("=======") or stripped.startswith(">>>>>>>"):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    if not cleaned:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(cleaned))


def _load_schedule_from_local_csv(seasons: List[int], include_future: bool) -> pd.DataFrame:
    candidates = [
        REPO_ROOT / "NFL_Schedule.csv",
        BACKEND_DIR / "data" / "Nfl_schedule_2025.csv",
        BACKEND_DIR / "data" / "Nfl_schedule_2025_2026.csv",
        REPO_ROOT / "frontend" / "public" / "nflSchedule.csv",
    ]
    source: Optional[Path] = None
    sch = pd.DataFrame()
    for path in candidates:
        if not path.exists():
            continue
        try:
            sch = _read_csv_with_conflict_cleanup(path)
            if not sch.empty:
                source = path
                break
        except Exception:
            continue

    if sch.empty or source is None:
        raise RuntimeError("Could not load schedules from backend or local CSV fallbacks.")

    logging.warning("Falling back to local schedule CSV: %s", source)

    if "gameday" in sch.columns and "game_date" not in sch.columns:
        sch = sch.rename(columns={"gameday": "game_date"})
    if "game_date" not in sch.columns:
        sch["game_date"] = None

    for col in ("season", "week", "home_team", "away_team"):
        if col not in sch.columns:
            raise RuntimeError(f"Fallback schedule missing required column: {col}")

    if "game_id" not in sch.columns:
        sch["game_id"] = (
            sch["season"].astype(str)
            + "_"
            + sch["week"].astype(str).str.zfill(2)
            + "_"
            + sch["away_team"].astype(str)
            + "_"
            + sch["home_team"].astype(str)
        )

    # Ensure expected modeling context columns exist.
    for col, default in (
        ("home_score", np.nan),
        ("away_score", np.nan),
        ("game_type", "REG"),
        ("away_moneyline", np.nan),
        ("home_moneyline", np.nan),
        ("spread_line", np.nan),
        ("total_line", np.nan),
        ("away_rest", np.nan),
        ("home_rest", np.nan),
    ):
        if col not in sch.columns:
            sch[col] = default

    sch["season"] = pd.to_numeric(sch["season"], errors="coerce")
    sch["week"] = pd.to_numeric(sch["week"], errors="coerce")
    sch = sch.dropna(subset=["season", "week", "home_team", "away_team"]).copy()
    sch["season"] = sch["season"].astype(int)
    sch["week"] = sch["week"].astype(int)
    sch = sch[sch["season"].isin([int(s) for s in seasons])].copy()
    if sch.empty:
        raise RuntimeError(f"Fallback schedule has no rows for requested seasons: {seasons}")

    # Match builder_v3.load_schedules output shape.
    sch = sch[
        [
            "season",
            "week",
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "game_type",
            "away_moneyline",
            "home_moneyline",
            "spread_line",
            "total_line",
            "away_rest",
            "home_rest",
        ]
    ].copy()

    if include_future:
        completed = sch.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
        future = sch[sch["home_score"].isna() | sch["away_score"].isna()].copy()
        future["home_score"] = None
        future["away_score"] = None
        future = future[future["game_type"] == "REG"].reset_index(drop=True)
        return pd.concat([completed, future], ignore_index=True)

    return sch.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)


def _install_schedule_fallback() -> None:
    original_loader = builder_v3.load_schedules

    def _safe_loader(seasons: List[int], include_future: bool = False) -> pd.DataFrame:
        try:
            return original_loader(seasons, include_future=include_future)
        except Exception as exc:
            logging.warning("Primary schedule backend failed (%s); trying local CSV fallback.", exc)
            return _load_schedule_from_local_csv(seasons, include_future=include_future)

    builder_v3.load_schedules = _safe_loader


def _row_count(df: pd.DataFrame) -> int:
    try:
        return int(len(df))
    except Exception:
        return 0


def _col_count(df: pd.DataFrame) -> int:
    try:
        return int(len(df.columns))
    except Exception:
        return 0


def _season_week_bounds(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "season_min": None,
        "season_max": None,
        "week_min": None,
        "week_max": None,
    }
    if df.empty:
        return out
    for src, dst in (
        ("season", "season"),
        ("week", "week"),
    ):
        if src in df.columns:
            vals = pd.to_numeric(df[src], errors="coerce").dropna()
            if not vals.empty:
                out[f"{dst}_min"] = int(vals.min())
                out[f"{dst}_max"] = int(vals.max())
    return out


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    reports_dir = Path(args.reports_dir).resolve()
    log_path = _setup_logging(out_dir)
    run_ts = datetime.now(timezone.utc)

    try:
        logging.info("=" * 72)
        logging.info("Dataset Builder Entry (production wrapper)")
        logging.info("start=%s end=%s out_dir=%s", args.start, args.end, out_dir)
        logging.info("=" * 72)

        _install_schedule_fallback()

        if not args.enable_heavy_stats:
            logging.info(
                "Heavy stat loaders are disabled (default). Using schedule/prior features only."
            )
            _disable_heavy_loaders()

        df = builder_v3.build_dataset(
            start_season=int(args.start),
            end_season=int(args.end),
            out_dir=out_dir,
            legacy_root_copy=bool(args.legacy_root_copy),
            production_mode=True,
            include_future=True,
            encode=str(args.encode),
            save_dominance_matrix=bool(args.save_dominance_matrix),
            no_calibration_rows=bool(args.no_calibration_rows),
            dominance_log=args.dominance_log,
        )
        _validate_output(df, strict_validation=bool(args.strict_validation))

        date_tag = run_ts.strftime("%Y%m%d")
        canonical_name = f"game_features_{date_tag}.csv"
        canonical_path = out_dir / canonical_name
        if canonical_path.exists():
            canonical_name = f"game_features_{run_ts.strftime('%Y%m%d_%H%M%S')}.csv"
            canonical_path = out_dir / canonical_name
        df.to_csv(canonical_path, index=False)

        latest_alias = out_dir / "game_features_latest.csv"
        df.to_csv(latest_alias, index=False)

        version_payload: Dict[str, Any] = {}
        if not args.skip_version_report:
            version_payload = write_dataset_version_report(
                data_dir=out_dir,
                reports_dir=reports_dir,
                limit=20,
            )

        summary: Dict[str, Any] = {
            "generated_at": run_ts.isoformat(),
            "output_path": str(canonical_path),
            "latest_alias": str(latest_alias),
            "sha256": file_sha256(canonical_path),
            "rows": _row_count(df),
            "columns": _col_count(df),
            "bounds": _season_week_bounds(df),
            "args": vars(args),
            "log_path": str(log_path),
            "version_report_latest": version_payload.get("latest"),
        }

        summary_path = out_dir / "dataset_build_latest.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        history_path = out_dir / "dataset_build_history.jsonl"
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary) + "\n")

        logging.info(
            "Dataset build complete: rows=%s cols=%s file=%s",
            summary["rows"],
            summary["columns"],
            canonical_path,
        )
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        logging.exception("Dataset build failed: %s", exc)
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": str(exc),
                    "log_path": str(log_path),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
