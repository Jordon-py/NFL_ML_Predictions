from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        x = float(value)
        if pd.isna(x):
            return None
        return x
    except Exception:
        return None


def _mean_col(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def _parse_ts(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        # Accept common metadata formats, e.g. "...Z" or "... UTC".
        text = text.replace(" UTC", "+00:00")
        dt = pd.to_datetime(text, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def file_sha256(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        data = path.read_bytes().replace(b"\r\n", b"\n")
        return hashlib.sha256(data).hexdigest()
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_latest_dataset_manifest(data_dir: Path) -> Dict[str, Any]:
    candidates = [
        data_dir / "latest_dataset.json",
        data_dir / "datasets" / "latest_dataset.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            payload = _read_json(candidate)
            if isinstance(payload, dict) and payload:
                payload["_manifest_path"] = str(candidate)
                return payload
    return {}


def _dataset_search_roots(data_dir: Path) -> List[Path]:
    roots = [data_dir, data_dir / "datasets"]
    seen: set[Path] = set()
    out: List[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _dataset_csv_candidates(data_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    for root in _dataset_search_roots(data_dir):
        if not root.exists():
            continue
        candidates.extend(root.glob("game_features*.csv"))
    deduped: Dict[str, Path] = {}
    for candidate in candidates:
        deduped[str(candidate.resolve())] = candidate.resolve()
    return list(deduped.values())


def _resolve_manifest_dataset_path(data_dir: Path, raw_path: str) -> Optional[Path]:
    path = Path(str(raw_path)).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                (data_dir / path).resolve(),
                (data_dir / "datasets" / path).resolve(),
                (data_dir.parent / path).resolve(),
                (data_dir.parent / "data" / path).resolve(),
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_latest_dataset(data_dir: Path, explicit_path: Optional[str] = None) -> Path:
    if explicit_path:
        raw = Path(explicit_path).expanduser()
        candidates = [raw]
        if not raw.is_absolute():
            candidates.append((data_dir / raw).resolve())
            candidates.append((data_dir / "datasets" / raw).resolve())
            candidates.append((data_dir.parent / raw).resolve())
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"DATASET_PATH does not exist: {raw}")

    latest_manifest = load_latest_dataset_manifest(data_dir)
    manifest_candidates = [
        latest_manifest.get("clean_dataset_path"),
        latest_manifest.get("completed_dataset_path"),
        latest_manifest.get("raw_dataset_path"),
    ]
    for raw_path in manifest_candidates:
        if not raw_path:
            continue
        resolved = _resolve_manifest_dataset_path(data_dir, str(raw_path))
        if resolved is not None:
            return resolved

    candidates = sorted(_dataset_csv_candidates(data_dir), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No game_features*.csv found in {data_dir}")
    return candidates[0]


def collect_dataset_versions(data_dir: Path, limit: int = 12) -> List[Dict[str, Any]]:
    files = sorted(_dataset_csv_candidates(data_dir), key=lambda p: p.stat().st_mtime)
    if limit > 0:
        files = files[-limit:]

    out: List[Dict[str, Any]] = []
    for p in files:
        stat = p.stat()
        sample = pd.read_csv(p, nrows=1)
        rows = max(0, sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore")) - 1)
        out.append(
            {
                "file_name": p.name,
                "path": str(p),
                "rows": int(rows),
                "columns": int(len(sample.columns)),
                "byte_size": int(stat.st_size),
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "sha256": file_sha256(p),
            }
        )
    return out


def write_dataset_version_report(
    data_dir: Path,
    reports_dir: Path,
    *,
    limit: int = 12,
) -> Dict[str, Any]:
    versions = collect_dataset_versions(data_dir=data_dir, limit=limit)
    latest = versions[-1] if versions else None
    previous = versions[-2] if len(versions) > 1 else None

    row_delta = None
    col_delta = None
    changed = None
    if latest and previous:
        row_delta = int(latest["rows"] - previous["rows"])
        col_delta = int(latest["columns"] - previous["columns"])
        changed = bool(latest["sha256"] != previous["sha256"])

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "latest": latest,
        "previous": previous,
        "row_delta_vs_previous": row_delta,
        "column_delta_vs_previous": col_delta,
        "content_changed_vs_previous": changed,
        "versions": versions,
    }

    out_dir = reports_dir / "versioning"
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_json = out_dir / "dataset_version_latest.json"
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    history_jsonl = out_dir / "dataset_version_history.jsonl"
    with history_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")

    md_lines = [
        "# Dataset Versioning Report",
        "",
        f"Generated: {payload['generated_at']}",
        "",
    ]
    if latest:
        md_lines.extend(
            [
                f"- Latest file: `{latest['file_name']}`",
                f"- Rows: `{latest['rows']}`",
                f"- Columns: `{latest['columns']}`",
                f"- SHA256: `{latest['sha256']}`",
            ]
        )
    if previous:
        md_lines.extend(
            [
                f"- Previous file: `{previous['file_name']}`",
                f"- Row delta vs previous: `{row_delta}`",
                f"- Column delta vs previous: `{col_delta}`",
                f"- Content changed: `{changed}`",
            ]
        )
    (out_dir / "dataset_version_latest.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return payload


def collect_performance_drift(model_root: Path, limit: int = 104) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for csv_path in sorted(model_root.glob("20*/models/cv_fold_metrics.csv")):
        run_models_dir = csv_path.parent
        run_id = run_models_dir.parent.name

        fold_df = pd.read_csv(csv_path)
        home_mae = _mean_col(fold_df, "home_mae_val")
        away_mae = _mean_col(fold_df, "away_mae_val")
        mae_vals = [x for x in (home_mae, away_mae) if x is not None]
        mae = float(sum(mae_vals) / len(mae_vals)) if mae_vals else None
        brier = _mean_col(fold_df, "win_brier_val")

        metadata = _read_json(run_models_dir / "metadata.json")
        summary = _read_json(run_models_dir / "training_summary.json")

        if brier is None:
            brier = _to_float(summary.get("win", {}).get("Brier_mean_val"))
        if mae is None:
            home_s = _to_float(summary.get("home", {}).get("MAE_mean_val"))
            away_s = _to_float(summary.get("away", {}).get("MAE_mean_val"))
            vals = [x for x in (home_s, away_s) if x is not None]
            if vals:
                mae = float(sum(vals) / len(vals))

        trained_at = (
            _parse_ts(metadata.get("timestamp"))
            or _parse_ts(metadata.get("training_timestamp_utc"))
            or _parse_ts(summary.get("training_timestamp_utc"))
        )
        if trained_at is None:
            try:
                trained_at = datetime.strptime(run_id, "%Y%m%d").replace(tzinfo=timezone.utc)
            except Exception:
                trained_at = datetime.fromtimestamp(run_models_dir.stat().st_mtime, tz=timezone.utc)

        points.append(
            {
                "run_id": run_id,
                "trained_at": trained_at.isoformat(),
                "brier": brier,
                "mae": mae,
                "home_mae": home_mae,
                "away_mae": away_mae,
                "source_csv": str(csv_path),
            }
        )

    points = sorted(points, key=lambda x: x["trained_at"])
    if limit > 0:
        points = points[-limit:]
    return points


def write_performance_drift_report(
    model_root: Path,
    reports_dir: Path,
    *,
    limit: int = 104,
) -> Dict[str, Any]:
    points = collect_performance_drift(model_root=model_root, limit=limit)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "points": points,
        "count": len(points),
    }

    out_dir = reports_dir / "drift"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "performance_drift_latest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    md_lines = [
        "# Performance Drift (Brier/MAE Over Time)",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "| Run | Trained At | Brier | MAE |",
        "| --- | --- | ---: | ---: |",
    ]
    for p in points:
        brier = "n/a" if p.get("brier") is None else f"{p['brier']:.4f}"
        mae = "n/a" if p.get("mae") is None else f"{p['mae']:.3f}"
        md_lines.append(f"| {p['run_id']} | {p['trained_at']} | {brier} | {mae} |")
    (out_dir / "performance_drift_latest.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return payload
