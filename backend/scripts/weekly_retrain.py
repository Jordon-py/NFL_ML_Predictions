#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.utils.ops_reporting import (  # noqa: E402
    resolve_latest_dataset,
    write_dataset_version_report,
    write_performance_drift_report,
)


def parse_args() -> argparse.Namespace:
    current_year = datetime.now(timezone.utc).year
    parser = argparse.ArgumentParser(
        description="Weekly automation: dataset versioning + retrain + drift report."
    )
    parser.add_argument(
        "--skip-dataset-build",
        action="store_true",
        help="Skip dataset rebuild and use existing latest dataset only.",
    )
    parser.add_argument(
        "--dataset-build-script",
        type=str,
        default=str((BACKEND_DIR / "scripts" / "build_csv_datasets.py").resolve()),
        help="Dataset build script entrypoint.",
    )
    parser.add_argument(
        "--build-start-season",
        type=int,
        default=max(1999, current_year - 8),
        help="Start season for dataset rebuild.",
    )
    parser.add_argument(
        "--build-end-season",
        type=int,
        default=current_year + 1,
        help="End season for dataset rebuild (inclusive).",
    )
    parser.add_argument(
        "--build-extra",
        action="append",
        default=[],
        help="Extra args forwarded to the dataset build script.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.getenv("DATASET_PATH"),
        help="Optional explicit dataset CSV path. Defaults to latest game_features*.csv.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str((BACKEND_DIR / "data").resolve()),
        help="Directory containing game_features*.csv files.",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str((BACKEND_DIR / "reports").resolve()),
        help="Report output directory.",
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default=str((BACKEND_DIR / "train_models.py").resolve()),
        help="Training script entrypoint.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="CV splits to pass to the training script.",
    )
    parser.add_argument(
        "--embargo",
        type=int,
        default=1,
        help="Embargo groups to pass to the training script.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only regenerate reports.",
    )
    parser.add_argument(
        "--train-extra",
        action="append",
        default=[],
        help="Extra args forwarded to the training script. Can be specified multiple times.",
    )
    return parser.parse_args()


def _run_train(
    *,
    python_exe: str,
    train_script: Path,
    dataset_path: Path,
    splits: int,
    embargo: int,
    extra_args: List[str],
) -> Dict[str, object]:
    cmd = [
        python_exe,
        str(train_script),
        "--data",
        str(dataset_path),
        "--production",
        "--splits",
        str(splits),
        "--embargo",
        str(embargo),
    ]
    cmd.extend(extra_args)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _run_dataset_build(
    *,
    python_exe: str,
    build_script: Path,
    data_dir: Path,
    reports_dir: Path,
    start_season: int,
    end_season: int,
    extra_args: List[str],
) -> Dict[str, object]:
    cmd = [
        python_exe,
        str(build_script),
        "--start",
        str(int(start_season)),
        "--end",
        str(int(end_season)),
        "--out-dir",
        str(data_dir),
        "--reports-dir",
        str(reports_dir),
        "--strict-validation",
    ]
    cmd.extend(extra_args)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    train_script = Path(args.train_script).resolve()
    build_script = Path(args.dataset_build_script).resolve()

    build_result: Dict[str, object] = {
        "skipped": bool(args.skip_dataset_build),
        "command": None,
        "returncode": None,
    }
    if not args.skip_dataset_build:
        build_result = _run_dataset_build(
            python_exe=sys.executable,
            build_script=build_script,
            data_dir=data_dir,
            reports_dir=reports_dir,
            start_season=int(args.build_start_season),
            end_season=int(args.build_end_season),
            extra_args=list(args.build_extra or []),
        )
        if int(build_result.get("returncode", 1)) != 0:
            raise RuntimeError(
                f"Dataset build failed (returncode={build_result['returncode']}). "
                "See report artifact for stderr_tail."
            )

    try:
        dataset_path = resolve_latest_dataset(
            data_dir=data_dir,
            explicit_path=args.dataset_path,
        )
    except FileNotFoundError:
        if args.dataset_path:
            print(
                json.dumps(
                    {
                        "warning": (
                            f"DATASET_PATH not found ({args.dataset_path}); "
                            "falling back to latest game_features*.csv"
                        )
                    }
                )
            )
        dataset_path = resolve_latest_dataset(
            data_dir=data_dir,
            explicit_path=None,
        )

    dataset_report = write_dataset_version_report(
        data_dir=data_dir,
        reports_dir=reports_dir,
        limit=12,
    )

    train_result: Dict[str, object] = {
        "skipped": bool(args.skip_train),
        "command": None,
        "returncode": None,
    }
    if not args.skip_train:
        train_result = _run_train(
            python_exe=sys.executable,
            train_script=train_script,
            dataset_path=dataset_path,
            splits=int(args.splits),
            embargo=int(args.embargo),
            extra_args=list(args.train_extra or []),
        )
        if int(train_result.get("returncode", 1)) != 0:
            raise RuntimeError(
                f"Training failed (returncode={train_result['returncode']}). "
                "See report artifact for stderr_tail."
            )

    drift_report = write_performance_drift_report(
        model_root=BACKEND_DIR,
        reports_dir=reports_dir,
        limit=104,
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_build": build_result,
        "dataset_report_latest": dataset_report.get("latest"),
        "train_result": train_result,
        "drift_points": drift_report.get("count", 0),
    }
    out_dir = reports_dir / "automation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "weekly_retrain_latest.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
