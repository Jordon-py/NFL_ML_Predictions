from __future__ import annotations

from pathlib import Path

from backend.scripts import weekly_retrain


class _CompletedProcess:
    def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_weekly_retrain_defaults_to_canonical_builddataset(monkeypatch):
    monkeypatch.setattr("sys.argv", ["weekly_retrain.py"])
    args = weekly_retrain.parse_args()

    assert Path(args.dataset_build_script).resolve() == (weekly_retrain.BACKEND_DIR / "builddataset.py").resolve()


def test_run_dataset_build_uses_current_builddataset_cli(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_run(cmd, cwd, text, capture_output):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["text"] = text
        captured["capture_output"] = capture_output
        return _CompletedProcess(stdout="ok", stderr="")

    monkeypatch.setattr(weekly_retrain.subprocess, "run", _fake_run)

    result = weekly_retrain._run_dataset_build(
        python_exe="python",
        build_script=weekly_retrain.BACKEND_DIR / "builddataset.py",
        data_dir=tmp_path / "datasets",
        reports_dir=tmp_path / "reports",
        start_season=2020,
        end_season=2025,
        extra_args=["--legacy-root-copy"],
    )

    cmd = captured["cmd"]
    assert result["returncode"] == 0
    assert "--start" in cmd
    assert "--end" in cmd
    assert "--out-dir" in cmd
    assert "--legacy-root-copy" in cmd
    assert "--reports-dir" not in cmd
    assert "--strict-validation" not in cmd
