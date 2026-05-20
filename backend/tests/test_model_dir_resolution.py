from pathlib import Path

from backend import main as main_module


def _write_bundle(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in ("home_model.joblib", "away_model.joblib", "win_clf_calibrated.joblib"):
        (path / name).write_bytes(b"bundle")


def test_find_models_dir_prefers_promoted_current_bundle(monkeypatch, tmp_path):
    backend_dir = tmp_path / "backend"
    data_dir = backend_dir / "data"
    current_dir = data_dir / "models" / "current"
    shared_dir = data_dir / "models"
    legacy_dir = backend_dir / "models"

    _write_bundle(current_dir)
    _write_bundle(shared_dir)
    _write_bundle(legacy_dir)

    monkeypatch.setattr(main_module, "BASE_DIR", backend_dir)
    monkeypatch.setattr(main_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(main_module, "CURRENT_MODELS_DIR", current_dir)
    monkeypatch.setattr(main_module, "SETTINGS", type("SettingsStub", (), {"resolved_models_dir": None})())
    monkeypatch.delenv("MODELS_DIR", raising=False)
    monkeypatch.delenv("MODELS_PATH", raising=False)
    monkeypatch.delenv("MODEL_DIR", raising=False)

    assert main_module._find_models_dir() == current_dir


def test_find_models_dir_prefers_backend_data_models_before_legacy_models(monkeypatch, tmp_path):
    backend_dir = tmp_path / "backend"
    data_dir = backend_dir / "data"
    current_dir = data_dir / "models" / "current"
    shared_dir = data_dir / "models"
    legacy_dir = backend_dir / "models"

    _write_bundle(shared_dir)
    _write_bundle(legacy_dir)

    monkeypatch.setattr(main_module, "BASE_DIR", backend_dir)
    monkeypatch.setattr(main_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(main_module, "CURRENT_MODELS_DIR", current_dir)
    monkeypatch.setattr(main_module, "SETTINGS", type("SettingsStub", (), {"resolved_models_dir": None})())
    monkeypatch.delenv("MODELS_DIR", raising=False)
    monkeypatch.delenv("MODELS_PATH", raising=False)
    monkeypatch.delenv("MODEL_DIR", raising=False)

    assert main_module._find_models_dir() == shared_dir
