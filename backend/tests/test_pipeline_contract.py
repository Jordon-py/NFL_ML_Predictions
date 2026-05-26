import json
from pathlib import Path

from backend.utils.ops_reporting import resolve_latest_dataset


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "backend" / "models"


def test_latest_dataset_manifest_resolves_to_existing_csv():
    dataset_path = resolve_latest_dataset(ROOT / "backend" / "data")

    assert dataset_path.exists()
    assert dataset_path.name.endswith("_clean.csv")


def test_active_feature_manifest_has_no_remaining_hard_leaks():
    manifest_path = MODELS_DIR / "feature_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["hard_leak_columns_remaining"] == []
    assert payload["dropped_reason_counts"]["same_week_player_stat"] >= 1


def test_active_metadata_exposes_canonical_artifact_keys():
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    artifacts = metadata["artifacts"]

    for key in ("preprocessor", "reg_home", "reg_away", "clf_home_win"):
        artifact_path = MODELS_DIR / artifacts[key]
        assert artifact_path.exists(), f"{key} points at missing artifact {artifact_path}"

    assert metadata["gate"]["passed"] is True
    assert metadata["feature_selection"]["hard_leak_columns_remaining"] == []
