# make_schema.py
"""
Schema writer for CSV artifacts.

What this provides
- Compact JSON schema: dtype, non-null %, min/max (numeric), example value
- Human-readable Markdown dictionary
- Required-column validation to catch pipeline drift early

Usage
- Import and call from build_csvs.py:
    from make_schema import write_schema_files
    write_schema_files(Path("data/team_game_iter3.csv"), title="Team-Game Dataset Schema")

Why this matters (teaching)
- New contributors learn columns fast without reading code
- CI can fail merges on missing/renamed fields
- Historical schemas track evolution of your dataset
"""

from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

def _col_profile(s: pd.Series) -> dict:
    """Summarize one column for schema."""
    dtype = str(s.dtype)
    non_null = float(s.notna().mean())
    is_num = pd.api.types.is_numeric_dtype(s)
    # use .dropna() to avoid errors on empty columns
    example = s.dropna().iloc[0] if s.notna().any() else None
    # cast min/max to float for JSON safety, else None
    minv = float(s.min()) if is_num and s.notna().any() else None
    maxv = float(s.max()) if is_num and s.notna().any() else None
    return {
        "dtype": dtype,
        "non_null_pct": round(non_null, 6),
        "min": minv,
        "max": maxv,
        "example": example,
        "description": ""  # fill manually over time
    }

def _to_markdown(schema: dict, title: str) -> str:
    lines = [f"# {title}", ""]
    lines.append("| Column | Dtype | Non-null % | Min | Max | Example | Description |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    for col, meta in schema.items():
        lines.append(
            f"| `{col}` | `{meta['dtype']}` | {meta['non_null_pct']:.3f} | "
            f"{'' if meta['min'] is None else meta['min']:.6g} | "
            f"{'' if meta['max'] is None else meta['max']:.6g} | "
            f"{'' if meta['example'] is None else str(meta['example'])[:40]} | "
            f"{meta['description']} |"
        )
    lines.append("")
    lines.append("> Tip: Add human descriptions over time, then commit.")
    return "\n".join(lines)

def write_schema_files(csv_path: Path, title: str = "Dataset Schema", required_core: set[str] | None = None) -> None:
    """
    Generate JSON + Markdown schema files next to the CSV.
    - required_core: set of columns that must exist (fast guard)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Validate required
    if required_core:
        missing = sorted(required_core - set(df.columns))
        if missing:
            raise SystemExit(f"[schema] Missing required columns: {missing}")

    # Build schema
    schema = {c: _col_profile(df[c]) for c in df.columns}

    # Write JSON
    json_path = csv_path.with_suffix(".schema.json")
    json_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False))

    # Write Markdown
    md_path = csv_path.with_suffix(".schema.md")
    md_path.write_text(_to_markdown(schema, title))

    print(f"[schema] Wrote {json_path} and {md_path}")

