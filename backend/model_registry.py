"""Model registry extension point.

Data shape:
- Input: model bundle metadata dictionaries loaded from backend model artifact
  directories.
- Output: normalized registry records for future model discovery work.

The active runtime still loads models through `backend/main.py` and the
pipeline-status services. This module is intentionally a placeholder until the
registry behavior is implemented.
"""
