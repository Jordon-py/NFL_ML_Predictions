from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent
IS_HEROKU = bool(os.getenv("DYNO"))


def _normalize_origin(raw: str) -> Optional[str]:
    value = (raw or "").strip().strip('"').strip("'").rstrip("/")
    if not value:
        return None
    if "://" not in value and "." in value:
        value = f"https://{value}"
    try:
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return None
    except Exception:
        return None
    return value


def _split_origins(raw: str) -> List[str]:
    out: List[str] = []
    for part in (raw or "").split(","):
        origin = _normalize_origin(part)
        if origin and origin not in out:
            out.append(origin)
    return out


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None if IS_HEROKU else str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = Field(
        default=("production" if IS_HEROKU else "development"),
        validation_alias=AliasChoices("APP_ENV", "ENVIRONMENT", "NODE_ENV"),
    )
    log_level: str = Field(default="INFO", validation_alias=AliasChoices("LOG_LEVEL"))

    allowed_origins_raw: str = Field(
        default="",
        validation_alias=AliasChoices("ALLOWED_ORIGINS", "CORS_ORIGINS"),
    )
    allow_origin_regex: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("CORS_ORIGINS_REGEX", "ALLOW_ORIGIN_REGEX"),
    )
    restrict_cors: bool = Field(default=True, validation_alias=AliasChoices("RESTRICT_CORS"))
    allow_vercel_previews: bool = Field(default=True, validation_alias=AliasChoices("ALLOW_VERCEL_PREVIEWS"))

    dataset_path: Optional[str] = Field(default=None, validation_alias=AliasChoices("DATASET_PATH"))
    schedule_path: Optional[str] = Field(default=None, validation_alias=AliasChoices("SCHEDULE_PATH"))
    models_dir: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("MODELS_DIR", "MODELS_PATH", "MODEL_DIR"),
    )

    enable_admin: bool = Field(default=False, validation_alias=AliasChoices("ENABLE_ADMIN"))
    admin_token: str = Field(default="", validation_alias=AliasChoices("ADMIN_TOKEN"))

    predict_cache_ttl_sec: int = Field(default=900, validation_alias=AliasChoices("PREDICT_CACHE_TTL_SEC"))
    predict_cache_max_items: int = Field(default=1000, validation_alias=AliasChoices("PREDICT_CACHE_MAX_ITEMS"))

    @model_validator(mode="after")
    def _validate_prod_cors(self) -> "Settings":
        if self.is_production and self.restrict_cors:
            if not self.allowed_origins and not self.effective_allow_origin_regex:
                raise ValueError(
                    "Production CORS is restricted but neither ALLOWED_ORIGINS nor CORS_ORIGINS_REGEX is configured."
                )
        return self

    @property
    def is_production(self) -> bool:
        env = (self.app_env or "").strip().lower()
        return env in {"prod", "production"} or IS_HEROKU

    @property
    def allowed_origins(self) -> List[str]:
        configured = _split_origins(self.allowed_origins_raw)
        defaults = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
            "https://new-nfl-predict.vercel.app",
            "https://nfl-ml-predictions.vercel.app",
        ]
        out = configured or defaults
        if not self.is_production:
            for origin in defaults:
                if origin.startswith("http://") and origin not in out:
                    out.append(origin)
        return out

    @property
    def effective_allow_origin_regex(self) -> Optional[str]:
        raw = (self.allow_origin_regex or "").strip()
        if raw:
            return raw
        if self.allow_vercel_previews:
            return r"^https://.*\.vercel\.app$"
        return None

    @property
    def resolved_schedule_path(self) -> Optional[Path]:
        if not self.schedule_path:
            return None
        p = Path(self.schedule_path).expanduser()
        if p.is_absolute():
            return p
        candidates = [
            (BACKEND_DIR / p).resolve(),
            (REPO_ROOT / p).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @property
    def resolved_models_dir(self) -> Optional[Path]:
        if not self.models_dir:
            return None
        p = Path(self.models_dir).expanduser()
        if p.is_absolute():
            return p
        candidates = [
            (BACKEND_DIR / p).resolve(),
            (REPO_ROOT / p).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

