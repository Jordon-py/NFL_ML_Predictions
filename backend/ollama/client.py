"""Ollama client helpers for premium NFL analysis.

Data shape:
- Input chat payload: list of dictionaries with `role` and `content` strings.
- Input prediction payload: dictionary with matchup, score, probability, and
  source fields.
- Output: dictionaries with `used_llm`, `model`, `latency_ms`, reply or
  explanation fields, and optional error details.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama
from dotenv import find_dotenv, load_dotenv

from backend.ollama.memory import NFLMemory


BACKEND_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if BACKEND_ENV_PATH.exists():
    load_dotenv(BACKEND_ENV_PATH)
else:
    load_dotenv(find_dotenv())

log = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        log.warning("Invalid %s value; using %.1f", name, default)
        return default


def _split_env_list(raw: Optional[str]) -> List[str]:
    """Split comma-separated env values while dropping blanks and quotes."""
    values: List[str] = []
    for part in (raw or "").split(","):
        value = part.strip().strip('"').strip("'")
        if value and value not in values:
            values.append(value)
    return values


def _normalize_ollama_host(raw: Optional[str]) -> Optional[str]:
    """Normalize an Ollama SDK host value.

    The Python SDK appends `/api/...` request paths itself, so cloud base URLs
    configured as `https://ollama.com/api` must be reduced to `https://ollama.com`.
    """
    value = (raw or "").strip().strip('"').strip("'").rstrip("/")
    if not value:
        return None
    if value.lower().endswith("/api"):
        value = value[:-4].rstrip("/")
    return value


def _is_cloud_host(host: str) -> bool:
    return "ollama.com" in (host or "").lower()


def _is_cloud_model(model: str) -> bool:
    return "-cloud" in (model or "").lower()


def _prefer_cloud_hosts(model: Optional[str] = None) -> bool:
    env_model = model or os.getenv("OLLAMA_MODEL", "")
    return bool(os.getenv("OLLAMA_API_KEY")) and _is_cloud_model(env_model)


def ollama_host_candidates(primary_host: Optional[str] = None, model: Optional[str] = None) -> List[str]:
    """Return normalized Ollama hosts, preferring cloud for cloud models."""
    raw_hosts: List[str] = []
    if primary_host:
        raw_hosts.extend(_split_env_list(primary_host))
    if _prefer_cloud_hosts(model):
        raw_hosts.append("https://ollama.com")
    raw_hosts.extend(_split_env_list(os.getenv("OLLAMA_BASE_URL")))
    raw_hosts.extend(_split_env_list(os.getenv("OLLAMA_HOST")))
    if not raw_hosts:
        raw_hosts.append("http://localhost:11434")

    candidates: List[str] = []
    for raw in raw_hosts:
        host = _normalize_ollama_host(raw)
        if host and host not in candidates:
            candidates.append(host)

    if _prefer_cloud_hosts(model):
        candidates.sort(key=lambda host: 0 if _is_cloud_host(host) else 1)

    return candidates or ["http://localhost:11434"]


def _primary_model() -> str:
    return (os.getenv("OLLAMA_MODEL", "gemma4:e4b").split(",")[0].strip() or "gemma4:e4b")


OLLAMA_HOST = ollama_host_candidates(model=_primary_model())[0]
OLLAMA_MODEL = _primary_model()
OLLAMA_TIMEOUT = _env_float("OLLAMA_TIMEOUT_S", 15.0)
FALLBACK_MODEL = "gemma4:e4b"


def ollama_unavailable_reply(errors: List[str]) -> str:
    """Return a user-safe fallback message when all Ollama calls fail."""
    detail = " ".join(errors).lower()
    if "failed to connect" in detail or "connection" in detail:
        return (
            "Premium AI analysis is temporarily unavailable because the Ollama runtime "
            "could not be reached. The matchup prediction context was still prepared; "
            "configure OLLAMA_BASE_URL, OLLAMA_MODEL, and optional OLLAMA_API_KEY for live AI commentary."
        )
    return (
        "Premium AI analysis is temporarily unavailable because all configured Ollama models failed. "
        "Check OLLAMA_MODEL and the Ollama service logs before retrying."
    )


class OllamaClient:
    """Small compatibility wrapper around ollama.AsyncClient."""

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: Optional[float] = None,
        api_key: Optional[str] = None,
    ):
        explicit_hosts = [
            normalized
            for normalized in (_normalize_ollama_host(raw) for raw in _split_env_list(host))
            if normalized
        ]
        api_key = api_key or os.getenv("OLLAMA_API_KEY")
        self.host = explicit_hosts[0] if explicit_hosts else ollama_host_candidates(model=model)[0]
        self.model = model or OLLAMA_MODEL
        self.timeout_s = float(timeout_s) if timeout_s is not None else OLLAMA_TIMEOUT
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = ollama.AsyncClient(host=self.host, timeout=self.timeout_s, headers=headers)


def strip_code_fences(text: str) -> str:
    """Remove common Markdown code fences from model JSON output."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.strip().lower().startswith("json"):
            cleaned = cleaned.strip()[4:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def build_explanation_prompt(pred: Dict[str, Any]) -> str:
    """Create the JSON-only prompt used by prediction explanation calls."""
    home = str(pred.get("home_team", "")).upper()
    away = str(pred.get("away_team", "")).upper()
    season = str(pred.get("season", "unknown"))
    memory = NFLMemory()
    if not memory.df.empty:
        data_context = memory.df.to_dict(orient="records")[0]
    else:
        data_context = memory.get_relevant_context(
            f"What are the key stats for {home} vs {away}, {season} for this game and matchup?"
        )
    return f"""
You are explaining an NFL game prediction to a regular sports fan use the provided dataset to inform your response and say why the prediction is good or bad {memory.df}.
Be concise, avoid claiming you saw injuries/weather unless provided.

Return ONLY valid JSON with keys:
  explanation: string (1 short paragraph)
  bullets: array of 3-6 short bullet strings
  caveats: array of 1-3 short caveat strings

Game:
  home_team: {home}
  away_team: {away}
  predicted_home_score: {pred.get("home_score")}
  predicted_away_score: {pred.get("away_score")}
  home_win_probability: {pred.get("home_win_probability")}
  prediction_source: {pred.get("prediction_source", "model")}
  {data_context}
""".strip()


def model_candidates(primary_model: str) -> List[str]:
    """Return primary, env-configured, and local fallback models in order."""
    candidates = [primary_model] if primary_model else []
    for model in os.getenv("OLLAMA_MODEL", "").split(","):
        model = model.strip()
        if model and model not in candidates:
            candidates.append(model)
    if FALLBACK_MODEL not in candidates:
        candidates.append(FALLBACK_MODEL)
    return candidates


async def chat_once(
    *,
    messages: List[Dict[str, str]],
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Send one normalized chat payload to Ollama with model fallbacks."""
    active_model = model or OLLAMA_MODEL
    active_timeout = float(timeout_s) if timeout_s is not None else OLLAMA_TIMEOUT
    errors: List[str] = []

    for active_host in ollama_host_candidates(host, model=active_model):
        for candidate in model_candidates(active_model):
            try:
                client = OllamaClient(
                    host=active_host,
                    model=candidate,
                    timeout_s=active_timeout,
                ).client
                response = await asyncio.wait_for(
                    client.chat(model=candidate, messages=messages),
                    timeout=active_timeout + 0.25,
                )
                return {
                    "ok": True,
                    "model": candidate,
                    "host": active_host,
                    "content": (response.message.content or "").strip(),
                }
            except Exception as exc:
                errors.append(f"{active_host} {candidate}: {exc}")
                log.warning("Ollama chat failed for host %s model %s: %s", active_host, candidate, exc)

    return {
        "ok": False,
        "model": active_model,
        "host": host or OLLAMA_HOST,
        "error": "; ".join(errors) or "ollama chat failed",
    }


async def explain_prediction(
    pred: Dict[str, Any],
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate a structured explanation payload using Ollama, best-effort."""
    started = time.perf_counter()
    active_model = model or OLLAMA_MODEL
    result = await chat_once(
        host=host,
        model=active_model,
        timeout_s=timeout_s,
        messages=[{"role": "user", "content": build_explanation_prompt(pred)}],
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    if not result.get("ok"):
        return {
            "used_llm": False,
            "model": active_model,
            "latency_ms": latency_ms,
            "error": result.get("error") or "ollama chat failed",
        }

    content = str(result.get("content") or "")
    try:
        parsed = json.loads(strip_code_fences(content))
    except Exception:
        return {
            "used_llm": True,
            "model": result.get("model") or active_model,
            "latency_ms": latency_ms,
            "explanation": content,
            "bullets": [],
            "caveats": [],
            "error": None,
        }

    return {
        "used_llm": True,
        "model": result.get("model") or active_model,
        "latency_ms": latency_ms,
        "explanation": str(parsed.get("explanation", "")).strip(),
        "bullets": list(parsed.get("bullets", []) or []),
        "caveats": list(parsed.get("caveats", []) or []),
        "error": None,
    }


async def chat_messages(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Chat with Ollama using a list of role/content messages."""
    chat_payload: List[Dict[str, str]] = []
    if system_prompt is None:
        try:
            system_prompt = NFLMemory().system_prompt
        except Exception as exc:
            log.warning("NFL memory system prompt unavailable: %s", exc)
    if system_prompt:
        chat_payload.append({"role": "system", "content": system_prompt})

    for msg in messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role not in {"user", "assistant", "system"}:
            role = "user"
        chat_payload.append({"role": role, "content": content})

    if not chat_payload:
        return {"used_llm": False, "error": "no messages to send", "model": model or OLLAMA_MODEL}

    started = time.perf_counter()
    result = await chat_once(host=host, model=model, timeout_s=timeout_s, messages=chat_payload)
    latency_ms = int((time.perf_counter() - started) * 1000)

    if not result.get("ok"):
        return {
            "used_llm": False,
            "model": model or OLLAMA_MODEL,
            "latency_ms": latency_ms,
            "error": result.get("error") or "ollama chat failed",
        }

    return {
        "used_llm": True,
        "model": result.get("model") or model or OLLAMA_MODEL,
        "latency_ms": latency_ms,
        "reply": str(result.get("content") or "").strip(),
    }
