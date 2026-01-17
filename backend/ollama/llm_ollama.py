# ==========================================
# File: backend/ollama/llm_ollama.py
# Role: Backend helper module.
# Input Data: Function inputs.
# Output Data: Module outputs.
# Dependencies: __future__, asyncio, json, os
# Notes: Shared utilities.
# ==========================================

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if "```" in t else t
        t = t.replace("json", "", 1).strip()
    if t.endswith("```"):
        t = t[:-3].strip()
    return t


def _normalize_host(host: str) -> str:
    cleaned = str(host or "").strip().rstrip("/")
    if cleaned.lower().endswith("/api"):
        cleaned = cleaned[:-4]
    return cleaned


def _resolve_host(host: Optional[str]) -> str:
    if host:
        return _normalize_host(host)
    env_host = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL")
    if env_host:
        return _normalize_host(env_host)
    return "http://127.0.0.1:11434"


def _ollama_headers() -> Dict[str, str]:
    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


async def _ollama_list_model_names(*, host: str, timeout_s: float) -> list[str]:
    host = _normalize_host(host)
    if not host:
        return []
    url = f"{host}/api/tags"
    timeout = httpx.Timeout(timeout_s)
    headers = _ollama_headers()
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    models: list[str] = []
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        for item in data["models"]:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model")
            if name:
                models.append(str(name))
    return models


def _pick_fallback_model(models: list[str]) -> Optional[str]:
    if not models:
        return None
    preferred = ("mistral:latest",)
    for m in preferred:
        if m in models:
            return m
    return models[0]


def _build_prompt(pred: Dict[str, Any]) -> str:
    home = str(pred.get("home_team", "")).upper()
    away = str(pred.get("away_team", "")).upper()
    hs = pred.get("home_score")
    as_ = pred.get("away_score")
    p_home = pred.get("home_win_probability")
    src = pred.get("prediction_source", "model")

    return f"""
You are explaining an NFL game prediction to a regular sports fan.
Be concise, avoid claiming you saw injuries/weather unless provided.

Return ONLY valid JSON with keys:
  explanation: string (1 short paragraph)
  bullets: array of 3-6 short bullet strings
  caveats: array of 1-3 short caveat strings

Game:
  home_team: {home}
  away_team: {away}
  predicted_home_score: {hs}
  predicted_away_score: {as_}
  home_win_probability: {p_home}
  prediction_source: {src}
"""


async def _ollama_chat(
    *,
    host: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout_s: float,
) -> Dict[str, Any]:
    """
    Minimal async client for Ollama's REST API.

    Ref: POST /api/chat
    https://github.com/ollama/ollama/blob/main/docs/api.md
    """
    host = _normalize_host(host)
    if not host:
        return {"ok": False, "error": "OLLAMA_HOST not set"}

    model = str(model or "").strip()
    if not model:
        return {"ok": False, "error": "OLLAMA_MODEL not set"}

    url = f"{host}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    timeout = httpx.Timeout(timeout_s)
    headers = _ollama_headers()

    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            resp = await client.post(url, json=payload)
    except Exception as e:
        return {"ok": False, "error": f"ollama request failed: {e}", "model": model, "host": host}

    try:
        resp.raise_for_status()
    except Exception as e:
        body_preview = ""
        body_error = None
        try:
            body_preview = resp.text[:500]
            data = resp.json()
            if isinstance(data, dict):
                body_error = data.get("error") or data.get("detail") or data.get("message")
        except Exception:
            body_preview = ""
        return {
            "ok": False,
            "error": str(body_error).strip() if body_error else f"ollama http error: {e}",
            "model": model,
            "host": host,
            "status_code": resp.status_code,
            "body": body_preview,
        }

    try:
        data = resp.json()
    except Exception as e:
        return {
            "ok": False,
            "error": f"ollama returned non-json: {e}",
            "model": model,
            "host": host,
            "status_code": resp.status_code,
            "body": (resp.text[:500] if resp.text else ""),
        }

    message = data.get("message") if isinstance(data, dict) else None
    content = ""
    if isinstance(message, dict):
        content = str(message.get("content") or "")
    return {"ok": True, "model": model, "host": host, "raw": data, "content": content}


async def explain_prediction(
    pred: Dict[str, Any],
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate a JSON explanation payload using Ollama (best-effort)."""
    host = _resolve_host(host)
    model = model or os.getenv("OLLAMA_MODEL", "qwen3-coder:480b-cloud")
    timeout_s = float(timeout_s) if timeout_s is not None else float(os.getenv("OLLAMA_TIMEOUT_S", "6"))

    prompt = _build_prompt(pred)
    try:
        t0 = time.perf_counter()
        result = await asyncio.wait_for(
            _ollama_chat(
                host=host,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout_s=timeout_s,
            ),
            timeout=timeout_s + 0.25,
        )

        # If the configured model doesn't exist locally, retry once with a known available model.
        if (
            not result.get("ok")
            and int(result.get("status_code") or 0) == 404
            and isinstance(result.get("error"), str)
            and "model" in result["error"].lower()
            and "not found" in result["error"].lower()
        ):
            names = await _ollama_list_model_names(host=host, timeout_s=min(timeout_s, 5.0))
            fallback_model = _pick_fallback_model(names)
            if fallback_model and fallback_model != model:
                model = fallback_model
                result = await asyncio.wait_for(
                    _ollama_chat(
                        host=host,
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        timeout_s=timeout_s,
                    ),
                    timeout=timeout_s + 0.25,
                )

        latency_ms = int((time.perf_counter() - t0) * 1000)

        if not result.get("ok"):
            return {
                "used_llm": False,
                "model": model,
                "latency_ms": latency_ms,
                "error": result.get("error") or "ollama chat failed",
            }

        content = str(result.get("content") or "")

        raw = _strip_code_fences(content)
        try:
            obj = json.loads(raw)
        except Exception:
            # Best-effort: return plain text when the model didn't return JSON.
            return {
                "used_llm": True,
                "model": model,
                "latency_ms": latency_ms,
                "explanation": content.strip(),
                "bullets": [],
                "caveats": [],
                "error": None,
            }
        return {
            "used_llm": True,
            "model": model,
            "latency_ms": latency_ms,
            "explanation": str(obj.get("explanation", "")).strip(),
            "bullets": list(obj.get("bullets", []) or []),
            "caveats": list(obj.get("caveats", []) or []),
        }
    except asyncio.TimeoutError:
        return {"used_llm": False, "error": "ollama request timed out", "model": model}
    except Exception as e:
        return {"used_llm": False, "error": str(e), "model": model}


async def chat_messages(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Chat with Ollama using a list of role/content messages."""
    host = _resolve_host(host)
    model = model or os.getenv("OLLAMA_MODEL", "mistral:latest")
    timeout_s = float(timeout_s) if timeout_s is not None else float(os.getenv("OLLAMA_TIMEOUT_S", "6"))

    chat_payload: List[Dict[str, str]] = []
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
        return {"used_llm": False, "error": "no messages to send", "model": model}

    try:
        t0 = time.perf_counter()
        result = await asyncio.wait_for(
            _ollama_chat(host=host, model=model, messages=chat_payload, timeout_s=timeout_s),
            timeout=timeout_s + 0.25,
        )

        if (
            not result.get("ok")
            and int(result.get("status_code") or 0) == 404
            and isinstance(result.get("error"), str)
            and "model" in result["error"].lower()
            and "not found" in result["error"].lower()
        ):
            names = await _ollama_list_model_names(host=host, timeout_s=min(timeout_s, 5.0))
            fallback_model = _pick_fallback_model(names)
            if fallback_model and fallback_model != model:
                model = fallback_model
                result = await asyncio.wait_for(
                    _ollama_chat(host=host, model=model, messages=chat_payload, timeout_s=timeout_s),
                    timeout=timeout_s + 0.25,
                )

        latency_ms = int((time.perf_counter() - t0) * 1000)

        if not result.get("ok"):
            return {"used_llm": False, "error": result.get("error") or "ollama chat failed", "model": model}

        content = str(result.get("content") or "")

        return {
            "used_llm": True,
            "model": model,
            "latency_ms": latency_ms,
            "reply": str(content or "").strip(),
        }
    except asyncio.TimeoutError:
        return {"used_llm": False, "error": "ollama request timed out", "model": model}
    except Exception as e:
        return {"used_llm": False, "error": str(e), "model": model}


async def _demo() -> None:
    pred = {
        "home_team": "KC",
        "away_team": "BUF",
        "home_score": 24,
        "away_score": 27,
        "home_win_probability": 0.46,
        "prediction_source": "model",
    }
    result = await explain_prediction(pred)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(_demo())
