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


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if "```" in t else t
        t = t.replace("json", "", 1).strip()
    if t.endswith("```"):
        t = t[:-3].strip()
    return t


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


async def explain_prediction(
    pred: Dict[str, Any],
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate a JSON explanation payload using Ollama (best-effort)."""
    try:
        from ollama import AsyncClient
    except Exception as e:
        return {"used_llm": False, "error": f"ollama python package not available: {e}"}

    host = host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = model or os.getenv("OLLAMA_MODEL", "deepseek-v3.2:cloud")
    timeout_s = float(timeout_s) if timeout_s is not None else float(os.getenv("OLLAMA_TIMEOUT_S", "6"))

    prompt = _build_prompt(pred)
    try:
        client = AsyncClient(host=host)
        t0 = time.perf_counter()
        resp = await asyncio.wait_for(
            client.chat(model=model, messages=[{"role": "user", "content": prompt}]),
            timeout=timeout_s,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        content = ""
        if isinstance(resp, dict):
            message = resp.get("message")
            if isinstance(message, dict):
                content = message.get("content", "") or ""
        else:
            msg = getattr(resp, "message", None)
            content = getattr(msg, "content", "") if msg is not None else ""

        raw = _strip_code_fences(content)
        obj = json.loads(raw)
        return {
            "used_llm": True,
            "model": model,
            "latency_ms": latency_ms,
            "explanation": str(obj.get("explanation", "")).strip(),
            "bullets": list(obj.get("bullets", []) or []),
            "caveats": list(obj.get("caveats", []) or []),
        }
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
    try:
        from ollama import AsyncClient
    except Exception as e:
        return {"used_llm": False, "error": f"ollama python package not available: {e}"}

    host = host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = model or os.getenv("OLLAMA_MODEL", "deepseek-v3.2:cloud")
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
        client = AsyncClient(host=host)
        t0 = time.perf_counter()
        resp = await asyncio.wait_for(
            client.chat(model=model, messages=chat_payload),
            timeout=timeout_s,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        content = ""
        if isinstance(resp, dict):
            message = resp.get("message")
            if isinstance(message, dict):
                content = message.get("content", "") or ""
        else:
            msg = getattr(resp, "message", None)
            content = getattr(msg, "content", "") if msg is not None else ""

        return {
            "used_llm": True,
            "model": model,
            "latency_ms": latency_ms,
            "reply": str(content or "").strip(),
        }
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
