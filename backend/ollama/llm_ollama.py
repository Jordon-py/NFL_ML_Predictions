# ==============================================================================
# File: backend/ollama/llm_ollama.py
# Role: LLM Integration Module for Ollama
#
# OVERVIEW FOR DEVELOPERS:
# This module provides a bridge between the NFL Prediction system and Ollama,
# a local LLM runner. It supports two primary modes of interaction:
#
# 1. Structured Inference (explain_prediction):
#    Used by the API to turn raw model numbers into human-readable JSON explanations.
#
# 2. Interactive Chat (chat):
#    A streaming interface that supports "Reasoning Models" (e.g., DeepSeek, Gemma 4).
#
# KEY CONCEPT: The "Thinking" Flow
# Modern reasoning models don't just give an answer; they "think" first.
# In the API stream, this appears as a 'thinking' field. Once the model finishes
# reasoning, it switches to the 'content' field for the final answer.
# This module handles that transition seamlessly to provide a clean UI experience.
# ==============================================================================

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv, find_dotenv
import httpx
import ollama

# Automatically find and load .env from the project root
load_dotenv(find_dotenv())
class OllamaClient:
    """
    A wrapper for the Ollama AsyncClient.

    It handles configuration via environment variables, allowing the same code
    to run locally (localhost:11434) or in a cloud environment via OLLAMA_BASE_URL.
    """
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None, timeout_s: Optional[float] = None):
        self.host = host or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma4:e4b").split(",")[0].strip()
        self.timeout_s = timeout_s or float(os.getenv("OLLAMA_TIMEOUT_S", "300"))
        self.client = ollama.AsyncClient(host=self.host, timeout=self.timeout_s)
        self.api_key = os.getenv("OLLAMA_API_KEY")

client = OllamaClient()

async def chat() -> Any:
    """Interactive chat loop demonstrating streaming and thinking capabilities."""
    client = OllamaClient()
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]

    while True:
        user_prompt = input("\nUser: ")
        if user_prompt.lower() in ("exit", "quit"):
            break

        messages.append({'role': 'user', 'content': user_prompt})

        print("\nAssistant: ", end='', flush=True)
        # State tracking for reasoning models
        in_thinking = False # True if the model is currently emitting 'thinking' tokens
        full_content = ''  # Final answer accumulation
        full_thinking = '' # Reasoning process accumulation

        try:
            stream = await client.client.chat(
                model=client.model,
                messages=messages,
                stream=True,
            )

            async for chunk in stream:
                # 1. Handle the 'thinking' field (Reasoning/Chain-of-Thought)
                if hasattr(chunk.message, 'thinking') and chunk.message.thinking:
                    if not in_thinking:
                        in_thinking = True
                        print('\n[Thinking...]', end='', flush=True)
                    print(chunk.message.thinking, end='', flush=True)
                    full_thinking += chunk.message.thinking

                # 2. Handle the 'content' field (The actual answer)
                elif chunk.message.content:
                    # If we were thinking, we've now transitioned to the final answer
                    if in_thinking:
                        in_thinking = False
                        print('\n[Answer]: ', end='', flush=True)
                    print(chunk.message.content, end='', flush=True)
                    full_content += chunk.message.content

            # IMPORTANT: We must save the assistant's response back into the
            # conversation history so the model remembers what it said in the next turn.
            messages.append({
                'role': 'assistant',
                'content': full_content,
                'thinking': full_thinking
            })
            print("\n")

        except Exception as e:
            print(f"\nError during chat: {e}")
            break


asyncio.run(chat())
def _strip_code_fences(text: str) -> str:
    """
    LLMs often wrap JSON in markdown blocks like ```json { ... } ```.
    This helper strips those fences so json.loads() can parse the raw string.
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if "```" in t else t
        t = t.replace("json", "", 1).strip()
    if t.endswith("```"):
        t = t[:-3].strip()
    return t


def _normalize_host(host: str) -> str:
    return str(host or "").strip().rstrip("/")


def _parse_ollama_response_text(raw_text: str) -> Dict[str, Any]:
    """
    Parse either a normal JSON response or Ollama's newline-delimited streaming JSON.

    JUNIOR DEV NOTE:
    When 'stream=True', Ollama doesn't return one big JSON object. Instead, it returns
    many small JSON objects (one per line). This function detects if the input is
    a single object or a stream of objects, and merges the 'content' chunks into
    one final response.
    """
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("empty response body")

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        raise ValueError("expected top-level JSON object")
    except json.JSONDecodeError:
        chunks: list[str] = []
        last_obj: Dict[str, Any] | None = None

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            last_obj = obj

            message = obj.get("message")
            if isinstance(message, dict):
                chunk = message.get("content")
                if chunk is not None:
                    chunks.append(str(chunk))

        if last_obj is None:
            raise ValueError("no JSON objects found in response body")

        if chunks:
            merged = dict(last_obj)
            merged_message = dict(merged.get("message") or {})
            merged_message["content"] = "".join(chunks)
            merged["message"] = merged_message
            return merged

        return last_obj


async def _ollama_list_model_names(*, host: str, timeout_s: float) -> list[str]:
    host = _normalize_host(host)
    if not host:
        return []
    url = f"{host}/api/tags"
    timeout = httpx.Timeout(timeout_s)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
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
        preferred = ["gemma4:e4b", "gemma4:31b-cloud"]
    for m in preferred:
        if m in models:
            return m
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
        # We consume a single completed response; non-streaming avoids NDJSON parsing
        # issues on the common path, while we still keep a fallback parser below.
        "stream": False,
    }

    timeout = httpx.Timeout(timeout_s)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
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
        data = _parse_ollama_response_text(resp.text)
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
    host = host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = model or os.getenv("OLLAMA_MODEL", "qwen3-coder:480b-cloud")
    timeout_s = float(timeout_s) if timeout_s is not None else float(os.getenv("OLLAMA_TIMEOUT_S", "15"))

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


# Demo/test code has been moved to a separate script (e.g., demo_llm_ollama.py) to avoid accidental execution in production environments.
