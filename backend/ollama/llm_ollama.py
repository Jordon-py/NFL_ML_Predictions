# ==============================================================================
# File: backend/ollama/llm_ollama.py
# Role: NFL Dataset Q&A Agent powered by Ollama + Gemma
#
# OVERVIEW:
#   A simple agent that loads an NFL game-features CSV and answers questions
#   about it using a Gemma model via Ollama (cloud or local).
#
# USAGE:
#   # As a library:
#   from backend.ollama.llm_ollama import NFLAgent
#   agent = NFLAgent()
#   answer = await agent.ask("Who had the best home record in 2024?")
#
#   # As a script (interactive chat):
#   python backend/ollama/llm_ollama.py
#
# CONFIG (via .env):
#   OLLAMA_BASE_URL  – Ollama server URL  (default: http://localhost:11434)
#   OLLAMA_MODEL     – Primary model name (default: gemma4:e4b)
#   OLLAMA_API_KEY   – Bearer token for cloud Ollama (optional)
#   OLLAMA_TIMEOUT_S – Request timeout in seconds (default: 15)
#   NFL_DATASET_PATH – Path to the game-features CSV (has a sensible default)
# ==============================================================================

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama
import pandas as pd
from dotenv import find_dotenv, load_dotenv

# ── Config ───────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv())

# Resolve the default dataset path relative to this file's location
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_CSV = _THIS_DIR.parent / "data" / "datasets" / "game_features_20260531_clean.csv"

log = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        log.warning("Invalid %s value; using %.1f", name, default)
        return default


OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b").split(",")[0].strip()
OLLAMA_TIMEOUT = _env_float("OLLAMA_TIMEOUT_S", 15.0)
FALLBACK_MODEL = "gemma4:e4b"  # Always available locally


def _ollama_unavailable_reply(errors: List[str]) -> str:
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
    ):
        self.host = host or OLLAMA_HOST
        self.model = model or OLLAMA_MODEL
        self.timeout_s = float(timeout_s) if timeout_s is not None else OLLAMA_TIMEOUT
        api_key = os.getenv("OLLAMA_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = ollama.AsyncClient(host=self.host, timeout=self.timeout_s, headers=headers)


class NFLAgent:
    """
    Ask questions about NFL game data using Ollama + Gemma.

    Loads a game-features CSV, builds a compact data summary, and uses it
    as context so the LLM can answer questions accurately.
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        model: Optional[str] = None,
        host: Optional[str] = None,
    ):
        # ── Model & client setup ─────────────────────────────────────────
        self.model = model or OLLAMA_MODEL
        self.host = host or OLLAMA_HOST
        self.client = OllamaClient(host=self.host, model=self.model).client

        # ── Load the NFL dataset ─────────────────────────────────────────
        path = csv_path or os.getenv("NFL_DATASET_PATH", str(_DEFAULT_CSV))
        self.csv_path = Path(path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"NFL dataset not found: {self.csv_path}")
        self.df = pd.read_csv(path)
        log.info("Loaded %s NFL games from %s", f"{len(self.df):,}", self.csv_path.name)

        # ── Pre-build a compact data summary for the system prompt ───────
        self._data_summary = self._summarize_data()

    # ── Data summary (sent to the model as context) ──────────────────────

    def _summarize_data(self) -> str:
        """
        Build a concise text summary of the dataset.

        WHY: Sending 242 raw columns as JSON would blow up the context window
        and confuse the model. Instead, we give it schema + key stats so it
        can reason about the data intelligently.
        """
        df = self.df
        teams = sorted(df["home_team"].dropna().unique()) if "home_team" in df.columns else []
        seasons = sorted(df["season"].dropna().unique()) if "season" in df.columns else []

        # Key columns the model should know about (human-readable subset)
        key_cols = [
            "season", "week", "game_id", "home_team", "away_team",
            "home_points_for", "away_points_for", "point_diff", "winner",
            "home_prior_win_pct_3", "home_prior_pf_avg_3", "home_prior_pa_avg_3",
            "away_prior_win_pct_3", "away_prior_pf_avg_3", "away_prior_pa_avg_3",
            "home_prior_off_epa_per_play_3", "away_prior_off_epa_per_play_3",
            "spread_line", "total_line", "game_type",
            "home_win_prob_spread", "away_win_prob_spread",
            "surface", "roof", "temp", "wind",
        ]
        # Only include columns that actually exist in this CSV
        available = [c for c in key_cols if c in df.columns]

        return (
            f"DATASET: NFL game features ({len(df):,} rows, {len(df.columns)} columns)\n"
            f"SEASONS: {seasons[0] if seasons else 'unknown'}-{seasons[-1] if seasons else 'unknown'}\n"
            f"TEAMS ({len(teams)}): {', '.join(teams)}\n"
            f"KEY COLUMNS: {', '.join(available)}\n"
            f"ALL COLUMNS: {', '.join(df.columns[:60])}... ({len(df.columns)} total)\n"
        )

    def _build_system_prompt(self) -> str:
        """System prompt that turns the LLM into an NFL data analyst."""
        return (
            "You are an NFL data analyst. You have access to a dataset of NFL games.\n"
            "Answer questions using ONLY the data described below.\n"
            "Be concise, use numbers and stats when possible.\n"
            "If a question can't be answered from this data, say so.\n\n"
            f"{self._data_summary}"
        )

    # ── Core Q&A method ──────────────────────────────────────────────────

    async def ask(self, question: str) -> str:
        """
        Ask a single question about the NFL data. Returns the answer string.

        Flow: builds a context-enriched prompt → sends to Ollama → returns text.
        If the primary model fails, retries with the local fallback model.
        """
        # Build a small data slice relevant to the question (top-level stats)
        # For specific team/season queries, we filter and include a preview
        context = self._get_relevant_context(question)

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": f"{question}\n\nRelevant data:\n{context}"},
        ]

        # Create a list of all models to try (primary, env options, and fallback)
        models_to_try = [self.model]
        env_models = [m.strip() for m in os.getenv("OLLAMA_MODEL", "").split(",") if m.strip()]
        for m in env_models:
            if m not in models_to_try:
                models_to_try.append(m)
        if FALLBACK_MODEL not in models_to_try:
            models_to_try.append(FALLBACK_MODEL)

        # Try models in order, falling back dynamically
        errors = []
        for idx, model in enumerate(models_to_try):
            try:
                response = await self.client.chat(model=model, messages=messages)
                return (response.message.content or "").strip()
            except Exception as e:
                err_msg = f"Model '{model}' failed: {e}"
                errors.append(err_msg)
                log.warning(err_msg)
                if idx < len(models_to_try) - 1:
                    log.info("Trying next Ollama model: %s", models_to_try[idx + 1])
                continue

        return _ollama_unavailable_reply(errors)

    def _get_relevant_context(self, question: str) -> str:
        """
        Extract a small, relevant slice of data based on the question.

        WHY: Instead of dumping the whole CSV, we look for team names or
        season numbers in the question and filter down to a manageable chunk.
        """
        df = self.df
        q_upper = question.upper()

        # Check if a specific team is mentioned
        if "home_team" not in df.columns:
            return "Dataset does not include a home_team column."

        teams = df["home_team"].dropna().unique()
        mentioned = [t for t in teams if t in q_upper]

        # Check if a specific season is mentioned
        seasons = df["season"].dropna().unique() if "season" in df.columns else []
        mentioned_seasons = [s for s in seasons if str(int(s)) in question]

        # Filter the data
        filtered = df
        if mentioned:
            filtered = filtered[
                (filtered["home_team"].isin(mentioned)) |
                (filtered["away_team"].isin(mentioned) if "away_team" in filtered.columns else False)
            ]
        if mentioned_seasons:
            filtered = filtered[filtered["season"].isin(mentioned_seasons)]

        # Pick display columns that exist
        display_cols = [
            "season", "week", "home_team", "away_team",
            "home_points_for", "away_points_for", "winner",
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]

        # Return a preview (max 30 rows to keep context manageable)
        preview = filtered[display_cols].head(30)
        summary = f"Showing {len(preview)} of {len(filtered)} matching games:\n"
        return summary + preview.to_string(index=False)

    async def explain_prediction(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        return await explain_prediction(pred, host=self.host, model=self.model)


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        t = parts[1] if len(parts) > 1 else t
        if t.strip().lower().startswith("json"):
            t = t.strip()[4:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def _build_explanation_prompt(pred: Dict[str, Any]) -> str:
    home = str(pred.get("home_team", "")).upper()
    away = str(pred.get("away_team", "")).upper()
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
  predicted_home_score: {pred.get("home_score")}
  predicted_away_score: {pred.get("away_score")}
  home_win_probability: {pred.get("home_win_probability")}
  prediction_source: {pred.get("prediction_source", "model")}
""".strip()


def _model_candidates(primary_model: str) -> List[str]:
    candidates = [primary_model]
    for model in os.getenv("OLLAMA_MODEL", "").split(","):
        model = model.strip()
        if model and model not in candidates:
            candidates.append(model)
    if FALLBACK_MODEL not in candidates:
        candidates.append(FALLBACK_MODEL)
    return candidates


async def _chat_once(
    *,
    messages: List[Dict[str, str]],
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    active_host = host or OLLAMA_HOST
    active_model = model or OLLAMA_MODEL
    active_timeout = float(timeout_s) if timeout_s is not None else OLLAMA_TIMEOUT
    client = OllamaClient(host=active_host, model=active_model, timeout_s=active_timeout).client
    errors: List[str] = []

    for candidate in _model_candidates(active_model):
        try:
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
            errors.append(f"{candidate}: {exc}")
            log.warning("Ollama chat failed for model %s: %s", candidate, exc)

    return {
        "ok": False,
        "model": active_model,
        "host": active_host,
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
    result = await _chat_once(
        host=host,
        model=active_model,
        timeout_s=timeout_s,
        messages=[{"role": "user", "content": _build_explanation_prompt(pred)}],
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
        parsed = json.loads(_strip_code_fences(content))
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
    result = await _chat_once(host=host, model=model, timeout_s=timeout_s, messages=chat_payload)
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


async def _nfl_agent_chat(self: NFLAgent) -> None:
    """Interactive REPL chat loop. Type 'exit' or 'quit' to stop."""
    print(f"\nNFL Agent | Model: {self.model} | {len(self.df):,} games loaded")
    print("Type your question (or 'exit' to quit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        answer = await self.ask(user_input)
        print(f"\nAgent: {answer}\n")


NFLAgent.chat = _nfl_agent_chat


async def chat() -> Any:
    """Compatibility interactive chat entrypoint."""
    agent = NFLAgent()
    await agent.chat()


# ── Entry point (only runs when executed directly) ───────────────────────
if __name__ == "__main__":
    agent = NFLAgent()
    asyncio.run(agent.chat())
