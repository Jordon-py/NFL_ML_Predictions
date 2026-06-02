"""Public NFL Ollama agent facade.

Data shape:
- Input dataset: CSV loaded by `NFLMemory`, one NFL game per row with identity
  columns and model features.
- Input chat question: plain text from the premium API or interactive CLI.
- Output: plain-text answer strings, structured explanation dictionaries, or
  interactive console messages.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from backend.ollama.client import (
    FALLBACK_MODEL,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OllamaClient,
    chat_messages,
    explain_prediction,
    model_candidates,
    ollama_unavailable_reply,
)
from backend.ollama.memory import NFLMemory


log = logging.getLogger(__name__)


class NFLAgent:
    """
    Ask questions about NFL game data using Ollama.

    Loads the feature dataset through `NFLMemory`, builds compact data context,
    and sends bounded prompts to Ollama with model fallback support.
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        model: Optional[str] = None,
        host: Optional[str] = None,
    ):
        self.model = model or OLLAMA_MODEL
        self.host = host or OLLAMA_HOST
        self.client = OllamaClient(host=self.host, model=self.model).client
        self.memory = NFLMemory(csv_path=csv_path)

        self.csv_path = self.memory.csv_path
        self.df = self.memory.df
        self._data_summary = self.memory.data_summary

    def _summarize_data(self) -> str:
        """Compatibility wrapper for callers that used the old private method."""
        return self.memory.summarize_data()

    def _build_system_prompt(self) -> str:
        """Compatibility wrapper for the dataset-backed system prompt."""
        return self.memory.build_system_prompt()

    def _get_relevant_context(self, question: str) -> str:
        """Compatibility wrapper for bounded question-specific dataset context."""
        return self.memory.relevant_context(question)

    async def ask(self, question: str) -> str:
        """
        Ask a single question about the NFL data and return the answer string.

        Flow: system prompt plus relevant row preview -> Ollama chat -> text
        reply. If the primary model fails, environment and local fallback models
        are tried in order.
        """
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": f"{question}\n\nRelevant data:\n{self._get_relevant_context(question)}",
            },
        ]

        errors = []
        for idx, model in enumerate(model_candidates(self.model)):
            try:
                response = await self.client.chat(model=model, messages=messages)
                return (response.message.content or "").strip()
            except Exception as exc:
                err_msg = f"Model '{model}' failed: {exc}"
                errors.append(err_msg)
                log.warning(err_msg)
                candidates = model_candidates(self.model)
                if idx < len(candidates) - 1:
                    log.info("Trying next Ollama model: %s", candidates[idx + 1])

        return ollama_unavailable_reply(errors)

    async def explain_prediction(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a structured prediction explanation with this agent config."""
        return await explain_prediction(pred, host=self.host, model=self.model)

    async def chat(self) -> None:
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


async def chat() -> Any:
    """Compatibility interactive chat entrypoint."""
    agent = NFLAgent()
    await agent.chat()


__all__ = [
    "FALLBACK_MODEL",
    "NFLAgent",
    "OLLAMA_HOST",
    "OLLAMA_MODEL",
    "OllamaClient",
    "chat",
    "chat_messages",
    "explain_prediction",
]


if __name__ == "__main__":
    asyncio.run(chat())
