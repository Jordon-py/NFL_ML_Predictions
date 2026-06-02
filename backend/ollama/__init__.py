# ==========================================
# File: backend/ollama/__init__.py
# Role: Package initializer for ollama module.
# Exports: Premium AI/Ollama helper surface.
# ==========================================

from backend.ollama.llm_ollama import NFLAgent, chat_messages, explain_prediction

__all__ = ["NFLAgent", "chat_messages", "explain_prediction"]
