// ==========================================
// File: frontend/src/components/LLMChat/LLMChat.jsx
// Role: React component for UI rendering.
// Input Data: Props (data and callbacks).
// Output Data: JSX markup.
// Dependencies: react, ../../api/client.js, ./LLMChat.css
// Notes: Presentation-focused component.
// ==========================================

/**
 * FILE: frontend/src/components/LLMChat/LLMChat.jsx
 * PURPOSE: Chat UI that sends the active prediction context to the backend LLM.
 * INPUT PROPS:
 *   - prediction: UnifiedPredictionResponse (flat fields for a single game).
 * OUTPUT / SIDE EFFECTS: Sends chat payloads to /llm/chat and renders responses.
 * DEPENDENCIES: React, client.js, LLMChat.css
 */

import React, { useEffect, useRef, useState } from "react";
import { chatLLM } from "../../api/client.js";
import "./LLMChat.css";

const MAX_MESSAGES = 12;

function compactObject(value) {
  if (!value || typeof value !== "object") return null;
  const entries = Object.entries(value).filter(
    ([key, value]) => {
      return value !== undefined && value !== null && value !== "";
    }
  );
  return entries.length ? Object.fromEntries(entries) : null;
}

function buildPredictionContext(prediction) {
  if (!prediction) return null;
  const home = prediction.home_team;
  const away = prediction.away_team;
  const season = prediction.season;
  const week = prediction.week;

  return compactObject({
    game_id: prediction.game_id,
    home_team: home ? String(home).trim().toUpperCase() : undefined,
    away_team: away ? String(away).trim().toUpperCase() : undefined,
    season: Number.isFinite(Number(season)) ? Number(season) : undefined,
    week: Number.isFinite(Number(week)) ? Number(week) : undefined,
    home_score: prediction.home_score,
    away_score: prediction.away_score,
    home_win_probability: prediction.home_win_probability,
    away_win_probability: prediction.away_win_probability,
    prediction_source: prediction.prediction_source,
  });
}

function buildContextLabel(predictionContext) {
  if (!predictionContext) return "No prediction selected yet.";
  const home = predictionContext.home_team || "HOME";
  const away = predictionContext.away_team || "AWAY";
  const week = predictionContext.week ? `Week ${predictionContext.week}` : null;
  const season = predictionContext.season ? String(predictionContext.season) : null;
  const suffix = [week, season].filter(Boolean).join(" - ");
  return suffix ? `${home} vs ${away} - ${suffix}` : `${home} vs ${away}`;
}

export default function LLMChat({ prediction }) {
  const [messages, setMessages] = useState(() => [
    {
      role: "assistant",
      content:
        "Ask me about a matchup, the model output, or how to read the prediction.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [meta, setMeta] = useState({ used_llm: false, llm_model: null, error: null });
  const scrollRef = useRef(null);

  const predictionContext = buildPredictionContext(prediction);
  const contextLabel = buildContextLabel(predictionContext);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const sendMessage = async (text) => {
    const content = text.trim();
    if (!content || loading) return;

    const nextMessages = [
      ...messages,
      { role: "user", content },
    ];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);

    try {
      const trimmedMessages = nextMessages.slice(-MAX_MESSAGES);
      const response = await chatLLM({
        messages: trimmedMessages,
        prediction: predictionContext,
      });

      const reply =
        response?.reply?.toString().trim() ||
        "No response returned. Try again in a moment.";

      setMeta({
        used_llm: Boolean(response?.used_llm),
        llm_model: response?.llm_model ?? null,
        error: response?.error ?? null,
      });

      setMessages([
        ...nextMessages,
        { role: "assistant", content: reply },
      ]);
    } catch (error) {
      const message = error?.message || "Unable to reach the LLM.";
      setMeta({ used_llm: false, llm_model: null, error: message });
      setMessages([
        ...nextMessages,
        { role: "assistant", content: `Error: ${message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    sendMessage(input);
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage(input);
    }
  };

  const badgeText = meta.used_llm
    ? `LLM: ${meta.llm_model || "active"}`
    : "LLM: fallback";

  const quickPrompts = [
    "Explain the current prediction in plain language.",
    "What are the biggest risks to this prediction?",
    "How should I interpret the win probability?",
  ];

  return (
    <section className="llm-chat card-shell" aria-live="polite">
      <header className="llm-chat__header">
        <div>
          <h3>LLM Coach</h3>
          <p className="llm-chat__context">{contextLabel}</p>
        </div>
        <span className={`llm-chat__badge ${meta.used_llm ? "on" : "off"}`}>
          {badgeText}
        </span>
      </header>

      <div className="llm-chat__messages" ref={scrollRef}>
        {messages.map((msg, index) => (
          <div
            key={`${msg.role}-${index}`}
            className={`llm-chat__bubble ${msg.role}`}
          >
            <span className="llm-chat__role">{msg.role}</span>
            <p>{msg.content}</p>
          </div>
        ))}
        {loading && (
          <div className="llm-chat__bubble assistant loading">
            <span className="llm-chat__role">assistant</span>
            <p>Thinking...</p>
          </div>
        )}
      </div>

      <div className="llm-chat__quick">
        {quickPrompts.map((prompt) => (
          <button
            key={prompt}
            type="button"
            className="llm-chat__chip"
            onClick={() => sendMessage(prompt)}
            disabled={loading}
          >
            {prompt}
          </button>
        ))}
      </div>

      <form className="llm-chat__input" onSubmit={handleSubmit}>
        <textarea
          rows={2}
          placeholder="Ask about the matchup, context, or model output..."
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>

      {meta.error && <p className="llm-chat__error">{meta.error}</p>}
    </section>
  );
}
