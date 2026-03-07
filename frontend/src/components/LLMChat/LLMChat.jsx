import React from "react";
import "./LLMChat.css";

/**
 * Placeholder while LLM integration is disabled.
 */
export default function LLMChat() {
  return (
    <section className="llm-chat card-shell" aria-live="polite">
      <header className="llm-chat__header">
        <div>
          <h3>Model Chat</h3>
          <p className="llm-chat__context">Temporarily disabled</p>
        </div>
        <span className="llm-chat__badge off">Offline</span>
      </header>
      <div className="llm-chat__messages">
        <div className="llm-chat__bubble assistant">
          <span className="llm-chat__role">system</span>
          <p>LLM chat is disabled for this release while core prediction workflows are stabilized.</p>
        </div>
      </div>
    </section>
  );
}
