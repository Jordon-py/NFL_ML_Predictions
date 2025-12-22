---
trigger: always_on
---

# Concise, Powerful Rules for Your AI Coding Assistant (v2)

Contract-First Always
Every boundary crossing (UI ↔ API ↔ model) must have a named schema + example request/response.

Diff-First, Minimal Change
Prefer small patches over rewrites. If a rewrite is needed, justify it and isolate it.

Clarity With Intent
Names carry meaning; comments explain why. Complex blocks get a short intent header.

End-to-End Dataflow Thinking
Trace: React state → API call → validation → core logic/model → response → UI render.

Consistency Across the Stack
React: functional + hooks/Context (no Redux). JS: async/await. Python: Pydantic models + small helpers. CSS: semantic LCH tokens.

Observability by Default
Structured logs, consistent error shapes, request context. Never log secrets.

Error-First Debug Loop
Observe → reproduce → explain → fix → validate. No symptom patches without a root-cause hypothesis.

Tests or Smoke Checks for Changes
Every change ships with a verification step: unit test, integration test, or explicit curl/build command.

Docs Are a Build Product
If contracts/logic change, update docs in the same change: docs/api, docs/dataflow, file headers where relevant.

Automation With Guardrails
Automate repetitive work, but show exactly what changed. Destructive actions require dry-run + explicit approval.

Security Hygiene, Always
Validate inputs, safe CORS, sane rate limits on risky endpoints, and keep .env and secrets out of outputs/logs.

Future-Proof Modularity
Keep components/helpers small and composable. Prefer clean interfaces over clever coupling.