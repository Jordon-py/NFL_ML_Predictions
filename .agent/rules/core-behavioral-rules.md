---
trigger: always_on
---

# Core Behavioral Rules

Rule 1 — Codebase Awareness:
Continuously maintain a full understanding of the project structure, including all front-end (React.jsx) and back-end files.
Always re-analyze dependencies and imports when the codebase changes.

Rule 2 — Dataflow Mapping:
Automatically generate and maintain dataflow.md in the project root.
Map all API endpoints, components, state flows, and key data transformations (inputs → processing → outputs).
Update this file after each major change.

Rule 3 — Artifact Management:
Maintain an artifacts/ directory with:

state.log → snapshot of the app state and version

last_5_tasks.md → summary of the last five actions performed by the LLM

next_5_tasks.md → planned actions or suggestions

important_info.md → key architecture notes or insights

screenshots/ (optional) → saved UI references if integrated with visual tools

Keep all artifacts synchronized and human-readable.

Rule 4 — Task Transparency:
Before starting any task, append it to next_5_tasks.md.
After completing it, move it to last_5_tasks.md with a short summary of what changed and why.

Rule 5 — Minimal Code Change Principle:
When modifying user code, make only the minimal necessary change to solve the issue.
Prioritize precision and simplicity over cleverness or complexity.

Rule 6 — Educational Code Style:
Whenever code must be complex, add educational inline comments that explain the logic clearly, focusing on clarity for human learning.
Each file should be self-explanatory for other developers.

Rule 7 — Documentation Discipline:
If a file lacks a top-level docstring or comment block, add one.
If one exists, update it to reflect new changes or dependencies.

Rule 8 — Context Preservation:
Before suggesting or implementing changes, read the entire file and related imports to maintain architectural and functional consistency.
Cross-reference changes against dataflow.md.

Rule 9 — Verification and Sanity Checks:
After generating or updating code, validate syntax and logic, simulate expected behavior, and highlight potential bugs or side effects in comments.

Rule 10 — Collaborative Awareness:
Always write explanations and commit messages that assume collaboration.
Communicate decisions, rationale, and next steps as if working with another developer reviewing your PR.