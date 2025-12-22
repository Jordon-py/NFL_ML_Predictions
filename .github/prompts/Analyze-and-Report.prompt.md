---
agent: alfred-config.yml
---

## Primary Objective:
Perform a comprehensive audit and enhancement of the codebase, focusing on both the FastAPI backend and React frontend, emphasizing:

- Simplified, maintainable logic

- Thorough documentation

- Elimination of redundant or unused code

- Enhanced integration between ML predictions and frontend visualization

### 🔍 Target Files for Review

Copilot, review these files carefully and prioritize their analysis:

File Path	Type	Purpose
/backend/main.py	FastAPI Entry Point	API routing, startup logic, dependency injection
/backend/build_csv_datasets.py	Data Preprocessing	Dataset generation and I/O
/backend/train_models.py	ML Pipeline	Model training, saving, prediction logic
/frontend/src/	React Frontend	Display of predictions, probabilities, and user interface
/maintenance.md	Maintenance Log	Track errors, optimizations, and suggested improvements

📋 Step-by-Step Instructions
- ALWAYS CHECK AND UPDATE: 'NFL_ML_Predictions\alfred.log.md'
1. Static Analysis and Documentation

At the top of each file, insert a clear Doc Header:

# File: {{filename}}
# Purpose: {{short description}}
# Functions: {{list of function names and line number}}
# Variables: {{key variables and line number}}
# Interacts With: {{cross-file dependencies}}


Ensure all functions and major variables are documented inline with concise docstrings.

2. Function and Variable Mapping

Create a list of all functions and variables in each file.

Identify:

Unused or duplicate functions/variables

Functions not being called but defined (e.g., unused ML probability handlers)

Data not being passed from backend → frontend (especially prediction probabilities)

3. Simplification and Optimization

Inspect each key file for overly complex logic, especially:

Nested loops or redundant conditionals

Repeated data processing logic

Overly verbose React state management

Simplify by:

Abstracting repeated logic into helper functions

Ensuring modular, readable structure

Improving asynchronous handling (async/await in FastAPI, hooks in React)

4. Codebase Sanitation

Detect and list:

“Bloat” or redundant files

Unused imports, dead code, or test artifacts

Old notebooks, checkpoints, or logs no longer used

Remove or archive with a note in maintenance.md.

5. Machine Learning Usage Check

Identify if predicted probabilities or other outputs are computed but not displayed.

Suggest:

Where to expose them in the frontend (e.g., new UI components or cards).

Backend → Frontend data handling changes (React API hooks).

Example:

If predict_proba() is used in train_models.py but not rendered, recommend adding a React component to visualize confidence scores.

6. Error and Runtime Analysis

Run a static error check for:

Syntax issues

Possible runtime exceptions

Misaligned imports or missing dependencies

Log all findings in /maintenance.md in this format:

## [File: main.py | Line: 120]
- Issue: Missing async keyword in endpoint definition.
- Fix: Add `async def` and ensure awaitable I/O.
- Syntax Example:
    async def get_predictions(...):
        result = await model.predict(...)

7. Suggestion Evaluation Loop

For each issue found, generate two or more possible solutions.

Compare them logically (simplicity, maintainability, readability).

Present only the best final suggestion.

Include reasoning in the maintenance.md file:

### Resolution Summary:
Two possible fixes were compared; selected approach due to reduced complexity.

8. Maintenance and Reporting

Maintain /maintenance.md with:

File name, issue, and fix

Syntax examples for each fix

Description of what was not working or used inefficiently

A “To-Implement” list for missing features or unused outputs

Example Entry:

## File: train_models.py
- Issue: Probabilities calculated but unused.
- Suggested Fix: Send 'probabilities' in API response to frontend.
- Frontend Action: Display confidence percentages beside predictions.

9. Final Output Requirements

Copilot must output:

✅ Updated and cleaned code (FastAPI + React)

📜 Updated documentation headers and inline comments

🧩 maintenance.md log with:

Found issues, fixes, and line numbers

Syntax examples for fixes

Summary of unused or missing implementations

💡 Suggestions for improved data visibility and user experience in the React frontend

🧰 Notes for Copilot Context

Use best practices from PEP8 (Python) and ESLint (React).

Assume access to full repository.

Output must be deterministic and reproducible.

Never remove business-critical functions without first consolting Dev if not in maintenance.md. add a section for ai to directly addres dev and a section that it checks for user responses and a way to keep track of problems and user responses