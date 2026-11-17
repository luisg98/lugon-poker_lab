# Lugon Poker Lab

Research environment for No-Limit Texas Hold'em analysis. Full pipeline from raw hand states to screen capture, hand evaluation, equity estimation, and LLM-assisted strategic guidance.

The engine is strictly decoupled from any advisory layer — heuristics, local models (Ollama), or cloud LLMs only receive facts the engine has confirmed.

## Architecture

JSON hand state → validator → evaluator → equity solver → pot odds/EV. The vision layer (screen capture / synthetic calibration) produces the same `HandState` without touching the math. Historical sequences feed per-player statistical profiles with explicit uncertainty bounds. A pluggable advisor (heuristic, Ollama, OpenAI-compatible, Gemini) consumes a validated `DecisionContext`.

See [ARCHITECTURE.md](docs/ARCHITECTURE.md), [PROJECT_MAP.md](docs/PROJECT_MAP.md), and [WORKFLOWS.md](docs/WORKFLOWS.md).

## Status

v1.0.11 — active development. See [ROADMAP.md](ROADMAP.md) and [CHANGELOG.md](CHANGELOG.md).
