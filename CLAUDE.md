# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Python Backend
```bash
# Install dependencies
uv sync

# Run CLI
uv run main.py "your research question"
uv run main.py --interactive
uv run main.py --graph-format sp --max-step-num 3 --debug "query"

# Run FastAPI server
uv run server.py --port 8000 --reload

# Format / lint
make format   # black with preview
make lint     # check formatting

# Tests
make test                              # run all tests
uv run pytest tests/test_api.py -v    # single test file
make coverage                         # coverage report
```

### Web Frontend
```bash
cd web
pnpm install
pnpm dev      # development server (port 3000)
pnpm build
pnpm check    # lint + TypeScript check
```

### LangGraph Studio (debugging)
```bash
make langgraph-dev
```

## Architecture

**StackPlanner** is a multi-agent research system. A user query flows through these layers:

1. **Graph layer** (`src/graph/`) — LangGraph state machines define the execution flow. `builder.py` constructs graphs; `nodes.py` and `sp_nodes.py` implement node logic. Graph formats: `base`, `sp`, `sp_xxqg`, `xxqg`.

2. **Central Agent** (`src/agents/CentralAgent.py`) — The main decision-maker. Chooses one of five actions: `THINK`, `REFLECT`, `SUMMARIZE`, `DELEGATE`, `FINISH`. Manages `MemoryStack` to track execution history.

3. **Sub-agents** (`src/agents/`) — Specialized executors: `ResearcherAgent` (search/retrieval), `CoderAgent` (code execution), `CommonReactAgent` (ReAct reasoning). Orchestrated by `SubAgentManager.py`.

4. **Tools** (`src/tools/`) — Search (Tavily, DuckDuckGo, Brave, Arxiv), Python REPL, web crawling, RAG retrieval, TTS.

5. **Memory** (`src/memory/`) — `MemoryStack` with bounded size; entries store action type, agent type, content, timestamp.

6. **State** (`src/graph/types.py`) — `State` extends LangGraph's `MessagesState`. Key fields: `user_query`, `current_plan`, `observations`, `memory_stack`, `final_report`, `background_investigation_results`, `need_human_interaction`.

7. **Prompts** (`src/prompts/`) — Markdown templates rendered via `template.py`. `central_agent.md` (17 KB) is the most critical.

8. **Server** (`src/server/`) — FastAPI with SSE for streaming agent output; Human-in-the-loop (HITL) endpoints.

9. **Web** (`web/`) — Next.js 15 + React 19 frontend; Zustand for state; SSE client for real-time streaming; MCP integration.

## Configuration

Copy `.env.example` → `.env` and `conf.yaml.example` → `conf.yaml`. Key settings:
- **LLM**: model name, API key/base URL. Supports OpenAI, Ollama, DeepSeek, Qwen, Gemini, Azure, Doubao.
- **Search**: `TAVILY_API_KEY`, `BRAVE_SEARCH_API_KEY`, or DuckDuckGo (no key needed).
- **Graph format**: `base` (default) or `sp` (optimized stack-planner algorithm).
- **RAG**: Ragflow endpoint/API key.

See `docs/configuration_guide.md` for per-provider examples.

## Key Files

| File | Purpose |
|---|---|
| `src/agents/CentralAgent.py` | Core decision logic (~42 KB) |
| `src/agents/SubAgentManager.py` | Sub-agent orchestration (~64 KB) |
| `src/graph/nodes.py` | Graph node implementations (~41 KB) |
| `src/graph/sp_nodes.py` | SP algorithm-specific nodes |
| `src/graph/builder.py` | Constructs LangGraph state machines |
| `src/graph/types.py` | `State` type definition |
| `src/prompts/central_agent.md` | Central agent prompt (~17 KB) |
| `workflow.py` | Top-level workflow entry point |
| `main.py` | CLI entry point |
| `server.py` | FastAPI server entry point |

## Special Features

- **Human-in-the-Loop**: Agent pauses at plan stage for user feedback; controlled via `need_human_interaction` in State.
- **Background Investigation**: Optional pre-execution web search run before planning.
- **Multiple output formats**: Standard report, PowerPoint (`src/ppt/`), Podcast (`src/podcast/`), Prose (`src/prose/`).
- **QA Benchmark pipeline**: Scripts for evaluating on PopQA, MATH, TriviaQA, GPQA, GSM8K benchmarks.
