# Architecture

This document summarizes the architecture and runtime flow of the S12 agent.

## System Context
```mermaid
graph TB
  User[User CLI] --> Main[main.py interactive loop]
  Main --> AgentLoop[agent/agent_loop3.py]
  AgentLoop --> MCP[MultiMCP dispatcher]
  MCP --> MCPServers[MCP servers via stdio/sse]
  MCPServers --> LocalRAG[mcp_servers/mcp_server_2.py]
  LocalRAG --> Docs[documents + faiss_index]

  AgentLoop --> LLM[LLM via ModelManager]
  LLM --> Gemini[Gemini API]
  LLM --> Ollama[Ollama HTTP]
  AgentLoop --> Memory[memory/memory_search.py]
  Memory --> MemoryIndex[faiss_index + memory_indexer]

  AgentLoop --> Sandbox[action/executor.py]
  Sandbox --> MCP
```

## Component Breakdown
```mermaid
graph LR
  subgraph Entry
    Main[main.py]
  end

  subgraph Orchestration
    Loop[AgentLoop]
    Ctx[ContextManager]
    Sess[AgentSession]
  end

  subgraph Reasoning
    Perception[Perception]
    Decision[Decision]
    Summarizer[Summarizer]
  end

  subgraph Execution
    Exec[execute_step.py]
    Sandbox[action/executor.py]
  end

  subgraph BrowserAutomation
    BrowserAgent[browser_agent/browser_agent.py]
    BrowserMemory[memory/browser_agent_memory.py]
  end

  subgraph Tooling
    Multi[MultiMCP]
    MCPServer[mcp_server_2.py]
  end

  subgraph Data
    MemorySearch[MemorySearch]
    Prompts[prompts/*.txt]
    Config[config/*.yaml, models.json]
    SandboxState[action/sandbox_state/*.json]
  end

  Main --> Loop
  Loop --> Ctx
  Loop --> Sess
  Loop --> Perception
  Loop --> Decision
  Loop --> Summarizer
  Loop --> Exec
  Exec --> Sandbox
  Sandbox --> Multi
  Multi --> MCPServer

  Loop --> BrowserAgent
  BrowserAgent --> BrowserMemory

  Loop --> MemorySearch
  Perception --> Prompts
  Decision --> Prompts
  Summarizer --> Prompts
  Loop --> Config
  Sandbox --> SandboxState
```

## BrowserAgent Upgrades
- **LLM-guided element ranking:** ranks similar interactive elements before click/input.
- **Visited page memory:** tracks visited page hashes and failed actions to avoid loops.
- **Form flow chaining:** follows multi-stage forms across redirects and submit steps.
- **Persistent state:** saved under `memory/browser_agent_state/` per run/session.

## Core Runtime Flow
```mermaid
sequenceDiagram
  participant User
  participant Main
  participant Loop as AgentLoop
  participant BA as BrowserAgent
  participant BM as BrowserMemory
  participant Perc as Perception
  participant Dec as Decision
  participant Exec as Executor
  participant MCP as MultiMCP
  participant Tool as MCP Tool
  participant Sum as Summarizer

  User->>Main: enter query
  Main->>Loop: run(query)
  Loop->>Perc: build_perception_input + run()
  Perc-->>Loop: perception output
  Loop->>Dec: build_decision_input + run()
  Dec-->>Loop: plan_graph + code_variants
  Loop->>Exec: execute_step_with_mode
  Exec->>BA: run(instruction)
  BA->>BM: load_state(session_id or run_id)
  BA->>MCP: get_interactive_elements / call tool
  MCP-->>BA: result
  BA->>BM: save_state(updated_memory)
  Exec->>MCP: function_wrapper(tool)
  MCP->>Tool: call tool
  Tool-->>MCP: result
  MCP-->>Exec: result
  Exec-->>Loop: step result
  Loop->>Perc: perception on step_result
  Perc-->>Loop: updated perception
  alt goal achieved
    Loop->>Sum: summarize()
    Sum-->>Loop: final summary
  else continue
    Loop->>Dec: replan/next step
  end
```

## Data Flow
```mermaid
flowchart TD
  Query[User Query] --> PInput[build_perception_input]
  PInput --> Perception
  Perception --> POut[Perception JSON]
  POut --> DInput[build_decision_input]
  DInput --> Decision
  Decision --> Plan[plan_graph + code_variants]
  Plan --> Execute[execute_step_with_mode]
  Execute --> BA[BrowserAgent]
  BA --> BM[BrowserMemoryStore]
  BM --> BA
  Execute --> Results[step result + globals update]
  Results --> PInput2[build_perception_input(step_result)]
  PInput2 --> Perception

  Results --> Context[ContextManager graph/globals]
  Context --> SummaryInput
  SummaryInput --> Summarizer
  Summarizer --> FinalAnswer
```

## Retrieval Intelligence
```mermaid
flowchart LR
  Query[User Query] --> Mem[MemorySearch]
  Query --> Rewrite[Query rewriting]
  Rewrite --> RAG[search_stored_documents_rag]
  Mem --> Rank[Hybrid scoring: fuzz + NER]
  RAG --> Faiss[FAISS index]
  RAG --> Rerank[Lightweight reranking]
  Faiss --> Context[ContextManager.globals]

  Rank --> Context
  Rerank --> Context
  Context --> Decision[Decision plan]
  Context --> Perception[Perception input]
```

## Memory Stratification
```mermaid
graph TD
  Raw[Raw inputs & outputs] --> Session[AgentSession snapshots]
  Session --> ShortTerm[ContextManager.globals + plan graph]
  Raw --> LongTerm[Memory index (faiss)]
  LongTerm --> Recall[MemorySearch]
  Recall --> ShortTerm
```

## Feedback + Learning Loop
```mermaid
sequenceDiagram
  participant Exec as Execution
  participant Perc as Perception
  participant Dec as Decision
  participant Ctx as ContextManager
  participant Sum as Summarizer

  Exec-->>Ctx: step result + globals
  Ctx-->>Perc: snapshot_type=step_result
  Perc-->>Dec: updated perception
  Dec-->>Exec: new code_variants + next_step_id
  Dec-->>Ctx: updated plan_graph
  alt goal achieved
    Perc-->>Sum: summarize
    Sum-->>Ctx: attach summary + persist
  end
```

## Observability + Evaluation
```mermaid
graph LR
  Logs[log_step/log_error/log_json_block] --> Console[STDOUT/STDERR]
  Snapshots[AgentSession snapshots] --> MemoryStore[memory/ + action/sandbox_state]
  Plans[ContextManager graph] --> FinalPlan[save_final_plan]
```

## Failure Handling + Recovery
- **Step failures:** `StepExecutionTracker` limits retries per step and forces a replan to `ROOT` after repeated failures.
- **Tool timeouts:** timeouts trigger a bounded retry; if repeated, the loop replans from `ROOT`.
- **Planner failures:** invalid or erroring Decision output retries a limited number of times, then summarizes partial results.
- **Retrieval failures:** retrieval-related errors return a clarification request to the user.
- **Early summarize:** Perception can route to summarizer when goals are met or further steps are unhelpful.

## Scaling Plan
```mermaid
flowchart LR
  CLI[CLI / UI] --> API[Agent API]
  API --> Queue[Job Queue]
  Queue --> Workers[Executor Workers]
  Workers --> MCP[MultiMCP]
  MCP --> Tools[MCP Tools]
  Workers --> Store[Session Store]
  Store --> Analytics[Metrics + Traces]
```
- **Split processes:** separate agent API, MCP servers, and indexer services.
- **Queue execution:** move `execute_step` to background workers.
- **Cache retrieval:** memoize RAG results by query + corpus hash.
- **Batch indexing:** periodic embeddings, avoid reindex on startup.

## Hardening Plan
```mermaid
flowchart TD
  Input[User input] --> Validate[Input validation]
  Validate --> Sandbox[action/executor.py]
  Sandbox --> Guard[Resource limits]
  Sandbox --> ToolGate[Tool allow-list]
  ToolGate --> MCP[MCP tools]
  MCP --> External[Network/Files]
```
- **Sandbox limits:** reduce `SAFE_BUILTINS`, tighten `ALLOWED_MODULES`, enforce memory/CPU/time limits.
- **Tool allow-list:** scope tools to task category; deny network/file side effects by default.
- **Secrets:** move API keys to a secret manager; rotate regularly.
- **Audit trail:** append immutable logs for decisions, tool calls, and errors.

## Evaluation Plan
```mermaid
flowchart LR
  Golden[Golden queries] --> Run[Batch run]
  Run --> Metrics[Metrics + scores]
  Metrics --> Dash[Dashboards]
  Run --> Review[Human review]
```
- **Golden set:** fixed tasks with expected outputs for regression.
- **Heuristic signals:** track `original_goal_achieved`, retry counts, tool error rates.
- **Retrieval metrics:** recall@k, MRR, source coverage.
- **Cost + latency:** per-step timing and model cost.

## Explicit Evaluation Signals
```mermaid
flowchart TD
  Perception[Perception output] --> Signals[Heuristic signals]
  Execution[Execution result] --> Signals
  Signals --> Status[Step/Session status]
  Status --> Summary[Summarizer input]
```
- **Heuristic examples:** completion flags (`original_goal_achieved`, `local_goal_achieved`), route switching, retry limits, and error states.
- **Why it matters:** these signals gate summarization, replanning, and termination.

## Safety / Trust Boundaries
```mermaid
flowchart TD
  Code[LLM-generated code] --> Sandbox[action/executor.py]
  Sandbox -->|Allowed modules| PySafe[SAFE_BUILTINS + ALLOWED_MODULES]
  Sandbox -->|Tool calls| MCP[MultiMCP]
  MCP --> MCPServer[mcp_server_2.py]
  MCPServer --> External[HTTP: Ollama/Trafilatura/FAISS/Files]
```

## Responsibilities Summary
- `main.py` wires config, initializes `MultiMCP`, and runs the interactive loop.
- `agent/agent_loop3.py` orchestrates perception -> decision -> execution -> perception.
- `agent/contextManager.py` stores plan graph, globals, and step state.
- `agent/agentSession.py` stores snapshots for observability and replay.
- `action/executor.py` runs sandboxed tool-calling code.
- `mcp_servers/mcp_server_2.py` provides local RAG tools (FAISS + document parsing).

## Minor Reasoning Role Clarification
- **Perception:** interprets state and decides whether to summarize or continue.
- **Decision:** creates/updates the plan graph and tool-calling code variants.
- **Summarizer:** composes the final response when goal is achieved.
