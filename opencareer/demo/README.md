# OpenCareer Demo

A multi-agent career assistant system built with a **judge-and-handoff** architecture: a Brain Agent analyzes user needs and routes to specialized agents (Work, Emotion), all powered by real DeepSeek API calls.

## Overview

OpenCareer is a comprehensive career assistant that helps job seekers with skill development, interview preparation, and emotional support through a coordinated multi-agent system.

### Key Features

- **4-Agent System**: Brain Agent (judge-and-handoff entry), Work Agent (career functions), Emotion Agent (emotional support), Log Agent (background info extraction)
- **YAML Prompt System**: All agent prompts managed via `PromptRegistry` from YAML files
- **3 Core SKILLs**: Learning Plan, Mock Interview, Emotion Support
- **Persona System**: Shared communication identity (`personas/base.yaml`) + agent-specific additions
- **Emotion Recognition Integration**: Emotion analysis pipeline extracted from dedicated skill
- **Shared LLM Client**: All agents share a single `LLMClient` for DeepSeek API calls

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    quick_dialogue.py                          │
│          (entry point — loads agents, routes responses)       │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                     Brain Agent                               │
│                (Judge-and-Handoff Entry)                      │
│   1. _analyze_demand() — LLM-based demand + emotion analysis │
│   2. _determine_target_agent() — priority-based routing      │
│      crisis/high intensity  ──────→ emotion_agent            │
│      job_related            ──────→ work_agent               │
│      mixed                  ──────→ emotion_agent (first)    │
│      unknown/low confidence ──────→ clarify with user        │
│   3. _build_dialogue_response() — persona-based reply        │
│   4. Returns {target_agent, demand_analysis, response}       │
└───────────────────────────┬──────────────────────────────────┘
                            │ handoff via target_agent
              ┌─────────────┴─────────────┐
              ▼                           ▼
    ┌──────────────────┐      ┌──────────────────┐
    │   Work Agent     │      │  Emotion Agent    │
    │ (career advice)  │      │(emotional support)│
    │                   │      │                   │
    │- Conversational   │      │- Support response │
    │  advice           │      │  w/ intensity     │
    │- MCP skill calls  │      │  calibration      │
    │  (mock)           │      │- Crisis detection │
    └──────────────────┘      └──────────────────┘
              │                        │
              └──────────┬─────────────┘
                         ▼
              ┌──────────────────┐
              │   Log Agent      │
              │ (fire-and-forget)│
              └──────────────────┘
```

### Routing Rules

Every user message goes through **full re-analysis** by Brain Agent:

1. **crisis** / `support_intensity == "high"` → EmotionAgent (safety first)
2. `suggested_action == "emotional_support"` → EmotionAgent
3. `demand_type == "job_related"` → WorkAgent
4. `demand_type == "mixed"` → EmotionAgent first (can handoff to WorkAgent)
5. Low confidence / unknown → clarify with user

## Project Structure

```
opencareer/
├── agents/                    # 4 Intelligent Agents
│   ├── base_agent.py         # Agent base class
│   ├── llm_client.py         # Shared DeepSeek API client
│   ├── conversation_context.py # Shared conversation state
│   ├── persona.py            # Persona system prompt builder
│   ├── brain/                # Brain Agent (judge-and-handoff entry)
│   ├── work_agent/           # Work Agent (career functions)
│   ├── emotion_agent/        # Emotion Agent (support)
│   └── log_agent/            # Log Agent (info extraction, fire-and-forget)
├── prompts/                   # YAML prompt system
│   ├── registry.py           # PromptRegistry — dot-notation key access
│   ├── personas/             # Shared + agent-specific persona prompts
│   │   ├── base.yaml         # Core identity, communication principles
│   │   ├── brain.yaml        # BrainAgent-specific instructions
│   │   ├── work.yaml         # WorkAgent-specific instructions
│   │   └── emotion.yaml      # EmotionAgent-specific instructions
│   ├── system/               # Agent system prompts
│   │   ├── emotion_analysis.yaml    # Emotion analysis (from emotion-recognition)
│   │   └── emotion_agent_prompt.yaml # Emotion agent behavior guidelines
│   └── dialogues/            # Dialogue prompt templates
├── skills/                   # SKILLs (modular functions)
│   ├── base_skill.py         # SKILL base class
│   ├── skill_registry.py     # SKILL registry
│   ├── learning_plan/        # Learning Plan SKILL
│   ├── mock_interview/       # Mock Interview SKILL
│   └── emotion_support/      # Emotion Support SKILL
├── mcp/                      # MCP Server (code exists, not active in demo)
│   ├── server.py             # FastAPI MCP Server
│   ├── skill_loader.py       # SKILL dynamic loader
│   └── tools.py              # Tool definitions
├── memory/                   # Memory System (stub implementations)
│   ├── memory_manager.py     # Unified memory manager
│   ├── vector_memory.py      # ChromaDB vector storage (stub)
│   └── structured_memory.py  # SQLite/JSON structured storage (stub)
├── scheduler/                # Task Scheduler (configured, not active)
│   ├── task_scheduler.py     # dramatiq + Redis scheduler
│   └── dramatiq_worker.py    # Worker implementation
└── tests/                    # (planned — not yet created)
    ├── test_agents.py
    ├── test_skills.py
    └── test_integration.py
```

## Quick Start

### Prerequisites

- Python 3.9+
- DeepSeek API key

### Installation

1. **Navigate to the demo directory:**
   ```bash
   cd demo
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your DeepSeek API key
   ```

### Running the Interactive Demo

```bash
python quick_dialogue.py
```

This starts an interactive terminal session with the full multi-agent system:
- Brain Agent opens with a personalized AI greeting
- Every message is analyzed by DeepSeek API and routed to the appropriate agent
- Supports career advice, interview prep, emotional support, and mixed requests

Available commands:
- `quit`, `exit`, `q` — End conversation
- `help`, `?` — Show example prompts

### Main Entry Point

| Script | Description |
|--------|-------------|
| `quick_dialogue.py` | **Main entry point** — full multi-agent interactive dialogue |

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# DeepSeek API (required)
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com
CHAT_MODEL=deepseek-chat
CHAT_TEMPERATURE=0.7
CHAT_MAX_TOKENS=2000

# Redis (for dramatiq — not active in demo)
REDIS_URL=redis://localhost:6379/0

# ChromaDB (vector memory — stub implementation)
CHROMA_DB_PATH=./data/chromadb

# SQLite (structured memory — stub implementation)
SQLITE_DB_PATH=./data/opencareer.db

# MCP Server (server code exists, not started in demo)
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8000
```

## Core Components

### 1. Brain Agent
- **Role**: Judge-and-handoff entry point (NOT a multi-agent orchestrator)
- Analyzes user needs + emotion state via DeepSeek API
- Routes to a **single** target agent based on priority rules
- Generates persona-based dialogue responses
- All messages go through Brain Agent — no direct routing between downstream agents

### 2. LLM Client
- Shared `LLMClient` instance across all agents
- Two calling modes:
  - `chat()` — free-form text generation
  - `chat_json()` — structured JSON output (for demand analysis)
- Reads config from environment variables

### 3. Prompt System
- **PromptRegistry**: Loads all prompts from YAML files, accessible via dot notation
  - `_registry.get("system.emotion_analysis.system_prompt")` → YAML content
  - `_registry.get("personas.base.agent_identity")` → section from persona YAML
- **Persona System**: Base identity (`base.yaml`) + agent-specific additions layered together

### 4. Work Agent
- Routes career-related requests to SKILLs (via mock MCP calls in demo)
- Has built-in emotional distress detection (reads `support_intensity` from context)
- Calibrates response tone based on `support_intensity` level

### 5. Emotion Agent
- Detects emotional state from demand analysis
- Uses `emotion_agent_prompt.yaml` for response calibration
- Support intensity levels: `low` / `medium` / `high` / `crisis`
- Validates emotions before offering advice (crisis → pause all work tasks)

### 6. Log Agent
- Runs in background (fire-and-forget) for all requests
- Extracts structured information from conversations
- Stores extracted data in memory system (stub)

### 7. SKILLs
- **Learning Plan**: Creates personalized learning paths (mock response in demo)
- **Mock Interview**: Conducts practice interviews with feedback (mock response in demo)
- **Emotion Support**: Provides emotional support strategies (mock response in demo)

### 8. MCP Server
- FastAPI-based server for dynamic SKILL discovery and execution
- Code exists at `opencareer/mcp/server.py` but **not started during demo**
- Demo uses mock MCP responses instead

### 9. Memory System
- **Vector Memory**: ChromaDB for semantic conversation search (stub)
- **Structured Memory**: SQLite/JSON for user profiles and skills (stub)
- **Memory Manager**: Unified interface (not connected in demo)

### 10. Task Scheduler
- **dramatiq + Redis**: Configured for async background task processing
- **Not active** in current demo — configuration exists in `.env` only

## Key Design Decisions

1. **Judge-and-handoff over orchestration**: Brain Agent does NOT call multiple agents and merge responses. It judges the demand and routes to exactly one specialized agent.
2. **Every turn re-analyzed**: Each user message triggers a fresh LLM-based demand + emotion analysis — no cached routing decisions.
3. **Direct API over LangChain**: Agents call DeepSeek API directly via `aiohttp` instead of using LangChain, reducing dependency complexity.
4. **YAML-driven prompts**: All prompts are managed as YAML files through `PromptRegistry`, enabling prompt edits without touching Python code.
5. **Priority-based emotion routing**: Emotional safety takes precedence — crisis/high-intensity messages always route to EmotionAgent first, even if the content is also job-related.

## Development

### Adding New Prompts

1. Add a YAML file in `prompts/system/` or edit existing ones
2. Access via `_registry = get_global_registry()` then `_registry.get("path.to.key")`

### Adding New SKILLs

1. Create a new directory in `skills/`
2. Add a `skill.py` implementing `BaseSkill`
3. Include a `SKILL.md` with metadata

### Adding New Agents

1. Create a new directory in `agents/`
2. Extend `BaseAgent` class
3. Implement `process_user_request()` method
4. Register with the Brain Agent's agent registry

### Adding New Demo Scripts

New dialogue scripts should follow the established pattern:
1. Call `setup_real_agents()` to initialize all agents with shared LLM client
2. Call `brain_agent.get_opening_greeting()` for the first message
3. Call `process_request(brain_agent, user_input)` for each user turn
4. Route responses based on `target_agent` from the result

## License

MIT License — see LICENSE file for details.
