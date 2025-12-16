# DeepAgents Ollama Implementation - Task List

## Project Status: EXECUTION

---

## Phase 1: Project Setup & Infrastructure
- [x] Create `/showroom` folder structure
- [x] Initialize `pyproject.toml` with dependencies (deepagents, langchain, ollama, streamlit, tavily)
- [x] Create `.env` file for local configuration
- [x] Set up `uv` virtual environment

## Phase 2: Core Backend Implementation
- [x] Implement `research_agent/` module structure
  - [x] `__init__.py`
  - [x] `prompts.py` - Adapted prompts for local LLM
  - [x] `tools.py` - Tavily search + think tool
  - [x] `agent.py` - Main deep agent with sub-agents
- [x] Configure Ollama integration with `gpt-oss:20b`
- [x] Implement `create_deep_agent` with:
  - [x] Planning and task decomposition (`write_todos`)
  - [x] Context management (file system tools)
  - [x] Subagent spawning (research sub-agent)
- [x] Add recursion limit and max iteration controls

## Phase 3: Streamlit GUI Implementation
- [x] Create `app.py` - Main Streamlit application
- [x] Implement research phases:
  - [x] Query input phase
  - [x] Research execution phase (with to-do tracking)
  - [x] Results display phase
- [x] Add current step visualization
- [x] Add previous step display
- [x] Implement max attempts warning UI
- [x] Configure port 8508

## Phase 4: Quality & Reflection System
- [x] Implement critique sub-agent
- [x] Add reflection workflow for quality checking
- [x] Implement max attempts loop prevention
- [x] Add quality issue display in GUI

## Phase 5: Verification & Testing
- [x] Test local Ollama connectivity
- [x] Test web search functionality
- [x] Test sub-agent spawning
- [x] End-to-end research workflow test
- [x] GUI functionality testing

## Phase 6: Documentation & Finalization
- [x] Document implementation steps
- [x] Create README.md
- [x] Final code review and cleanup
