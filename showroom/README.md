# Deep Researcher (Local Ollama Edition)

A fully local deep researcher using LangChain's DeepAgents framework with Ollama for LLM inference.

## Features

- 🔍 **Deep Research**: Multi-step research with sub-agent delegation
- 🤖 **Local LLM**: Uses Ollama with `gpt-oss:20b` model
- 📋 **Task Planning**: Automatic task decomposition with `write_todos`
- 🔄 **Sub-Agents**: Research and critique sub-agents for quality
- 🌐 **Web Search**: Tavily integration for real-time web research
- 🖥️ **Streamlit GUI**: Interactive web interface on port 8508

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) with `gpt-oss:20b` model installed
- [uv](https://github.com/astral-sh/uv) for environment management
- [Tavily API key](https://tavily.com/) for web search

## Quick Start

1. **Install Ollama model** (if not already done):
   ```bash
   ollama pull gpt-oss:20b
   ```

2. **Install dependencies**:
   ```bash
   cd showroom
   uv sync
   ```

3. **Configure environment** (edit `.env` if needed):
   ```bash
   # .env file contains:
   TAVILY_API_KEY="your-tavily-key"
   OLLAMA_MODEL="gpt-oss:20b"
   ```

4. **Run the application**:
   ```bash
   uv run streamlit run app.py --server.port 8508
   ```

5. **Open browser**: Navigate to `http://localhost:8508`

## Architecture

```
showroom/
├── app.py                    # Streamlit GUI
├── agent.py                  # Deep agent configuration
├── research_agent/
│   ├── __init__.py
│   ├── prompts.py            # All prompt templates
│   └── tools.py              # Tavily search + think tool
├── pyproject.toml            # Dependencies
└── .env                      # Configuration
```

## How It Works

1. **User Query**: Enter your research question
2. **Planning**: Agent creates a TODO list breaking down the research
3. **Sub-Agent Delegation**: Research sub-agents conduct focused searches
4. **Synthesis**: Main agent consolidates findings
5. **Critique**: Critique sub-agent reviews the report
6. **Final Report**: Comprehensive report with citations

## Configuration

Edit `.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `gpt-oss:20b` | Ollama model to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `MAX_CONCURRENT_RESEARCH_UNITS` | `3` | Max parallel sub-agents |
| `MAX_RESEARCHER_ITERATIONS` | `3` | Max research rounds |
| `RECURSION_LIMIT` | `1000` | Max agent steps |

## License

MIT License
