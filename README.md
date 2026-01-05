# DeepAgents Ollama - Local Research Agent

A fully local deep research agent built with **LangGraph** and **Ollama** for comprehensive web research and analysis. This implementation uses LangGraph's StateGraph architecture to create a sophisticated agentic workflow that can conduct multi-step research, reflect on findings, and generate comprehensive reports.

## 🚀 Features

- **🔍 Deep Research**: Multi-step research workflow with 15-20 web searches
- **🤖 Local LLM**: Uses Ollama with configurable models (default: `qwen3:14b`)
- **📋 Strategic Planning**: Automatic research planning with reflection phases
- **🔄 Agentic Workflow**: LangGraph StateGraph with conditional routing and tool binding
- **🌐 Web Search**: Tavily integration for real-time web research with full content fetching
- **🧠 Reflection System**: Built-in think_tool for strategic decision-making
- **🖥️ Streamlit GUI**: Interactive web interface with real-time progress tracking
- **⚡ Memory Management**: CUDA memory optimization for local GPU usage
- **🔒 Quality Control**: Answer validation with word count and citation requirements

## 🏗️ Architecture

This project implements a sophisticated agentic workflow using **LangGraph's StateGraph** pattern:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Agent Node    │───▶│   Tool Router    │───▶│   Tool Node     │
│                 │    │                  │    │                 │
│ • LLM reasoning │    │ • Conditional    │    │ • tavily_search │
│ • Tool calling  │    │   routing        │    │ • think_tool    │
│ • State updates │    │ • Flow control   │    │ • submit_answer │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                       │                       │
         │                       ▼                       │
         │              ┌──────────────────┐             │
         │              │   End Condition  │             │
         │              │                  │             │
         └──────────────│ • Answer accepted│◀────────────┘
                        │ • Max iterations │
                        │ • Force submit   │
                        └──────────────────┘
```

### Key Components

1. **StateGraph Architecture**: Uses LangGraph's `StateGraph` for workflow orchestration
2. **Tool Binding**: LLM bound to tools using `model.bind_tools()`
3. **Conditional Routing**: Smart routing based on agent responses and research progress
4. **State Management**: Typed state with message history and iteration tracking
5. **Memory Optimization**: Automatic CUDA memory clearing between operations

## 📦 Installation

### Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.ai/)** with models installed
- **[uv](https://github.com/astral-sh/uv)** for dependency management
- **[Tavily API key](https://tavily.com/)** for web search

### Quick Setup

1. **Install Ollama model**:
   ```bash
   ollama pull qwen3:14b
   # or other supported models: qwen3:8b, qwen3:30b, qwen3-coder:30b
   ```

2. **Clone and setup**:
   ```bash
   cd deepagents_ollama/showroom
   uv sync
   ```

3. **Configure environment**:
   ```bash
   # Edit .env file
   TAVILY_API_KEY="your-tavily-api-key"
   OLLAMA_MODEL="qwen3:14b"
   OLLAMA_BASE_URL="http://localhost:11434"
   ```

4. **Run the application**:
   ```bash
   uv run streamlit run app.py --server.port 8508 --server.headless false
   ```

5. **Access**: Open `http://localhost:8508`

## 🔧 Technical Implementation

### Core Libraries

- **[LangGraph](https://github.com/langchain-ai/langgraph)**: StateGraph workflow orchestration
- **[LangChain](https://github.com/langchain-ai/langchain)**: LLM abstraction and tool integration
- **[langchain-ollama](https://github.com/langchain-ai/langchain/tree/master/libs/partners/ollama)**: Ollama ChatModel integration
- **[Streamlit](https://streamlit.io/)**: Web interface framework
- **[Tavily](https://tavily.com/)**: Web search API with content fetching
- **[httpx](https://www.python-httpx.org/)**: Async HTTP client for web scraping
- **[markdownify](https://github.com/matthewwithanm/python-markdownify)**: HTML to Markdown conversion

### Agentic Workflow Details

#### 1. State Definition
```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    research_findings: List[str]
    iteration_count: int
```

#### 2. Tool Integration
```python
tools = [tavily_search, think_tool, submit_final_answer]
model_with_tools = model.bind_tools(tools)
```

#### 3. Conditional Routing Logic
```python
def should_continue(state: AgentState) -> str:
    # Route based on:
    # - Tool calls present → "tools"
    # - Answer accepted → "end"
    # - Max iterations → "force_submit"
    # - Need submission → "continue_without_answer"
```

#### 4. Research Workflow Phases

**Phase 1: Planning** (1 think_tool call)
- Create research plan with 5-7 specific tasks
- Identify key aspects to investigate

**Phase 2: Research** (15-20 searches minimum)
- Overview and recent news (searches 1-3)
- Expert analysis and opinions (searches 4-6)
- Historical context and trends (searches 7-9)
- Data sources and statistics (searches 10-12)
- Contrarian views and alternatives (searches 13-15)
- Deep dives into discovered aspects (searches 16-20)

**Phase 3: Reflection** (think_tool after every 2-3 searches)
- Track search count explicitly
- Assess information gaps
- Plan remaining searches

**Phase 4: Final Answer** (submit_final_answer tool)
- Comprehensive analysis (1000+ words)
- Multiple source citations (5+ URLs)
- Well-structured sections

### Memory Management

The system includes sophisticated memory management for local GPU usage:

```python
def clear_cuda_memory(verbose=True):
    """Clear CUDA memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
```

Memory is cleared:
- Before each LLM invocation
- After each tool execution
- After search operations
- During reflection phases

## 🎯 Usage Examples

### Basic Research Query
```
"What are the latest developments in quantum computing in 2024?"
```

### Complex Analysis Request
```
"Analyze the impact of AI regulation on startup innovation, including recent policy changes, industry responses, and expert predictions for 2025."
```

### Technical Deep Dive
```
"Explain the technical architecture and performance implications of retrieval-augmented generation (RAG) systems, including recent improvements and benchmarks."
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen3:14b` | Ollama model to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_TIMEOUT_S` | `300` | Model invocation timeout |
| `TAVILY_API_KEY` | Required | Tavily search API key |
| `TAVILY_SEARCH_TIMEOUT_S` | `30` | Search timeout |
| `MAX_CONCURRENT_RESEARCH_UNITS` | `3` | Max parallel operations |
| `MAX_RESEARCHER_ITERATIONS` | `3` | Max research rounds |
| `RECURSION_LIMIT` | `100` | Max agent iterations |

### Supported Models

- `qwen3:8b` - Fast, lightweight model
- `qwen3:14b` (default) - Balanced performance and speed
- `qwen3:30b` - High-quality model with enhanced reasoning capabilities
- `qwen3-coder:30b` - High-quality model with coding expertise
- `llama3:8b` - Faster, lower resource usage
- `llama3:70b` - Highest quality, requires significant resources
- `mistral:7b` - Alternative option
- `mixtral:8x7b` - Mixture of experts model

## 📁 Project Structure

```
deepagents_ollama/
├── README.md                    # This file
├── SPEC.md                      # Technical specifications
├── showroom/                    # Main application
│   ├── app.py                   # Streamlit GUI application
│   ├── agent.py                 # LangGraph agent implementation
│   ├── agent_deepagents.py      # Alternative DeepAgents implementation
│   ├── ollama_adapter.py        # Ollama integration utilities
│   ├── memory_utils.py          # CUDA memory management
│   ├── research_agent/          # Research tools and prompts
│   │   ├── __init__.py
│   │   ├── tools.py             # Tavily search, think_tool, submit_answer
│   │   └── prompts.py           # System prompts and instructions
│   ├── docs/                    # Documentation
│   │   ├── walkthrough.md       # Implementation walkthrough
│   │   └── hints.md             # Usage tips and troubleshooting
│   ├── pyproject.toml           # Dependencies and configuration
│   ├── .env                     # Environment configuration
│   └── README.md                # Showroom-specific documentation
```

## 🔍 How It Works

### 1. User Input Processing
- User enters research question via Streamlit interface
- System initializes AgentState with user message
- LangGraph begins workflow execution

### 2. Agent Decision Making
- LLM processes current state and system instructions
- Decides whether to search, reflect, or submit answer
- Makes tool calls based on research phase

### 3. Tool Execution
- **tavily_search**: Fetches web content and converts to markdown
- **think_tool**: Records strategic reflections for decision-making
- **submit_final_answer**: Validates and submits comprehensive answer

### 4. State Management
- Messages accumulated in conversation history
- Iteration count tracked for termination conditions
- Research findings stored for final synthesis

### 5. Quality Control
- Answer validation (word count, citation requirements)
- Automatic retry on rejection
- Force submission after sufficient research

## 🚨 Troubleshooting

### Common Issues

**Model Not Responding**
- Check Ollama server is running: `ollama serve`
- Verify model is installed: `ollama list`
- Check model name in `.env` matches installed model

**Search Failures**
- Verify Tavily API key is valid
- Check internet connection
- Review search timeout settings

**Memory Issues**
- Reduce model size (use `qwen3:7b` instead of `qwen3:14b`)
- Increase system RAM/VRAM
- Check CUDA memory clearing is working

**Infinite Loops**
- Agent has recursion limits and force submission
- Check system prompts are being followed
- Review iteration count in logs

## 📄 License

Apache 2.0 License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama Models](https://ollama.ai/library)
- [Tavily API Documentation](https://docs.tavily.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
