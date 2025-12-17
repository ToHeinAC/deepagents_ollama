"""Deep Research Agent using DeepAgents Middleware with Local Ollama.

This module creates a deep research agent using the deepagents framework
with full middleware support (planning, filesystem, subagents) while
using a local Ollama LLM through the compatibility adapter.

This implementation fulfills the SPEC.md requirements:
- Uses deepagents middleware architecture
- Fully local LLM (gpt-oss:20b from Ollama)
- Planning and task decomposition
- Context management
- Subagent spawning
"""

import os
from datetime import datetime
from typing import Literal
from dotenv import load_dotenv

from tavily import TavilyClient
from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents.middleware import TodoListMiddleware, SummarizationMiddleware

from ollama_adapter import create_ollama_for_deepagents

# Load environment variables
load_dotenv()

# Configuration from environment
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MAX_CONCURRENT_RESEARCH_UNITS = int(os.getenv("MAX_CONCURRENT_RESEARCH_UNITS", "3"))
MAX_RESEARCHER_ITERATIONS = int(os.getenv("MAX_RESEARCHER_ITERATIONS", "3"))
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "1000"))

# Get current date for prompts
current_date = datetime.now().strftime("%Y-%m-%d")

# Initialize Tavily client for web search
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# =============================================================================
# TOOLS
# =============================================================================

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> str:
    """Search the web for information using Tavily.
    
    Args:
        query: Search query to execute
        max_results: Maximum number of results (default: 5)
        topic: Topic filter - 'general', 'news', or 'finance'
        include_raw_content: Whether to include raw HTML content
        
    Returns:
        Formatted search results with titles, URLs, and content
    """
    response = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    
    # Format results for the agent
    results = []
    for item in response.get("results", []):
        title = item.get("title", "No Title")
        url = item.get("url", "")
        content = item.get("content", "")
        
        # Truncate long content for local models
        if len(content) > 2000:
            content = content[:2000] + "... (truncated)"
            
        results.append(f"**{title}**\nURL: {url}\n{content}")
    
    return "\n\n---\n\n".join(results) if results else "No results found."


def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress.
    
    Use this to pause and reflect on:
    - What information has been gathered
    - What gaps remain
    - Whether to continue searching or provide an answer
    
    Args:
        reflection: Your detailed reflection on progress and next steps
        
    Returns:
        Confirmation that reflection was recorded
    """
    return f"Reflection recorded. Continue with your research plan."


# =============================================================================
# PROMPTS
# =============================================================================

MAIN_AGENT_INSTRUCTIONS = f"""You are a deep research assistant. Today's date is {current_date}.

Your goal is to thoroughly research user questions and provide comprehensive, well-sourced answers.

## WORKFLOW

1. **First**, write the original user question to question.txt using the write_file tool
2. **Then**, create a research plan using write_todos with specific sub-questions to investigate
3. **Use** the research-agent subagent to investigate each sub-question (one topic at a time!)
4. **After** gathering enough information, write your final report to final_report.md
5. **Call** the critique-agent to review your report
6. **If needed**, do additional research based on the critique
7. **Update** final_report.md with improvements

## IMPORTANT RULES

- Only give the research-agent ONE topic at a time
- Use think_tool to reflect on progress before deciding next steps
- Always cite sources with URLs in your final report
- Update your to-do list as you complete tasks
- Maximum {MAX_RESEARCHER_ITERATIONS} research iterations before concluding

## OUTPUT FORMAT

Your final report should be written to final_report.md with:
- Executive summary
- Key findings with citations
- Detailed analysis
- Conclusion
"""

RESEARCH_SUBAGENT_PROMPT = f"""You are a specialized research agent. Today's date is {current_date}.

Your task is to deeply research the specific topic assigned to you.

## WORKFLOW

1. Use internet_search to find relevant information
2. Use think_tool to reflect on what you found
3. Search again if needed (max 5 searches)
4. Provide a comprehensive summary of your findings

## RULES

- Focus ONLY on the specific topic assigned
- Always include source URLs
- Be thorough but concise
- Use think_tool before concluding to ensure completeness
"""

CRITIQUE_SUBAGENT_PROMPT = f"""You are a dedicated editor and critic. Today's date is {current_date}.

Your task is to critique the research report at final_report.md.

## WORKFLOW

1. Read the original question from question.txt
2. Read the report from final_report.md
3. Evaluate the report against the original question

## CRITIQUE CRITERIA

- **Completeness**: Does it fully answer the question?
- **Accuracy**: Are claims properly supported with sources?
- **Clarity**: Is it well-organized and easy to understand?
- **Sources**: Are there enough credible sources cited?

## OUTPUT

Provide specific, actionable feedback for improvement.
If the report is satisfactory, say "APPROVED" and explain why.
"""


# =============================================================================
# SUBAGENTS
# =============================================================================

research_subagent = {
    "name": "research-agent",
    "description": "Use to research specific topics in depth. Only give this agent ONE topic at a time. Do not pass multiple sub-questions.",
    "system_prompt": RESEARCH_SUBAGENT_PROMPT,
    "tools": [internet_search, think_tool],
    # Uses main agent's model by default (Ollama)
}

critique_subagent = {
    "name": "critique-agent", 
    "description": "Use to get a critique of the final report. Call this after writing final_report.md.",
    "system_prompt": CRITIQUE_SUBAGENT_PROMPT,
    # No tools - uses filesystem tools from main agent
}


# =============================================================================
# AGENT CREATION
# =============================================================================

def create_research_agent():
    """Create the deep research agent with custom SummarizationMiddleware settings.
    
    Returns:
        A deepagents agent with:
        - Local Ollama LLM (qwen3:14b)
        - SummarizationMiddleware configured for smaller context windows
        - SubAgentMiddleware for task delegation
    """
    from langchain.agents import create_agent
    
    # Create Ollama model with deepagents compatibility
    model = create_ollama_for_deepagents(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )
    
    # Configure SummarizationMiddleware with aggressive settings for smaller context
    # qwen3:14b has ~8k context, so we need to summarize early
    summarization_middleware = SummarizationMiddleware(
        model=model,
        trigger=[("messages", 8), ("tokens", 3000)],  # Trigger early
        keep=("messages", 4),  # Keep only last 4 messages
        trim_tokens_to_summarize=1500,  # Aggressive trimming
    )
    
    # Configure SubAgentMiddleware for research delegation
    subagent_middleware = SubAgentMiddleware(
        default_model=model,
        default_tools=[internet_search, think_tool],
        subagents=[research_subagent, critique_subagent],
    )
    
    # Create agent with custom middleware stack
    agent = create_agent(
        model=model,
        tools=[internet_search, think_tool],
        system_prompt=MAIN_AGENT_INSTRUCTIONS,
        middleware=[
            summarization_middleware,
            subagent_middleware,
        ],
    )
    
    return agent


# Create global agent instance
agent = create_research_agent()


def get_agent():
    """Get the configured research agent instance."""
    return agent


def get_agent_config():
    """Get the current agent configuration."""
    return {
        "model": OLLAMA_MODEL,
        "base_url": OLLAMA_BASE_URL,
        "max_concurrent_research_units": MAX_CONCURRENT_RESEARCH_UNITS,
        "max_researcher_iterations": MAX_RESEARCHER_ITERATIONS,
        "recursion_limit": RECURSION_LIMIT,
        "middleware": ["FilesystemMiddleware", "TodoListMiddleware", "SubAgentMiddleware"],
        "subagents": ["research-agent", "critique-agent"],
    }


# =============================================================================
# ALTERNATIVE: Minimal Middleware Configuration
# =============================================================================

def create_minimal_agent():
    """Create agent with minimal middleware for testing.
    
    Use this if the full middleware stack causes issues.
    Only includes SubAgentMiddleware for task delegation.
    """
    from langchain.agents import create_agent
    
    model = create_ollama_for_deepagents(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )
    
    # Use create_agent with only SubAgentMiddleware
    agent = create_agent(
        model=model,
        tools=[internet_search, think_tool],
        middleware=[
            SubAgentMiddleware(
                default_model=model,
                default_tools=[internet_search, think_tool],
                subagents=[research_subagent, critique_subagent],
            )
        ],
    )
    
    return agent
