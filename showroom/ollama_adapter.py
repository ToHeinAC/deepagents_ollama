"""Ollama Adapter for DeepAgents Middleware Compatibility.

This module provides a wrapper around ChatOllama that sanitizes config objects
to be compatible with deepagents middleware, which expects 'overwrite' to be
iterable (list/tuple) rather than a single config object.

The adapter intercepts invoke/ainvoke calls and normalizes the config before
passing it to the underlying Ollama model.
"""

from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.runnables import RunnableConfig


class DeepAgentsOllamaAdapter(BaseChatModel):
    """Wrapper around ChatOllama that sanitizes config for deepagents middleware.
    
    DeepAgents middleware iterates over config["overwrite"] assuming it's a 
    list/tuple. Ollama's integration provides a single object, causing TypeError.
    This adapter intercepts calls and normalizes the config.
    
    Usage:
        from ollama_adapter import DeepAgentsOllamaAdapter
        from deepagents import create_deep_agent
        
        model = DeepAgentsOllamaAdapter(
            model="gpt-oss:20b",
            base_url="http://localhost:11434",
            temperature=0.0
        )
        
        agent = create_deep_agent(
            model=model,
            tools=[...],
            system_prompt="..."
        )
    """
    
    # The underlying ChatOllama instance
    ollama_model: ChatOllama
    
    # Model name for identification
    model_name: str = "ollama"
    
    def __init__(
        self,
        model: str = "gpt-oss:20b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        **kwargs
    ):
        """Initialize the adapter with ChatOllama configuration.
        
        Args:
            model: Ollama model name (e.g., "gpt-oss:20b", "llama3:8b")
            base_url: Ollama server URL
            temperature: Sampling temperature
            **kwargs: Additional ChatOllama parameters
        """
        # Initialize the underlying Ollama model
        ollama_model = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )
        
        # Call parent init with our model
        super().__init__(ollama_model=ollama_model, model_name=model)
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "deepagents-ollama-adapter"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model_name": self.model_name,
            "base_url": self.ollama_model.base_url,
        }
    
    def _sanitize_config(self, config: Optional[RunnableConfig] = None) -> Optional[RunnableConfig]:
        """Sanitize config to prevent middleware iteration errors.
        
        Deepagents middleware may try to iterate over config["overwrite"].
        This method ensures overwrite is either None, empty, or a proper list.
        
        Args:
            config: The runnable config that may contain problematic overwrite
            
        Returns:
            Sanitized config safe for deepagents middleware
        """
        if config is None:
            return None
            
        # Create a copy to avoid mutating the original
        sanitized = dict(config)
        
        # Handle the overwrite field that causes issues
        if "overwrite" in sanitized:
            overwrite = sanitized["overwrite"]
            
            # If it's not iterable (single object), wrap in list or remove
            if overwrite is not None and not isinstance(overwrite, (list, tuple)):
                # Option 1: Wrap in list (preserves the config)
                # sanitized["overwrite"] = [overwrite]
                
                # Option 2: Remove it (simpler, avoids middleware processing)
                sanitized["overwrite"] = None
        
        # Also handle configurable field if present
        if "configurable" in sanitized:
            configurable = sanitized["configurable"]
            if configurable is not None and not isinstance(configurable, dict):
                sanitized["configurable"] = {}
        
        return sanitized
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using the underlying Ollama model.
        
        Args:
            messages: List of messages to process
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional arguments
            
        Returns:
            ChatResult from the Ollama model
        """
        print(f"[ADAPTER] _generate called with {len(messages)} messages")
        result = self.ollama_model._generate(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs
        )
        print(f"[ADAPTER] _generate returned: {type(result)}")
        return result
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream responses from the underlying Ollama model."""
        print(f"[ADAPTER] _stream called with {len(messages)} messages")
        yield from self.ollama_model._stream(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs
        )
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate a response using the underlying Ollama model.
        
        Args:
            messages: List of messages to process
            stop: Stop sequences
            run_manager: Async callback manager
            **kwargs: Additional arguments
            
        Returns:
            ChatResult from the Ollama model
        """
        return await self.ollama_model._agenerate(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs
        )
    
    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the model with sanitized config.
        
        This is the main entry point that deepagents middleware calls.
        We sanitize the config before passing to the underlying model.
        
        Args:
            input: The input to process (messages)
            config: Runnable config that may need sanitization
            **kwargs: Additional arguments
            
        Returns:
            Model response
        """
        print(f"[ADAPTER] invoke called with input type: {type(input)}")
        sanitized_config = self._sanitize_config(config)
        result = self.ollama_model.invoke(input, config=sanitized_config, **kwargs)
        print(f"[ADAPTER] invoke returned: {type(result)}")
        return result
    
    def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        """Stream the model with sanitized config."""
        print(f"[ADAPTER] stream called with input type: {type(input)}")
        sanitized_config = self._sanitize_config(config)
        yield from self.ollama_model.stream(input, config=sanitized_config, **kwargs)
    
    async def ainvoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """Async invoke the model with sanitized config.
        
        Args:
            input: The input to process (messages)
            config: Runnable config that may need sanitization
            **kwargs: Additional arguments
            
        Returns:
            Model response
        """
        sanitized_config = self._sanitize_config(config)
        return await self.ollama_model.ainvoke(input, config=sanitized_config, **kwargs)
    
    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> Any:
        """Bind tools to the model.
        
        Returns the bound model directly since RunnableBinding handles
        invoke/stream properly.
        
        Args:
            tools: Tools to bind
            **kwargs: Additional arguments for bind_tools
            
        Returns:
            Bound model (RunnableBinding) with tools attached
        """
        print(f"[ADAPTER] bind_tools called with {len(tools)} tools")
        # Return the bound model directly - it's a RunnableBinding that works
        bound_model = self.ollama_model.bind_tools(tools, **kwargs)
        print(f"[ADAPTER] bind_tools returning: {type(bound_model)}")
        return bound_model
    
    def bind(self, **kwargs: Any) -> Any:
        """Bind additional arguments to the model."""
        print(f"[ADAPTER] bind called with kwargs: {list(kwargs.keys())}")
        bound_model = self.ollama_model.bind(**kwargs)
        print(f"[ADAPTER] bind returning: {type(bound_model)}")
        return bound_model
    
    def with_config(
        self,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> "DeepAgentsOllamaAdapter":
        """Return a new adapter with the given config.
        
        Args:
            config: Config to apply
            **kwargs: Additional config options
            
        Returns:
            New adapter with config applied
        """
        sanitized_config = self._sanitize_config(config)
        new_adapter = DeepAgentsOllamaAdapter.__new__(DeepAgentsOllamaAdapter)
        new_adapter.ollama_model = self.ollama_model.with_config(sanitized_config, **kwargs)
        new_adapter.model_name = self.model_name
        return new_adapter


def create_ollama_for_deepagents(
    model: str = "gpt-oss:20b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    **kwargs
) -> DeepAgentsOllamaAdapter:
    """Factory function to create an Ollama model compatible with deepagents.
    
    This is the recommended way to create an Ollama model for use with
    deepagents middleware.
    
    Args:
        model: Ollama model name
        base_url: Ollama server URL
        temperature: Sampling temperature
        **kwargs: Additional ChatOllama parameters
        
    Returns:
        DeepAgentsOllamaAdapter ready for use with create_deep_agent
        
    Example:
        from ollama_adapter import create_ollama_for_deepagents
        from deepagents import create_deep_agent
        
        model = create_ollama_for_deepagents(model="gpt-oss:20b")
        
        agent = create_deep_agent(
            model=model,
            tools=[internet_search],
            system_prompt="You are a research assistant.",
            subagents=[research_subagent],
        )
    """
    return DeepAgentsOllamaAdapter(
        model=model,
        base_url=base_url,
        temperature=temperature,
        **kwargs
    )
