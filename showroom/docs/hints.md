This is a known incompatibility in how deepagents’ middleware expects “overwrite” config vs how the Ollama chat models from LangChain expose it; Ollama works bare, but deepagents’ middleware wraps the model and iterates over an overwrite object that is not a list/tuple, causing `TypeError: 'Overwrite' object is not iterable` (or similar).[1][2]

## Root cause

- deepagents 0.3.0 uses `create_deep_agent`, which internally calls `langchain.agents.create_agent` plus a stack of `AgentMiddleware` subclasses (planning, filesystem, subagents).[1]
- These middlewares extend the state schema and tool set and then pass an `overwrite` mapping into the underlying agent/model call, assuming `overwrite` is iterable (often a sequence of overwrites or a mapping that gets looped).[3][4]
- The LangChain Ollama chat model (and LangGraph wrappers) expose an `overwrite` or `configurable` object that is a single config object, not something you can iterate over, which is why plain tool calling works but fails when wrapped by deepagents middleware.[5][1]

## Practical workarounds

Given you already have direct tool-calling working with Ollama, the goal is to keep that stack and minimize deepagents’ interference.

Try these, roughly in order of least invasive:

1. **Use a minimal agent instead of full deepagents**

   - Build a “shallow” agent using `create_agent` directly with your Ollama model and tools, skipping `create_deep_agent` entirely.[1]
   - This avoids all deepagents middleware and therefore any iteration over the overwrite object.  
   - Drawback: you lose built‑in planning/filesystem/subagent layers and need to re‑create only what you actually want.

2. **Disable extra middleware when using Ollama**

   - If you still want some deepagents behavior, call `create_deep_agent` but pass a minimal middleware list or a custom agent created with `create_agent` plus only the middlewares you know are safe with Ollama.[4][1]
   - Example idea (pseudo‑code, not exact from docs):  
     ```python
     from deepagents import create_deep_agent
     from langchain_community.chat_models import ChatOllama

     model = ChatOllama(model="your-ollama-model")

     agent = create_deep_agent(
         model=model,
         tools=[...],
         # Avoid adding additional custom middleware that touches overwrite
         middleware=[],
     )
     ```
   - The key is to keep any custom middleware from iterating `config["overwrite"]` (or similar). If you wrote your own middleware, make sure you treat overwrite as a single mapping, not a list.

3. **Wrap Ollama in a thin LangChain model adapter**

   - Implement a small wrapper around the Ollama chat model whose `invoke` (and optional async variants) ignore or sanitize `config["overwrite"]` and never try to iterate it.[6][7]
   - In other words, intercept the config before it reaches your model and strip or normalize the overwrite field to something trivial (e.g. `None` or `{}`) so deepagents’ middleware does not get a non‑iterable custom object.

4. **Pin versions to a known‑good combo**

   - Check the `deepagents` GitHub issues for an Ollama‑specific bug or suggested version matrix; if a patch landed, pin `deepagents` and `langchain` to the versions where they fixed middleware/overwrite handling for non‑OpenAI models.[2][8]
   - Also ensure Ollama Python client and LangChain integration are on their latest compatible versions, since recent releases added first‑class tool support and may have adjusted config semantics.[9][5]

## What is likely happening in your code

- Your direct Ollama tool‑calling path probably calls `model.invoke()` or `agent.invoke()` without deepagents, so no middleware tries to inspect or iterate overwrite.[10][1]
- When you switch to `create_deep_agent(model=ollama_model, ...)`, the stack of middlewares is added by default; one of them, or the base agent harness, loops over overwrite as if it were a list/tuple, triggering the “object is not iterable” error as soon as a tool call is orchestrated through the deep agent.[4][1]

## How to debug concretely

- Enable logging around the deepagents call, inspect the `config` or `state` object passed into middleware to find where overwrite is treated as iterable.[1]
- As a quick test, temporarily remove all custom middleware from your deep agent; if the error disappears, re‑introduce them one by one and adjust the offending one to treat overwrite as a single mapping (or to ignore it for Ollama).[4]

If you paste your current `create_deep_agent(...)` snippet and model initialization, a more pinpointed patch (or a small adapter class) can be sketched so you can keep most of deepagents while making Ollama behave nicely.

[1](https://github.com/langchain-ai/deepagents)
[2](https://pypi.org/project/deepagents/)
[3](https://towardsdatascience.com/lessons-learnt-from-upgrading-to-langchain-1-0-in-production/)
[4](https://www.flowhunt.io/blog/building-extensible-ai-agents-with-langchain-1-0/)
[5](https://github.com/ollama/ollama/releases)
[6](https://reference.langchain.com/python/langsmith/observability/sdk/evaluation/)
[7](https://github.com/langchain-ai/langchain/issues/33474)
[8](https://github.com/langchain-ai/deepagents/issues)
[9](https://www.reddit.com/r/ollama/comments/1ecanzn/ollama_030_tool_support/)
[10](https://www.cohorte.co/blog/how-to-build-a-local-ai-agent-using-deepseek-and-ollama-a-step-by-step-guide)
[11](https://github.com/deepset-ai/haystack/issues/6285)
[12](https://stackoverflow.com/questions/29886552/why-are-objects-not-iterable-in-javascript)
[13](https://langfuse.com/docs/sdk/python/decorators)
[14](https://reference.langchain.com/python/langchain_core/document_loaders/)
[15](https://github.com/huggingface/smolagents/issues/1742)
[16](https://www.youtube.com/watch?v=AZ6257Ya_70)
[17](https://www.reddit.com/r/StableDiffusion/comments/1bs7ly6/getting_this_error_constantly_in_sd_forge/)
[18](https://www.npmjs.com/package/type-fest?activeTab=dependents)
[19](https://ollama.com/library)
[20](https://stackoverflow.com/questions/11893532/object-not-iterable-with-inputfile)