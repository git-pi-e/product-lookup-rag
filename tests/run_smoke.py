import sys
import os
import types
from importlib import import_module

# Ensure repo root is on PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# Provide lightweight mocks for langchain imports so tests can run without packages
def _inject_langchain_mocks():
    # langchain.agents
    agents_mod = types.ModuleType("langchain.agents")

    class AgentExecutor:
        def __init__(self, *a, **k):
            pass

        def run(self, prompt):
            return ""

    def create_tool_calling_agent(*a, **k):
        return object()

    agents_mod.AgentExecutor = AgentExecutor
    agents_mod.create_tool_calling_agent = create_tool_calling_agent

    # langchain.tools / langchain_core.tools
    tools_mod = types.ModuleType("langchain.tools")

    class Tool:
        def __init__(self, name=None, func=None, description=None):
            self.name = name
            self.func = func
            self.description = description

    tools_mod.Tool = Tool

    # langchain_core.prompts
    prompts_mod = types.ModuleType("langchain_core.prompts")

    class StringPromptTemplate:
        def __init__(self, input_variables=None):
            self.input_variables = input_variables or []

        def format(self, **kwargs):
            return ""

    prompts_mod.StringPromptTemplate = StringPromptTemplate

    # langchain_openai.ChatOpenAI stub
    lc_openai = types.ModuleType("langchain_openai")

    class ChatOpenAI:
        def __init__(self, *a, **k):
            pass

    lc_openai.ChatOpenAI = ChatOpenAI

    # langchain.schema
    schema_mod = types.ModuleType("langchain.schema")

    class AgentAction:
        pass

    class AgentFinish:
        pass

    schema_mod.AgentAction = AgentAction
    schema_mod.AgentFinish = AgentFinish

    sys.modules.update(
        {
            "langchain.agents": agents_mod,
            "langchain.tools": tools_mod,
            "langchain_core.tools": tools_mod,
            "langchain_core.prompts": prompts_mod,
            "langchain_openai": lc_openai,
            "langchain.schema": schema_mod,
        }
    )


_inject_langchain_mocks()

# Import agent module and monkeypatch dependencies to avoid network/DB calls
agent_mod = import_module("app.services.agent")


class DummyLLMService:
    def __init__(self, *a, **k):
        pass

    def extract_entities(self, prompt: str):
        # return JSON string of entities
        return '{"color": "red", "category": "clothes", "age_group": "adults"}'

    def embed_text(self, text: str):
        return [0.0] * 1536


class DummyRetrievalService:
    def __init__(self, embedder=None):
        pass

    def query_by_entities(self, extracted_json: str):
        return [
            {
                "p": {"id": 123, "name": "Red Shirt"},
                "_meta": {"retrieval_method": "entities_exact_match"},
            }
        ]

    def query_by_prompt_similarity(self, prompt: str):
        return []

    def similar_items(self, product_id: int):
        return [{"id": 456, "name": "Red T-Shirt"}]


def run():
    # Patch classes used by AgentServices before building the agent
    agent_mod.LLMService = DummyLLMService
    agent_mod.RetrievalService = DummyRetrievalService

    out = agent_mod.agent_run_structured("Suggest some red clothes for adults")
    print(out)


if __name__ == "__main__":
    run()
