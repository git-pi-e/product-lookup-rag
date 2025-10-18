import json
from typing import Any, Dict, List

from langchain.agents import (
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.tools import Tool
from langchain import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish

from .llm import LLMService
from .retrieval import RetrievalService


class AgentServices:
    def __init__(self):
        self.llm = LLMService()
        self.retrieval = RetrievalService(embedder=self.llm.embed_text)

    def tool_query(self, user_prompt: str) -> List[Dict[str, Any]]:
        extracted = self.llm.extract_entities(user_prompt)
        res = self.retrieval.query_by_entities(extracted)
        matches: List[Dict[str, Any]] = []
        for r in res:
            matches.append({"id": r["p"]["id"], "name": r["p"]["name"]})
        return matches

    def tool_similarity_search(self, user_prompt: str) -> List[Dict[str, Any]]:
        res = self.retrieval.query_by_prompt_similarity(user_prompt)
        matches: List[Dict[str, Any]] = []
        for r in res:
            matches.append({"id": r["p"]["id"], "name": r["p"]["name"]})
        return matches

    def tool_similar_items(self, product_ids_json: str) -> List[Dict[str, Any]]:
        try:
            ids = json.loads(product_ids_json)
            if not isinstance(ids, list):
                return []
        except Exception:
            return []
        acc: List[Dict[str, Any]] = []
        for pid in ids:
            acc.extend(self.retrieval.similar_items(int(pid)))
        # Deduplicate, keep order
        seen = set()
        unique: List[Dict[str, Any]] = []
        for x in acc:
            if x["id"] not in seen:
                seen.add(x["id"])
                unique.append(x)
        return unique[:5]


prompt_template = """Your goal is to find products in the database that best match the user prompt.
You have access to these tools:

{tools}

Use the following format:

Question: the input prompt from the user
Thought: you should always think about what to do
Action: the action to take (one of: {tool_names})
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Rules to follow:
1. Start with Query tool using the full user prompt. If results are empty, try Similarity Search with the full prompt.
2. Once you have matches, call Similar Items with the list of product ids (as a JSON array) to fetch up to 5 similar items.
3. If still no results, ask for more context (e.g., category, color, brand) or say you couldn't find suitable products.

Return the final answer as:
- Number of matches
- One line per match: name (id)
- Similar items (up to 5): one line per item: name (id)

User prompt:
{input}

{agent_scratchpad}
"""


class CustomPromptTemplate(StringPromptTemplate):
    template: str

    def __init__(self, template: str, tools: List[Tool]):
        super().__init__(input_variables=["input", "intermediate_steps"])
        self.template = template
        self._tools = tools

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{t.name}: {t.description}" for t in self._tools])
        kwargs["tool_names"] = ", ".join([t.name for t in self._tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str):
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        import re

        match = re.search(
            r"Action: (.*?)[\n]*Action Input:[\s]*(.*)", llm_output, re.DOTALL
        )
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2).strip().strip('"')
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)


def build_agent() -> AgentExecutor:
    services = AgentServices()
    tools = [
        Tool(
            name="Query",
            func=services.tool_query,
            description="Find products by entities extracted from the user prompt",
        ),
        Tool(
            name="Similarity Search",
            func=services.tool_similarity_search,
            description="Find products by similarity to the full prompt",
        ),
        Tool(
            name="Similar Items",
            func=services.tool_similar_items,
            description="Given a JSON array of product ids, return up to 5 similar items",
        ),
    ]

    prompt = CustomPromptTemplate(template=prompt_template, tools=tools)
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    output_parser = CustomOutputParser()
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=[t.name for t in tools],
    )
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)


def agent_run(prompt: str) -> str:
    executor = build_agent()
    return executor.run(prompt)
