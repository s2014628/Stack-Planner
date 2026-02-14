import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.config.agents import AGENT_LLM_MAP
from src.graph.types import State
from src.llms.llm import get_llm_by_type
from src.memory import MemoryStack, MemoryStackEntry
from src.prompts.template import apply_prompt_template
from src.utils.logger import logger


LOCOMO_SYSTEM_PROMPT = """You are an intelligent assistant that answers questions based on conversation history.
You will be given a multi-session conversation between two people and a question about the conversation.
Your task is to analyze the conversation carefully and provide a short, accurate answer.

Key guidelines:
- Answer with exact words from the conversation whenever possible
- Keep your answer concise (a few words or a short phrase)
- If the question asks about time/date, provide the specific date mentioned in the conversation
- If the question asks about an event, describe it briefly using conversation details
- For adversarial questions (things not mentioned), say "Not mentioned in the conversation"
"""


class LoComoCentralAgent:

    def __init__(self):
        self.memory_stack = MemoryStack()

    def run_qa(self, conversation_history: str, question: str) -> Dict[str, Any]:
        llm = get_llm_by_type(AGENT_LLM_MAP.get("central_agent", "basic"))

        think_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="think",
            content=f"Analyzing conversation to answer: {question}",
        )
        self.memory_stack.push(think_entry)

        messages = [
            {"role": "system", "content": LOCOMO_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"--- Conversation History ---\n{conversation_history}\n"
                    f"--- End of Conversation ---\n\n"
                    f"Question: {question}\n\n"
                    f"Please provide a short, concise answer based on the conversation above. "
                    f"Answer with exact words from the conversation whenever possible."
                ),
            },
        ]

        try:
            response = llm.invoke(messages)
            answer = response.content.strip()
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            answer = ""

        answer_entry = MemoryStackEntry(
            timestamp=datetime.now().isoformat(),
            action="finish",
            agent_type="central_agent",
            content=f"Generated answer: {answer}",
            result={"answer": answer, "question": question},
        )
        self.memory_stack.push(answer_entry)

        return {
            "prediction": answer,
            "memory_stack_log": self.memory_stack.to_dict(),
        }


async def locomo_central_agent_node(state: State, config: RunnableConfig) -> Command:
    logger.info("LoCoMo Central Agent node activated")

    user_query = state.get("user_query", "")
    conversation_context = state.get("observations", [])

    context_str = "\n".join(conversation_context) if conversation_context else ""

    llm = get_llm_by_type(AGENT_LLM_MAP.get("central_agent", "basic"))

    messages = [
        {"role": "system", "content": LOCOMO_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    try:
        response = llm.invoke(messages)
        answer = response.content.strip()
    except Exception as e:
        logger.error(f"LoCoMo Central Agent failed: {e}")
        answer = f"Error: {e}"

    return Command(
        update={
            "messages": [AIMessage(content=answer, name="locomo_central_agent")],
            "final_report": answer,
            "current_node": "locomo_central_agent",
        },
        goto=END,
    )


# NOTE: Researcher node for LoCoMo - designed but commented out.
# Uncomment and adapt for benchmarks that need web search / retrieval.
#
# async def locomo_researcher_node(state: State, config: RunnableConfig) -> Command:
#     """Researcher node for LoCoMo - performs information retrieval if needed.
#
#     This node can be used for benchmarks where the agent needs to:
#     - Search external knowledge bases
#     - Retrieve relevant documents
#     - Crawl web pages for additional context
#
#     For LoCoMo benchmark, the conversation context is already provided,
#     so this node is not needed. Uncomment for other benchmarks.
#     """
#     from src.tools import get_web_search_tool, crawl_tool, search_docs_tool
#     from src.agents.ResearcherAgent_SP import ResearcherAgentSP
#
#     logger.info("LoCoMo Researcher Agent activated")
#     delegation_context = state.get("delegation_context", {})
#     task_description = delegation_context.get("task_description", "")
#
#     tools = [get_web_search_tool(10), crawl_tool, search_docs_tool]
#
#     research_agent = ResearcherAgentSP(
#         config=config, agent_type="researcher", default_tools=tools
#     )
#
#     try:
#         result_command = await research_agent.execute_agent_step(state)
#         result_observations = []
#         if result_command and result_command.update:
#             result_observations = result_command.update.get("observations", [])
#     except Exception as e:
#         logger.error(f"Researcher Agent failed: {e}")
#         result_observations = []
#
#     return Command(
#         update={
#             "messages": [HumanMessage(content="Research complete", name="researcher")],
#             "observations": result_observations,
#             "current_node": "locomo_central_agent",
#         },
#         goto="locomo_central_agent",
#     )


def build_locomo_graph():
    builder = StateGraph(State)

    builder.add_node("locomo_central_agent", locomo_central_agent_node)

    # NOTE: Researcher node designed but commented out for LoCoMo.
    # Uncomment for benchmarks that need retrieval/search.
    # builder.add_node("researcher", locomo_researcher_node)

    builder.add_edge(START, "locomo_central_agent")
    builder.add_edge("locomo_central_agent", END)

    return builder.compile()


async def run_locomo_single(
    user_query: str,
    conversation_history: str = "",
    question: str = "",
) -> Dict[str, Any]:
    agent = LoComoCentralAgent()
    result = agent.run_qa(conversation_history, question)
    return result


async def run_locomo_workflow(
    user_query: str,
    graph=None,
) -> Dict[str, Any]:
    if graph is None:
        graph = build_locomo_graph()

    initial_state = {
        "messages": [{"role": "user", "content": user_query}],
        "auto_accepted_plan": True,
        "enable_background_investigation": False,
        "user_query": user_query,
    }

    config = {
        "configurable": {
            "thread_id": "locomo_run",
            "max_plan_iterations": 1,
            "max_step_num": 1,
            "mcp_settings": {},
        },
        "recursion_limit": 10,
    }

    final_state = None
    async for s in graph.astream(input=initial_state, config=config, stream_mode="values"):
        if isinstance(s, dict):
            final_state = s

    final_report = ""
    memory_stack_log = []
    if final_state:
        final_report = final_state.get("final_report", "")
        memory_stack_raw = final_state.get("memory_stack", None)
        if memory_stack_raw and isinstance(memory_stack_raw, str):
            try:
                memory_stack_log = json.loads(memory_stack_raw)
            except json.JSONDecodeError:
                memory_stack_log = []

    return {
        "prediction": final_report,
        "memory_stack_log": memory_stack_log,
    }
