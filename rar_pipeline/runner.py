"""
rar_pipeline/runner.py
Created on October 13, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

from typing import Any

from langchain_core.runnables import RunnableConfig

from agentic_workflow import graph
from configuration import Configuration


conf = Configuration()

LAB_MAX_CONCURRENCY = 2

config = RunnableConfig(
    recursion_limit=conf.recursion_limit,
    max_concurrency=LAB_MAX_CONCURRENCY,
    configurable={
        "search_api": conf.search_api,
        "number_of_queries": conf.number_of_queries,
        "max_search_depth": conf.max_search_depth,
    },
)


async def run_agenticRAG_detailed(
    question: str,
    options: str,
    config: RunnableConfig,
) -> dict:
    user_content = (
        f"Question:\n{question}\n\n"
        f"Options:\n{options}"
    )

    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": user_content,
            }
        ],
        "sections": [],
        "completed_sections": [],
        "report_model_calls": [],
    }

    result: Any = await graph.ainvoke(
        initial_state,
        config=config,
    )

    if isinstance(result, dict):
        report = (
            result.get("final_report")
            or result.get("content")
        )
        report_model_calls = result.get(
            "report_model_calls",
            [],
        )
    else:
        report = result
        report_model_calls = []

    return {
        "report": report or "[ERROR] No report found",
        "report_model_calls": report_model_calls,
    }
