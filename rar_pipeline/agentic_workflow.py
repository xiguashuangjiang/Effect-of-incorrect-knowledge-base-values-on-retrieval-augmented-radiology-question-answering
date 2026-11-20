"""
rar_pipeline/agentic_workflow.py
Created on October 13, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

from typing import List, Annotated, TypedDict, Literal
import operator
import re
from pydantic import BaseModel, Field

from utils import init_chat_model, get_config_value, searxng_search
from local_search import (
    local_corpus_search,
    RETRIEVAL_STAGE,
    RETRIEVAL_SECTION,
)
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.types import Command, Send

from configuration import Configuration
from prompts_radiology import RESEARCH_INSTRUCTIONS

import logging

logger = logging.getLogger(__name__)


def get_search_tool(config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)

    if search_api.lower() == "searxng":
        return searxng_search
    elif search_api.lower() == "local":
        return local_corpus_search
    else:
        raise NotImplementedError(
            f"The search API '{search_api}' is not supported. "
            "Please set search_api to 'searxng' or 'local'."
        )


@tool
class Section(BaseModel):
    name: str = Field(
        description="The assigned answer option, e.g. 'Option A: ...'.",
    )
    supporting_evidence: str = Field(
        description=(
            "Up to 3 short bullet points supporting this option in the "
            "context of the original clinical question."
        ),
    )
    contradictory_evidence: str = Field(
        description=(
            "Up to 3 short bullet points arguing against this option in the "
            "context of the original clinical question."
        ),
    )
    synthesis: str = Field(
        description=(
            "A neutral 1-3 sentence synthesis of this option only. Do not "
            "rank options and do not state which option is correct."
        ),
    )
    sources: str = Field(
        description=(
            "A compact newline-separated list of at most 5 source article "
            "identifiers actually used."
        ),
    )


def _clean_field(value: str) -> str:
    return str(value or "").strip()


def _format_completed_section(section: Section) -> str:
    name = _clean_field(section.name)
    supporting = _clean_field(section.supporting_evidence)
    contradictory = _clean_field(section.contradictory_evidence)
    synthesis = _clean_field(section.synthesis)
    sources = _clean_field(section.sources)

    return (
        f"## {name}\n\n"
        f"**Supporting evidence**\n{supporting}\n\n"
        f"**Contradictory evidence**\n{contradictory}\n\n"
        f"**Neutral synthesis**\n{synthesis}\n\n"
        f"### Sources\n{sources}"
    )


def _build_final_report(completed_sections: list[Section]) -> str:
    intro = (
        "# Evidence Report\n\n"
        "The following option-specific sections summarize retrieved evidence "
        "relevant to the multiple-choice question. The options are not ranked "
        "and no final diagnosis is provided."
    )

    body = "\n\n".join(
        _format_completed_section(section)
        for section in completed_sections
    )

    ending = (
        "## Final note\n\n"
        "The sections above summarize evidence relevant to each answer option. "
        "The final answer should be determined independently by considering "
        "the clinical, imaging, pathological, and other question-specific "
        "findings together."
    )

    return f"{intro}\n\n{body}\n\n{ending}"


class ReportStateOutput(TypedDict):
    final_report: str
    report_model_calls: list[dict]


class ReportState(MessagesState):
    sections: list[str]
    completed_sections: Annotated[list, operator.add]
    report_model_calls: Annotated[list, operator.add]
    final_report: str


class SectionState(MessagesState):
    section: str
    completed_sections: list[Section]
    search_iterations: int
    section_error_retries: int
    report_model_calls: Annotated[list, operator.add]


class SectionOutputState(TypedDict):
    completed_sections: list[Section]
    report_model_calls: list[dict]


def _message_role_and_content(message):
    if isinstance(message, dict):
        return message.get("role", ""), str(message.get("content", ""))

    msg_type = getattr(message, "type", "")
    content = str(getattr(message, "content", ""))

    role_map = {
        "human": "user",
        "user": "user",
        "ai": "assistant",
        "assistant": "assistant",
        "system": "system",
        "tool": "tool",
    }
    return role_map.get(msg_type, msg_type), content


def _get_original_user_content(messages) -> str:
    for message in messages:
        role, content = _message_role_and_content(message)
        if role == "user" and content.strip():
            return content
    return ""


def _parse_fixed_option_sections(messages) -> List[str]:
    original = _get_original_user_content(messages)

    if not original:
        raise RuntimeError("Could not find the original user question.")

    match = re.search(r"(?im)^\s*Options\s*:\s*$", original)

    if not match:
        raise RuntimeError(
            "Could not find an 'Options:' block in the original question."
        )

    options_block = original[match.end():]
    sections = []

    for line in options_block.splitlines():
        line = line.strip()
        m = re.match(r"^([A-Z])\s*[:.)-]\s*(.+?)\s*$", line)
        if not m:
            continue

        label = m.group(1)
        text = m.group(2).strip()

        if text:
            sections.append(f"Option {label}: {text}")

    if not sections:
        raise RuntimeError(
            "The Options block was found, but no answer options could be parsed."
        )

    return sections


def _extract_usage(response) -> dict:
    response_metadata = getattr(response, "response_metadata", {}) or {}
    usage_metadata = getattr(response, "usage_metadata", {}) or {}
    token_usage = response_metadata.get("token_usage", {}) or {}

    model_served = (
        response_metadata.get("model_name")
        or response_metadata.get("model")
        or ""
    )

    prompt_tokens = (
        usage_metadata.get("input_tokens")
        if usage_metadata.get("input_tokens") is not None
        else token_usage.get("prompt_tokens")
    )

    completion_tokens = (
        usage_metadata.get("output_tokens")
        if usage_metadata.get("output_tokens") is not None
        else token_usage.get("completion_tokens")
    )

    return {
        "model_served": model_served,
        "finish_reason": response_metadata.get("finish_reason"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def _tool_output_char_counts(messages) -> list[int]:
    counts = []

    for message in messages:
        role, content = _message_role_and_content(message)
        if role == "tool":
            counts.append(len(content))

    return counts


def get_research_tools(config: RunnableConfig):
    search_tool = get_search_tool(config)
    tool_list = [search_tool, Section]
    return tool_list, {tool.name: tool for tool in tool_list}


async def deterministic_planner(
    state: ReportState,
    config: RunnableConfig,
):
    original_user_content = _get_original_user_content(state["messages"])
    sections_list = _parse_fixed_option_sections(state["messages"])

    logger.info(
        "Fixed option plan (%d sections): %s",
        len(sections_list),
        sections_list,
    )

    first_section = sections_list[0]

    researcher_message = {
        "role": "user",
        "content": (
            "Original clinical question and all answer options:\n\n"
            f"{original_user_content}\n\n"
            "Assigned option-specific research section:\n\n"
            f"{first_section}\n\n"
            "Research ONLY this answer option in relation to the original "
            "clinical question. Retrieve evidence directly relevant to "
            "this patient's clinical, imaging, pathological, anatomical, "
            "or technical context as appropriate. Keep the final Section "
            "concise: target about 250-400 words total, including the "
            "source list. Use a compact structure with (1) evidence "
            "supporting the option, (2) evidence arguing against it, "
            "and (3) a brief neutral synthesis. Do not repeat the full "
            "case history and do not write a general textbook review. "
            "Use at most 5 highly relevant sources. Do not drift into "
            "unrelated anatomical sites, age groups, or manifestations "
            "unless they are directly relevant to this differential. "
            "Do not state which answer option is correct."
        ),
    }

    return Command(
        goto=[
            Send(
                "research_team",
                {
                    "section": first_section,
                    "messages": [researcher_message],
                    "search_iterations": 0,
                    "section_error_retries": 0,
                    "report_model_calls": [],
                },
            )
        ],
        update={"sections": sections_list},
    )


async def dispatch_next_section(
    state: ReportState,
    config: RunnableConfig,
):
    sections = state.get("sections", [])
    completed_sections = state.get("completed_sections", [])

    completed_count = len(completed_sections)
    total_count = len(sections)

    logger.info(
        "Sequential research progress: %d/%d sections completed",
        completed_count,
        total_count,
    )

    if total_count == 0:
        raise RuntimeError("No fixed option sections are available.")

    if completed_count >= total_count:
        return Command(goto="finalize_report")

    next_section = sections[completed_count]
    original_user_content = _get_original_user_content(state["messages"])

    researcher_message = {
        "role": "user",
        "content": (
            "Original clinical question and all answer options:\n\n"
            f"{original_user_content}\n\n"
            "Assigned option-specific research section:\n\n"
            f"{next_section}\n\n"
            "Research ONLY this answer option in relation to the original "
            "clinical question. Retrieve evidence directly relevant to "
            "this patient's clinical, imaging, pathological, anatomical, "
            "or technical context as appropriate. Keep the final Section "
            "concise: target about 250-400 words total, including the "
            "source list. Use a compact structure with (1) evidence "
            "supporting the option, (2) evidence arguing against it, "
            "and (3) a brief neutral synthesis. Do not repeat the full "
            "case history and do not write a general textbook review. "
            "Use at most 5 highly relevant sources. Do not drift into "
            "unrelated anatomical sites, age groups, or manifestations "
            "unless they are directly relevant to this differential. "
            "Do not state which answer option is correct."
        ),
    }

    return Command(
        goto=[
            Send(
                "research_team",
                {
                    "section": next_section,
                    "messages": [researcher_message],
                    "search_iterations": 0,
                    "section_error_retries": 0,
                    "report_model_calls": [],
                },
            )
        ]
    )


async def finalize_report(
    state: ReportState,
    config: RunnableConfig,
):
    sections = state.get("sections", [])
    completed_sections = state.get("completed_sections", [])

    if not sections:
        raise RuntimeError("Finalizer reached without fixed option sections.")

    if len(completed_sections) != len(sections):
        raise RuntimeError(
            "Finalizer reached before all option sections completed: "
            f"{len(completed_sections)}/{len(sections)}."
        )

    return {
        "final_report": _build_final_report(completed_sections),
        "report_model_calls": state.get("report_model_calls", []),
    }


_PLACEHOLDER_VALUES = {
    "",
    "x",
    "xx",
    "xxx",
    "placeholder",
    "n/a",
    "na",
    "none",
    "unknown",
    "tbd",
    "-",
}


def _normalise_qc_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _looks_like_placeholder(value) -> bool:
    text_value = _normalise_qc_text(value).lower().strip(" .,:;[](){}\"'")
    return text_value in _PLACEHOLDER_VALUES


def _has_plausible_source_reference(value) -> bool:
    source_text = str(value or "").strip()

    if len(source_text) < 20:
        return False

    if re.search(r"https?://\S+", source_text, flags=re.IGNORECASE):
        return True

    if re.search(r"\b[\w.-]+\.txt\b", source_text, flags=re.IGNORECASE):
        return True

    return False


def _section_quality_issues(section_args: dict) -> list[str]:
    issues: list[str] = []

    name = _normalise_qc_text(section_args.get("name"))
    supporting = _normalise_qc_text(section_args.get("supporting_evidence"))
    contradictory = _normalise_qc_text(section_args.get("contradictory_evidence"))
    synthesis = _normalise_qc_text(section_args.get("synthesis"))
    sources = str(section_args.get("sources") or "").strip()

    if len(name) < 4 or _looks_like_placeholder(name):
        issues.append("name is missing or placeholder-like")

    if len(supporting) < 80 or _looks_like_placeholder(supporting):
        issues.append(
            "supporting_evidence is too short or placeholder-like "
            "(minimum 80 characters)"
        )

    if len(contradictory) < 80 or _looks_like_placeholder(contradictory):
        issues.append(
            "contradictory_evidence is too short or placeholder-like "
            "(minimum 80 characters)"
        )

    if len(synthesis) < 60 or _looks_like_placeholder(synthesis):
        issues.append(
            "synthesis is too short or placeholder-like "
            "(minimum 60 characters)"
        )

    if not _has_plausible_source_reference(sources):
        issues.append(
            "sources is missing/truncated; include at least one complete "
            ".txt article identifier or HTTP(S) URL"
        )

    return issues


async def research_agent(
    state: SectionState,
    config: RunnableConfig,
):
    configurable = Configuration.from_runnable_config(config)

    researcher_model = get_config_value(configurable.researcher_model)
    llm = init_chat_model(model=researcher_model, temperature=float(configurable.temperature), max_tokens=int(configurable.max_tokens))

    search_iterations = state.get("search_iterations", 0)
    max_search_depth = configurable.max_search_depth

    if search_iterations < max_search_depth:
        research_tool_list, _ = get_research_tools(config)
        extra_instruction = (
            f"\n\nSearch progress: {search_iterations} of "
            f"{max_search_depth} allowed search iterations have been used. "
            "You may search again if additional evidence is necessary, or "
            "call the Section tool when the section is complete."
        )
    else:
        research_tool_list = [Section]
        extra_instruction = (
            "\n\nThe maximum number of search iterations has been reached. "
            "Do not request any further searches. Use only the evidence "
            "already collected and call the Section tool now to complete "
            "this research section."
        )

    concise_section_instruction = (
        "\n\nOUTPUT CONSTRAINTS FOR THE FINAL Section TOOL CALL:\n"
        "- You MUST fill all five fields: name, supporting_evidence, "
        "contradictory_evidence, synthesis, and sources.\n"
        "- supporting_evidence: at most 3 short bullet points.\n"
        "- contradictory_evidence: at most 3 short bullet points.\n"
        "- synthesis: 1-3 short neutral sentences about this option only.\n"
        "- sources: at most 5 source article identifiers, one per line. Each source "
        "must be a complete article identifier ending in .txt, or a complete "
        "HTTP(S) URL. Do not output shortened/truncated source names.\n"
        "- Never use placeholder values such as x, placeholder, N/A, or TBD.\n"
        "- Keep the entire tool call concise; do not write a textbook review.\n"
        "- Do not repeat the full case history.\n"
        "- Do not state, imply, or rank which answer option is correct.\n"
        "- Finish every field cleanly; completeness is more important than detail."
    )

    system_prompt = (
        RESEARCH_INSTRUCTIONS.format(
            section_description=state["section"]
        )
        + concise_section_instruction
        + extra_instruction
    )

    bound_llm = llm.bind_tools(
        research_tool_list,
        parallel_tool_calls=False,
        tool_choice="required",
    )

    input_messages = (
        [{"role": "system", "content": system_prompt}]
        + state["messages"]
    )

    response = await bound_llm.ainvoke(input_messages)

    call_log = _extract_usage(response)
    call_log["tool_output_chars"] = _tool_output_char_counts(input_messages)

    return {
        "messages": [response],
        "report_model_calls": [call_log],
    }


async def research_agent_tools(
    state: SectionState,
    config: RunnableConfig,
):
    section_error_retries = state.get("section_error_retries", 0)
    result = []
    completed_section = None
    search_iterations = state.get("search_iterations", 0)

    research_tool_list, research_tools_by_name = get_research_tools(config)
    search_tool_name = research_tool_list[0].name

    for tool_call in getattr(state["messages"][-1], "tool_calls", []):
        tool_name = tool_call["name"]

        if tool_name not in research_tools_by_name:
            raise KeyError(f"Unknown research tool requested: {tool_name}")

        tool_obj = research_tools_by_name[tool_name]

        if tool_name == "Section":
            section_args = dict(tool_call.get("args") or {})
            section_args.setdefault("name", state.get("section", "Section"))

            quality_issues = _section_quality_issues(section_args)

            if quality_issues:
                section_error_retries += 1

                if section_error_retries >= 3:
                    raise RuntimeError(
                        "Repeated invalid Section tool call for "
                        f"{state.get('section')!r}; QC issues: "
                        + "; ".join(quality_issues)
                    )

                error_content = (
                    "Error: The Section tool call failed output QC. "
                    "Please call the Section tool again and correct these issue(s): "
                    + "; ".join(quality_issues)
                    + ". RETRY COMPACT MODE: do not perform any new search. "
                    "Call Section immediately using only evidence already retrieved. "
                    "Keep the retry substantially shorter than the previous call: "
                    "use exactly 1 short supporting bullet, exactly 1 short "
                    "contradictory bullet, and exactly 1 short neutral synthesis "
                    "sentence. In sources, include 1 to 3 COMPLETE source article "
                    "identifiers copied verbatim from the retrieved evidence; each "
                    "identifier must end in .txt, or use a complete HTTP(S) URL. "
                    "Do not shorten source identifiers. Completeness of every field "
                    "takes priority over detail. Do not use placeholders."
                )

                result.append(
                    {
                        "role": "tool",
                        "content": error_content,
                        "name": tool_name,
                        "tool_call_id": tool_call["id"],
                    }
                )

                logger.warning(
                    "Section QC failed for section %r "
                    "(retry %d/3): %s",
                    state.get("section"),
                    section_error_retries,
                    quality_issues,
                )
                continue

            tool_args = section_args
        else:
            tool_args = tool_call.get("args") or {}

        stage_token = RETRIEVAL_STAGE.set("research")
        section_token = RETRIEVAL_SECTION.set(state.get("section"))

        try:
            if hasattr(tool_obj, "ainvoke"):
                observation = await tool_obj.ainvoke(tool_args)
            else:
                observation = tool_obj.invoke(tool_args)
        finally:
            RETRIEVAL_STAGE.reset(stage_token)
            RETRIEVAL_SECTION.reset(section_token)

        result.append(
            {
                "role": "tool",
                "content": observation,
                "name": tool_name,
                "tool_call_id": tool_call["id"],
            }
        )

        if tool_name == "Section":
            completed_section = observation
        elif tool_name == search_tool_name:
            search_iterations += 1
            logger.info(
                "Research search iteration %d completed for section: %s",
                search_iterations,
                state.get("section"),
            )

    if completed_section:
        return {
            "messages": result,
            "completed_sections": [completed_section],
            "search_iterations": search_iterations,
            "section_error_retries": section_error_retries,
        }

    return {
        "messages": result,
        "search_iterations": search_iterations,
        "section_error_retries": section_error_retries,
    }


async def research_agent_should_continue(
    state: SectionState,
) -> Literal["research_agent_tools", END]:
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "research_agent_tools"

    return END


async def research_tools_should_continue(
    state: SectionState,
) -> Literal["research_agent", END]:
    if state.get("completed_sections"):
        return END

    return "research_agent"


research_builder = StateGraph(
    SectionState,
    output=SectionOutputState,
    config_schema=Configuration,
)

research_builder.add_node("research_agent", research_agent)
research_builder.add_node("research_agent_tools", research_agent_tools)
research_builder.add_edge(START, "research_agent")

research_builder.add_conditional_edges(
    "research_agent",
    research_agent_should_continue,
    {
        "research_agent_tools": "research_agent_tools",
        END: END,
    },
)

research_builder.add_conditional_edges(
    "research_agent_tools",
    research_tools_should_continue,
    {
        "research_agent": "research_agent",
        END: END,
    },
)

supervisor_builder = StateGraph(
    ReportState,
    input=MessagesState,
    output=ReportStateOutput,
    config_schema=Configuration,
)

supervisor_builder.add_node("deterministic_planner", deterministic_planner)
supervisor_builder.add_node("research_team", research_builder.compile())
supervisor_builder.add_node("dispatch_next_section", dispatch_next_section)
supervisor_builder.add_node("finalize_report", finalize_report)

supervisor_builder.add_edge(START, "deterministic_planner")
supervisor_builder.add_edge("research_team", "dispatch_next_section")
supervisor_builder.add_edge("finalize_report", END)

graph = supervisor_builder.compile()
