"""
src/noretrieval.py
Created on October 22, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field


def _install_noretrieval_behavior(
    local_search,
    agentic_workflow,
) -> None:

    @tool
    def noretrieval_search(
        queries: list[str] | str | None = None,
        max_results: int = 10,
    ) -> str:
        """
        Return no documents for every query and log each query
        with an empty result list for the no-retrieval condition.
        """
        if queries is None:
            query_list = []
        elif isinstance(queries, str):
            query_list = [queries]
        else:
            query_list = list(queries)

        for query in query_list:
            query = str(query).strip()

            if not query:
                continue

            log_record = {
                "stage":
                    local_search.RETRIEVAL_STAGE.get(),

                "section":
                    local_search.RETRIEVAL_SECTION.get(),

                "query":
                    query,

                "results":
                    [],
            }

            append_log = getattr(
                local_search,
                "_append_retrieval_log",
                None,
            )

            if callable(append_log):
                append_log(log_record)
            else:
                retrieval_log = getattr(
                    local_search,
                    "RETRIEVAL_LOG",
                    None,
                )

                if hasattr(retrieval_log, "get") and hasattr(
                    retrieval_log,
                    "set",
                ):
                    current_log = retrieval_log.get()
                    if current_log is None:
                        current_log = []
                        retrieval_log.set(current_log)
                    current_log.append(log_record)
                else:
                    retrieval_log.append(log_record)

        return "No documents found."

    @tool
    class Section(BaseModel):
        """
        Compact structured evidence for one answer option
        in the no-retrieval condition.
        """

        name: str = Field(
            description=(
                "The assigned answer option, "
                "e.g. 'Option A: ...'."
            ),
        )

        supporting_evidence: str = Field(
            description=(
                "Up to 3 short bullet points supporting "
                "this option in the context of the "
                "original clinical question."
            ),
        )

        contradictory_evidence: str = Field(
            description=(
                "Up to 3 short bullet points arguing "
                "against this option in the context of "
                "the original clinical question."
            ),
        )

        synthesis: str = Field(
            description=(
                "A neutral 1-3 sentence synthesis of "
                "this option only. Do not rank options "
                "and do not state which option is correct."
            ),
        )

        sources: str = Field(
            default="",
            description=(
                "Source article identifiers actually used. "
                "In the no-retrieval condition this may be empty."
            ),
        )

    agentic_workflow.local_corpus_search = (
        noretrieval_search
    )
    agentic_workflow.Section = (
        Section
    )

    original_section_qc = (
        agentic_workflow._section_quality_issues
    )

    def noretrieval_section_qc(
        section_args: dict,
    ) -> list[str]:
        issues = original_section_qc(
            section_args
        )

        return [
            issue
            for issue in issues
            if not issue.startswith(
                "sources is missing/truncated"
            )
        ]

    agentic_workflow._section_quality_issues = (
        noretrieval_section_qc
    )
