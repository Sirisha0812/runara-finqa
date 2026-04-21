"""LangChain + LangGraph FinQA chatbot powered by a vLLM-served HF model."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

try:
    from sympy import N, sympify

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from src.config import config
from src.logger import LoggerContext, get_logger
from src.retriever import FinQARetriever

logger = get_logger(__name__)


REASON_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a financial analyst answering FinQA-style questions.

Use only the provided evidence chunks. Financial answers must stay anchored to exact values from the evidence.

Respond with this exact structure:
REASONING:
- step 1
- step 2

CALCULATION:
<single arithmetic expression using only numbers and + - * / ( ) or NONE>

PRELIMINARY_ANSWER:
<short answer with units if available>

EVIDENCE_IDS:
- <chunk_id>
- <chunk_id>""",
        ),
        (
            "human",
            """Question:
{question}

Evidence Chunks:
{evidence}

Previous verification feedback:
{feedback}""",
        ),
    ]
)


VERIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are checking whether a FinQA answer is properly supported.

Review the reasoning against the evidence. Pay special attention to:
1. unsupported numbers
2. wrong arithmetic
3. missing units
4. claims that are not grounded in the evidence

Respond with this exact format:
VERIFICATION_STATUS: PASS | FAIL | UNCERTAIN
CONFIDENCE: HIGH | MEDIUM | LOW
ISSUES:
- <issue or NONE>""",
        ),
        (
            "human",
            """Question:
{question}

Evidence Chunks:
{evidence}

Reasoning Draft:
{reasoning}

Calculation Result:
{calculation_result}""",
        ),
    ]
)


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are formatting the final answer for a financial QA chatbot.

Keep it concise and factual. If verification failed or remained uncertain, state that clearly.

Return:
FINAL_ANSWER: <answer>
EXPLANATION: <one sentence>""",
        ),
        (
            "human",
            """Question:
{question}

Reasoning Draft:
{reasoning}

Preliminary Answer:
{preliminary_answer}

Calculation Result:
{calculation_result}

Verification Status:
{verification_status}

Verification Confidence:
{verification_confidence}

Verification Issues:
{verification_issues}""",
        ),
    ]
)


class AgentState(TypedDict):
    """State passed through the LangGraph workflow."""

    question: str
    example: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]
    reasoning: str
    preliminary_answer: str
    cited_evidence_ids: List[str]
    calculation_expression: Optional[str]
    calculation_result: Optional[str]
    verification_status: str
    verification_confidence: str
    verification_issues: List[str]
    retry_count: int
    failure_feedback: str
    final_answer: str
    trace: List[Dict[str, Any]]
    node_traces: List[Dict[str, Any]]


class FinQAAgent:
    """LangGraph workflow for FinQA reasoning over a single financial document."""

    def __init__(self) -> None:
        self.retriever = FinQARetriever()
        self.llm = ChatOpenAI(
            model=config.vllm.model,
            api_key=config.vllm.api_key,
            base_url=config.vllm.api_base,
            temperature=config.vllm.temperature,
            max_completion_tokens=config.vllm.max_tokens,
            timeout=config.vllm.timeout_seconds,
        )
        self.max_retries = config.agent.max_retries
        self.graph = self._build_graph()

        logger.info(
            "agent_initialized",
            model=config.vllm.model,
            api_base=config.vllm.api_base,
            max_retries=self.max_retries,
            retriever_scope="document_local",
        )

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("calculate", self._calculate_node)
        workflow.add_node("verify", self._verify_node)
        workflow.add_node("answer", self._answer_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "reason")
        workflow.add_edge("reason", "calculate")
        workflow.add_edge("calculate", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"retry": "reason", "answer": "answer"},
        )
        workflow.add_edge("answer", END)
        return workflow.compile()

    def _route_after_verify(self, state: AgentState) -> str:
        if (
            state["verification_status"] == "FAIL"
            and state["retry_count"] < self.max_retries
        ):
            return "retry"
        return "answer"

    @staticmethod
    def _format_evidence(chunks: List[Dict[str, Any]]) -> str:
        formatted = []
        for chunk in chunks:
            formatted.append(
                f"[{chunk['chunk_id']}] ({chunk['chunk_type']}, score={chunk['hybrid_score']:.3f}) "
                f"{chunk['text']}"
            )
        return "\n".join(formatted)

    def _record_node_trace(
        self,
        state: AgentState,
        node: str,
        start_time: float,
        status: str = "ok",
        **kwargs: Any,
    ) -> None:
        state["node_traces"].append(
            {
                "node": node,
                "duration_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "status": status,
                **kwargs,
            }
        )

    def _retrieve_node(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        question = state["question"]

        with LoggerContext(logger, "retrieve_node", question=question):
            chunks = self.retriever.retrieve_for_example(
                question=question,
                example=state["example"],
                k=config.agent.retrieval_top_k,
            )
            state["retrieved_chunks"] = chunks
            state["trace"].append(
                {
                    "node": "retrieve",
                    "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in chunks],
                    "top_hybrid_score": chunks[0]["hybrid_score"] if chunks else None,
                }
            )

        self._record_node_trace(state, "retrieve", start, status="ok")
        return state

    def _reason_node(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        feedback = state["failure_feedback"] or "NONE"
        if state["verification_status"] == "FAIL":
            state["retry_count"] += 1

        evidence = self._format_evidence(state["retrieved_chunks"])
        evidence = evidence[: config.agent.max_context_characters]

        with LoggerContext(
            logger,
            "reason_node",
            question=state["question"],
            retry_count=state["retry_count"],
        ):
            prompt = REASON_PROMPT.invoke(
                {
                    "question": state["question"],
                    "evidence": evidence,
                    "feedback": feedback,
                }
            )
            response = self.llm.invoke(prompt)
            content = response.content if isinstance(response.content, str) else str(response.content)

        reasoning, prelim, calc_expr, evidence_ids = _parse_reasoning_output(content)
        state["reasoning"] = reasoning
        state["preliminary_answer"] = prelim
        state["calculation_expression"] = calc_expr
        state["cited_evidence_ids"] = evidence_ids
        state["trace"].append(
            {
                "node": "reason",
                "retry_count": state["retry_count"],
                "calculation_expression": calc_expr,
                "preliminary_answer": prelim,
                "cited_evidence_ids": evidence_ids,
            }
        )
        self._record_node_trace(
            state,
            "reason",
            start,
            status="ok",
            retry_count=state["retry_count"],
        )
        return state

    def _calculate_node(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        expr = state["calculation_expression"]
        if not expr:
            state["calculation_result"] = None
            state["trace"].append({"node": "calculate", "skipped": True})
            self._record_node_trace(state, "calculate", start, status="skipped")
            return state

        result = _evaluate_expression(expr)
        state["calculation_result"] = result
        state["trace"].append(
            {
                "node": "calculate",
                "expression": expr,
                "result": result,
            }
        )
        self._record_node_trace(state, "calculate", start, status="ok")
        return state

    def _verify_node(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        evidence = self._format_evidence(state["retrieved_chunks"])
        evidence = evidence[: config.agent.max_context_characters]

        with LoggerContext(
            logger,
            "verify_node",
            question=state["question"],
            retry_count=state["retry_count"],
        ):
            prompt = VERIFY_PROMPT.invoke(
                {
                    "question": state["question"],
                    "evidence": evidence,
                    "reasoning": state["reasoning"],
                    "calculation_result": state["calculation_result"] or "NONE",
                }
            )
            response = self.llm.invoke(prompt)
            content = response.content if isinstance(response.content, str) else str(response.content)

        status, confidence, issues = _parse_verification_output(content)
        heuristic_issues = _run_heuristic_checks(
            reasoning=state["reasoning"],
            calculation_expression=state["calculation_expression"],
            calculation_result=state["calculation_result"],
            preliminary_answer=state["preliminary_answer"],
            retrieved_chunks=state["retrieved_chunks"],
        )
        merged_issues = list(dict.fromkeys(issues + heuristic_issues))

        if heuristic_issues and status == "PASS":
            status = "FAIL"
            confidence = "MEDIUM"

        state["verification_status"] = status
        state["verification_confidence"] = confidence
        state["verification_issues"] = merged_issues
        state["failure_feedback"] = "\n".join(f"- {issue}" for issue in merged_issues) or "NONE"
        state["trace"].append(
            {
                "node": "verify",
                "status": status,
                "confidence": confidence,
                "issues": merged_issues,
            }
        )
        self._record_node_trace(state, "verify", start, status=status.lower())
        return state

    def _answer_node(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        with LoggerContext(
            logger,
            "answer_node",
            question=state["question"],
            verification_status=state["verification_status"],
        ):
            prompt = ANSWER_PROMPT.invoke(
                {
                    "question": state["question"],
                    "reasoning": state["reasoning"],
                    "preliminary_answer": state["preliminary_answer"] or "NONE",
                    "calculation_result": state["calculation_result"] or "NONE",
                    "verification_status": state["verification_status"],
                    "verification_confidence": state["verification_confidence"],
                    "verification_issues": "\n".join(
                        f"- {issue}" for issue in state["verification_issues"]
                    )
                    or "- NONE",
                }
            )
            response = self.llm.invoke(prompt)
            content = response.content if isinstance(response.content, str) else str(response.content)

        state["final_answer"] = content
        state["trace"].append({"node": "answer", "final_answer": content})
        self._record_node_trace(state, "answer", start, status="ok")
        return state

    def run(self, question: str, example: Dict[str, Any]) -> Dict[str, Any]:
        """Run the workflow for a question plus its financial document."""
        initial_state: AgentState = {
            "question": question,
            "example": example,
            "retrieved_chunks": [],
            "reasoning": "",
            "preliminary_answer": "",
            "cited_evidence_ids": [],
            "calculation_expression": None,
            "calculation_result": None,
            "verification_status": "SKIPPED",
            "verification_confidence": "N/A",
            "verification_issues": [],
            "retry_count": 0,
            "failure_feedback": "",
            "final_answer": "",
            "trace": [],
            "node_traces": [],
        }

        with LoggerContext(logger, "agent_run", question=question):
            return self.graph.invoke(initial_state)


def _extract_block(text: str, start_label: str, end_labels: List[str]) -> str:
    if end_labels:
        end_pattern = "|".join(map(re.escape, end_labels))
        pattern = rf"{re.escape(start_label)}\s*\n(.*?)(?=\n(?:{end_pattern})\s*\n|\Z)"
    else:
        pattern = rf"{re.escape(start_label)}\s*\n(.*)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_reasoning_output(text: str) -> Tuple[str, str, Optional[str], List[str]]:
    reasoning = _extract_block(
        text,
        "REASONING:",
        ["CALCULATION:", "PRELIMINARY_ANSWER:", "EVIDENCE_IDS:"],
    )
    calculation = _extract_block(
        text,
        "CALCULATION:",
        ["PRELIMINARY_ANSWER:", "EVIDENCE_IDS:"],
    )
    preliminary_answer = _extract_block(
        text,
        "PRELIMINARY_ANSWER:",
        ["EVIDENCE_IDS:"],
    )
    evidence_block = _extract_block(text, "EVIDENCE_IDS:", [])
    evidence_ids = [
        line.strip().lstrip("- ").strip()
        for line in evidence_block.splitlines()
        if line.strip()
    ]

    calculation = calculation.strip()
    if not calculation or calculation.upper() == "NONE":
        calculation = None

    return reasoning, preliminary_answer, calculation, evidence_ids


def _parse_verification_output(text: str) -> Tuple[str, str, List[str]]:
    status_match = re.search(r"VERIFICATION_STATUS:\s*(PASS|FAIL|UNCERTAIN)", text)
    confidence_match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", text)
    issues_block = _extract_block(text, "ISSUES:", [])

    issues = []
    for line in issues_block.splitlines():
        cleaned = line.strip().lstrip("- ").strip()
        if cleaned and cleaned.upper() != "NONE":
            issues.append(cleaned)

    return (
        status_match.group(1) if status_match else "UNCERTAIN",
        confidence_match.group(1) if confidence_match else "LOW",
        issues,
    )


def _evaluate_expression(expr: str) -> str:
    try:
        if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", expr):
            raise ValueError(f"Unsafe expression refused: {expr}")

        if SYMPY_AVAILABLE:
            result = float(N(sympify(expr)))
        else:
            result = float(eval(expr))  # noqa: S307
        return str(round(result, 6))
    except Exception as exc:  # pragma: no cover - defensive path
        return f"ERROR: {exc}"


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", text)


def _run_heuristic_checks(
    reasoning: str,
    calculation_expression: Optional[str],
    calculation_result: Optional[str],
    preliminary_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> List[str]:
    issues: List[str] = []
    evidence_text = " ".join(chunk["text"] for chunk in retrieved_chunks).lower().replace(",", "")

    if calculation_expression:
        for number in _extract_numbers(calculation_expression):
            if number in {"100", "1000"}:
                continue
            if number.replace(",", "") not in evidence_text:
                issues.append(f"Calculation uses unsupported number {number}.")

    if calculation_result and not calculation_result.startswith("ERROR"):
        prelim_num = _extract_first_number(preliminary_answer)
        calc_num = _extract_first_number(calculation_result)
        if prelim_num is not None and calc_num is not None and abs(prelim_num - calc_num) > 0.01:
            issues.append("Preliminary answer does not match the computed result.")

    if calculation_result and calculation_result.startswith("ERROR"):
        issues.append("Calculation step failed to execute safely.")

    if not retrieved_chunks:
        issues.append("No evidence chunks were retrieved.")

    return issues


def _extract_first_number(text: str) -> Optional[float]:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


if __name__ == "__main__":
    from src.data_loader import load_finqa_dataset

    dataset = load_finqa_dataset(split="validation")
    example = dataset[0]
    agent = FinQAAgent()
    result = agent.run(example["question"], example)
    print(result["final_answer"])
