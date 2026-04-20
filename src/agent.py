"""LangGraph-based FinQA agent with retrieval, reasoning, calculation, verification, and retry."""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langgraph.graph import END, StateGraph
from openai import OpenAI
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

MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Full state passed through every node of the LangGraph workflow."""
    question: str
    retrieved_docs: List[Dict[str, Any]]
    reasoning: str
    calculation_expression: Optional[str]
    calculation_result: Optional[str]
    verification_status: str        # "PASS" | "FAIL" | "UNCERTAIN" | "SKIPPED"
    verification_issues: List[str]  # empty when status is PASS or SKIPPED
    verification_confidence: str    # "HIGH" | "MEDIUM" | "LOW" | "N/A"
    retry_count: int                # incremented each time reason_node runs as a retry
    failure_feedback: Optional[str] # injected into reason_node prompt on retry
    retry_exhausted: bool           # True when FAIL and retry_count >= MAX_RETRIES
    final_answer: str
    trace: List[Dict[str, Any]]
    node_traces: List[Dict[str, Any]]  # timing audit trail: one entry per node execution


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

REASON_SYSTEM = """You are a financial analysis expert. Given a question and relevant financial documents, reason step by step to find the answer.

Your response MUST follow this exact format with no extra text between sections:

REASONING:
[Step-by-step reasoning referencing specific numbers from the documents]

CALCULATION:
[A single Python arithmetic expression using only numbers and + - * / ( ).
Example: 3.8 / 0.01
Write NONE if the answer is directly readable from the documents with no arithmetic needed]

PRELIMINARY_ANSWER:
[Your answer before applying the calculation, or the final answer if CALCULATION is NONE]"""

VERIFY_SYSTEM = """You are a financial fact-checker. Your job is to verify that a reasoning chain and its answer are consistent with the source documents provided.

Check three things:
1. Are the numbers used in the reasoning actually present in (or derivable from) the retrieved documents?
2. Is the arithmetic expression correct for the stated reasoning steps?
3. Does the calculation result match the preliminary answer?

Your response MUST follow this exact format:

VERIFICATION_STATUS: PASS | FAIL | UNCERTAIN
CONFIDENCE: HIGH | MEDIUM | LOW
ISSUES:
- [Issue 1, or write NONE if no issues]
- [Issue 2, etc.]"""

ANSWER_SYSTEM = """You are a financial analysis assistant. Given the full reasoning trace, calculation result, and verification outcome, provide a concise final answer following the instruction below exactly."""


# ---------------------------------------------------------------------------
# Answer-mode instructions (injected per routing outcome)
# ---------------------------------------------------------------------------

_ANSWER_MODE_INSTRUCTIONS: Dict[str, str] = {
    # PASS + HIGH  — clean, confident answer
    "CONFIDENT": (
        "Verification PASSED with HIGH confidence. "
        "State the answer directly and confidently.\n\n"
        "Format:\nFINAL_ANSWER: [value]\nEXPLANATION: [one sentence]"
    ),
    # PASS + MEDIUM — answer with a light caveat
    "CAVEAT": (
        "Verification PASSED but with MEDIUM confidence. "
        "State the answer and add a brief caveat noting that confidence is medium.\n\n"
        "Format:\nFINAL_ANSWER: [value]\nEXPLANATION: [one sentence including caveat]"
    ),
    # UNCERTAIN — flag for human review
    "HUMAN_REVIEW": (
        "Verification returned UNCERTAIN. "
        "State the best available answer but clearly flag it for human review.\n\n"
        "Format:\nFINAL_ANSWER: [value] (flagged for human review)\nEXPLANATION: [one sentence noting uncertainty]"
    ),
    # FAIL + retries exhausted — low-confidence warning
    "MAX_RETRIES": (
        f"Verification FAILED after {MAX_RETRIES} retry attempts. "
        "State the best available answer but clearly warn that max retries were reached and confidence is low.\n\n"
        "Format:\nFINAL_ANSWER: [value] (low confidence — max retries reached)\nEXPLANATION: [one sentence noting failure]"
    ),
}


def _answer_mode(v_status: str, v_confidence: str, retry_exhausted: bool) -> str:
    """Map verification outcome to one of the four answer modes."""
    if retry_exhausted:
        return "MAX_RETRIES"
    if v_status == "PASS" and v_confidence == "HIGH":
        return "CONFIDENT"
    if v_status == "PASS":          # MEDIUM or LOW
        return "CAVEAT"
    if v_status == "UNCERTAIN":
        return "HUMAN_REVIEW"
    return "CAVEAT"                 # FAIL routed here only if somehow past routing guard


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class FinQAAgent:
    """LangGraph-based agent for FinQA numerical reasoning over financial reports."""

    def __init__(self, index_path: str = "./data/faiss_index"):
        self.retriever = FinQARetriever(index_path=index_path)
        self.llm = OpenAI(
            base_url=config.vllm.api_base,
            api_key="not-needed",
        )
        self.model = config.vllm.model
        self.graph = self._build_graph()

        logger.info(
            "agent_initialized",
            model=self.model,
            index_path=index_path,
            sympy_available=SYMPY_AVAILABLE,
            max_retries=MAX_RETRIES,
        )

    # -----------------------------------------------------------------------
    # Graph construction
    # -----------------------------------------------------------------------

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("calculator", self._calculator_node)
        workflow.add_node("verifier", self._verifier_node)
        workflow.add_node("answer", self._answer_node)

        # Fixed edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "reason")
        workflow.add_edge("reason", "calculator")
        workflow.add_edge("calculator", "verifier")
        workflow.add_edge("answer", END)

        # Conditional edge: verifier → reason (retry) | answer
        workflow.add_conditional_edges(
            "verifier",
            self._route_after_verifier,
            {
                "retry": "reason",
                "answer": "answer",
            },
        )

        return workflow.compile()

    # -----------------------------------------------------------------------
    # Routing function
    # -----------------------------------------------------------------------

    def _route_after_verifier(self, state: AgentState) -> str:
        """
        Route to 'retry' (back to reason_node) when verification FAILs and
        retries remain; otherwise route to 'answer'.
        """
        if state["verification_status"] == "FAIL" and state["retry_count"] < MAX_RETRIES:
            logger.info(
                "routing_to_retry",
                retry_count=state["retry_count"],
                max_retries=MAX_RETRIES,
            )
            return "retry"

        logger.info(
            "routing_to_answer",
            verification_status=state["verification_status"],
            retry_count=state["retry_count"],
            retry_exhausted=state["retry_exhausted"],
        )
        return "answer"

    # -----------------------------------------------------------------------
    # Node: retrieve
    # -----------------------------------------------------------------------

    def _retrieve_node(self, state: AgentState) -> AgentState:
        """Hybrid FAISS + BM25 retrieval of the top-4 most relevant documents."""
        question = state["question"]
        _t0 = time.perf_counter()

        with LoggerContext(logger, "retrieve_node", question=question):
            docs = self.retriever.retrieve_hybrid(question, k=4)

            state["retrieved_docs"] = docs
            state["trace"].append({
                "node": "retrieve",
                "num_docs": len(docs),
                "top_hybrid_score": docs[0]["hybrid_score"] if docs else None,
                "retrieved_questions": [d["question"] for d in docs],
            })

        state["node_traces"].append({"node": "retrieve", "duration_ms": round((time.perf_counter() - _t0) * 1000, 2), "status": "ok"})
        return state

    # -----------------------------------------------------------------------
    # Node: reason
    # -----------------------------------------------------------------------

    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a compact context block."""
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(
                f"--- Document {i} "
                f"(hybrid_score={doc.get('hybrid_score', 0):.3f}) ---\n"
                f"{doc['context'][:1500]}"
            )
        return "\n\n".join(parts)

    def _reason_node(self, state: AgentState) -> AgentState:
        """
        Send question + retrieved context to the LLM for step-by-step reasoning.

        On retry runs (failure_feedback is set):
        - Increments retry_count
        - Resets previous verification state for a clean slate
        - Injects the verifier's issues into the prompt so the LLM can correct them
        """
        question = state["question"]
        failure_feedback = state.get("failure_feedback")
        is_retry = failure_feedback is not None
        _t0 = time.perf_counter()

        if is_retry:
            state["retry_count"] += 1
            # Reset verification fields so the new attempt starts clean
            state["verification_status"] = "SKIPPED"
            state["verification_issues"] = []
            state["verification_confidence"] = "N/A"

        attempt_num = state["retry_count"] + 1  # 1-based for display
        context_str = self._format_context(state["retrieved_docs"])

        retry_block = ""
        if is_retry:
            retry_block = (
                f"\n\n--- CORRECTION REQUIRED (Attempt {attempt_num} of {MAX_RETRIES + 1}) ---\n"
                f"Your previous reasoning was rejected by the verifier. Issues found:\n"
                f"{failure_feedback}\n"
                "Please carefully re-examine the documents and correct these specific problems."
            )

        user_prompt = (
            f"Question: {question}\n\n"
            f"Retrieved Financial Documents:\n{context_str}"
            f"{retry_block}\n\n"
            "Please analyze the documents and reason step by step."
        )

        with LoggerContext(
            logger, "reason_node",
            question=question,
            attempt=attempt_num,
            is_retry=is_retry,
            model=self.model,
        ):
            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": REASON_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=config.vllm.max_tokens,
                    temperature=config.vllm.temperature,
                )
                reasoning_text = response.choices[0].message.content or ""
            except Exception as e:
                logger.error("reason_llm_failed", error=str(e))
                reasoning_text = (
                    "REASONING:\nLLM endpoint unavailable.\n\n"
                    "CALCULATION:\nNONE\n\n"
                    f"PRELIMINARY_ANSWER:\nLLM error: {e}"
                )

        calc_expr = _parse_calculation(reasoning_text)

        state["reasoning"] = reasoning_text
        state["calculation_expression"] = calc_expr
        state["trace"].append({
            "node": "reason",
            "attempt": attempt_num,
            "is_retry": is_retry,
            "reasoning_preview": reasoning_text[:400],
            "calculation_expression": calc_expr,
        })
        state["node_traces"].append({"node": "reason", "attempt": attempt_num, "duration_ms": round((time.perf_counter() - _t0) * 1000, 2), "status": "ok"})

        return state

    # -----------------------------------------------------------------------
    # Node: calculator
    # -----------------------------------------------------------------------

    def _calculator_node(self, state: AgentState) -> AgentState:
        """Safely evaluate the arithmetic expression extracted from reasoning."""
        expr = state.get("calculation_expression")
        _t0 = time.perf_counter()

        with LoggerContext(logger, "calculator_node", expression=expr):
            if not expr:
                state["calculation_result"] = None
                state["trace"].append({"node": "calculator", "skipped": True, "reason": "no_expression"})
                state["node_traces"].append({"node": "calculator", "duration_ms": round((time.perf_counter() - _t0) * 1000, 2), "status": "skipped"})
                return state

            result_str = _evaluate_expression(expr)
            logger.info("calculation_result", expression=expr, result=result_str)

            state["calculation_result"] = result_str
            state["trace"].append({
                "node": "calculator",
                "expression": expr,
                "result": result_str,
            })

        state["node_traces"].append({"node": "calculator", "duration_ms": round((time.perf_counter() - _t0) * 1000, 2), "status": "ok"})
        return state

    # -----------------------------------------------------------------------
    # Node: verifier
    # -----------------------------------------------------------------------

    def _verifier_node(self, state: AgentState) -> AgentState:
        """
        Check that reasoning and calculation are consistent with retrieved docs.

        Also sets:
        - failure_feedback: formatted issues string when status is FAIL
        - retry_exhausted: True when FAIL and no retries remain
        """
        question = state["question"]
        reasoning = state["reasoning"]
        calc_expr = state.get("calculation_expression")
        calc_result = state.get("calculation_result")
        context_str = self._format_context(state["retrieved_docs"])
        _t0 = time.perf_counter()

        calc_block = ""
        if calc_expr:
            calc_block = f"\nCalculation performed: {calc_expr} = {calc_result}"

        user_prompt = (
            f"Question: {question}\n\n"
            f"Retrieved Financial Documents (source of truth):\n{context_str}\n\n"
            f"Reasoning produced:\n{reasoning}"
            f"{calc_block}\n\n"
            "Please verify whether the reasoning and numbers are consistent with the documents."
        )

        with LoggerContext(logger, "verifier_node", question=question, model=self.model):
            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": VERIFY_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=512,
                    temperature=0.0,
                )
                verify_text = response.choices[0].message.content or ""
            except Exception as e:
                logger.error("verifier_llm_failed", error=str(e))
                verify_text = (
                    "VERIFICATION_STATUS: UNCERTAIN\n"
                    "CONFIDENCE: LOW\n"
                    f"ISSUES:\n- Verifier LLM call failed: {e}"
                )

        status, confidence, issues = _parse_verification(verify_text)

        # Build failure_feedback for the next reason attempt
        if status == "FAIL":
            state["failure_feedback"] = "\n".join(f"- {i}" for i in issues) if issues else "- Unspecified verification failure"
        else:
            state["failure_feedback"] = None

        # Mark exhausted when this FAIL would exceed the retry budget
        state["retry_exhausted"] = (status == "FAIL" and state["retry_count"] >= MAX_RETRIES)

        logger.info(
            "verification_completed",
            status=status,
            confidence=confidence,
            num_issues=len(issues),
            retry_count=state["retry_count"],
            retry_exhausted=state["retry_exhausted"],
        )

        state["verification_status"] = status
        state["verification_confidence"] = confidence
        state["verification_issues"] = issues
        state["trace"].append({
            "node": "verifier",
            "attempt": state["retry_count"] + 1,
            "status": status,
            "confidence": confidence,
            "issues": issues,
            "retry_exhausted": state["retry_exhausted"],
            "raw_preview": verify_text[:300],
        })
        state["node_traces"].append({"node": "verifier", "attempt": state["retry_count"] + 1, "duration_ms": round((time.perf_counter() - _t0) * 1000, 2), "status": status.lower()})

        return state

    # -----------------------------------------------------------------------
    # Node: answer
    # -----------------------------------------------------------------------

    def _answer_node(self, state: AgentState) -> AgentState:
        """
        Produce the final answer with tone scaled to verification outcome:

          PASS + HIGH     → CONFIDENT  — direct answer
          PASS + MEDIUM   → CAVEAT     — answer with light caveat
          UNCERTAIN       → HUMAN_REVIEW — flagged for human review
          retries exhausted → MAX_RETRIES — low-confidence warning
        """
        question = state["question"]
        calc_result = state.get("calculation_result")
        v_status = state["verification_status"]
        v_confidence = state["verification_confidence"]
        retry_exhausted = state["retry_exhausted"]

        mode = _answer_mode(v_status, v_confidence, retry_exhausted)
        mode_instruction = _ANSWER_MODE_INSTRUCTIONS[mode]
        _t0 = time.perf_counter()

        calc_line = f"\nCalculation Result: {calc_result}" if calc_result else ""
        issues_line = ""
        if state["verification_issues"] and v_status != "PASS":
            issues_line = "\nVerification Issues:\n" + "\n".join(
                f"- {i}" for i in state["verification_issues"]
            )
        retry_line = f"\nRetry attempts used: {state['retry_count']} / {MAX_RETRIES}"

        user_prompt = (
            f"Question: {question}\n\n"
            f"Reasoning Trace:\n{state['reasoning']}"
            f"{calc_line}\n\n"
            f"Verification: {v_status} (Confidence: {v_confidence})"
            f"{issues_line}"
            f"{retry_line}\n\n"
            f"Instruction: {mode_instruction}"
        )

        with LoggerContext(
            logger, "answer_node",
            question=question,
            verification_status=v_status,
            answer_mode=mode,
        ):
            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": ANSWER_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=256,
                    temperature=0.0,
                )
                answer_text = response.choices[0].message.content or ""
            except Exception as e:
                logger.error("answer_llm_failed", error=str(e))
                answer_text = _fallback_answer(state["reasoning"], calc_result, e)

        state["final_answer"] = answer_text
        state["trace"].append({
            "node": "answer",
            "answer_mode": mode,
            "final_answer": answer_text,
        })
        state["node_traces"].append({"node": "answer", "answer_mode": mode, "duration_ms": round((time.perf_counter() - _t0) * 1000, 2), "status": "ok"})

        return state

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def load_index(self) -> bool:
        """Load the saved retriever index. Returns True on success."""
        with LoggerContext(logger, "load_retriever_index"):
            loaded = self.retriever.load_index()
        return loaded

    def run(self, question: str) -> Dict[str, Any]:
        """Run the full agent workflow and return the complete state."""
        initial_state: AgentState = {
            "question": question,
            "retrieved_docs": [],
            "reasoning": "",
            "calculation_expression": None,
            "calculation_result": None,
            "verification_status": "SKIPPED",
            "verification_issues": [],
            "verification_confidence": "N/A",
            "retry_count": 0,
            "failure_feedback": None,
            "retry_exhausted": False,
            "final_answer": "",
            "trace": [],
            "node_traces": [],
        }

        with LoggerContext(logger, "agent_run", question=question):
            result = self.graph.invoke(initial_state)

        return result


# ---------------------------------------------------------------------------
# Pure helpers (no state mutation)
# ---------------------------------------------------------------------------

def _parse_calculation(text: str) -> Optional[str]:
    """Extract the CALCULATION expression from LLM output."""
    match = re.search(
        r"CALCULATION:\s*\n(.+?)(?:\n\nPRELIMINARY|\n\nREASONING|\Z)",
        text,
        re.DOTALL,
    )
    if not match:
        return None
    expr = match.group(1).strip()
    return None if expr.upper() == "NONE" or not expr else expr


def _parse_verification(text: str) -> Tuple[str, str, List[str]]:
    """Extract (status, confidence, issues) from verifier LLM output."""
    status_match = re.search(r"VERIFICATION_STATUS:\s*(PASS|FAIL|UNCERTAIN)", text, re.IGNORECASE)
    status = status_match.group(1).upper() if status_match else "UNCERTAIN"

    conf_match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", text, re.IGNORECASE)
    confidence = conf_match.group(1).upper() if conf_match else "LOW"

    issues: List[str] = []
    issues_match = re.search(r"ISSUES:\s*\n((?:- .+\n?)+)", text, re.IGNORECASE)
    if issues_match:
        for line in issues_match.group(1).splitlines():
            line = line.strip().lstrip("- ").strip()
            if line and line.upper() != "NONE":
                issues.append(line)

    return status, confidence, issues


def _evaluate_expression(expr: str) -> str:
    """Evaluate an arithmetic expression safely using sympy or restricted eval."""
    try:
        if SYMPY_AVAILABLE:
            result = float(N(sympify(expr)))
        else:
            if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", expr):
                raise ValueError(f"Unsafe expression refused: {expr}")
            result = float(eval(expr))  # noqa: S307
        return str(round(result, 6))
    except Exception as e:
        return f"Error: {e}"


def _fallback_answer(reasoning: str, calc_result: Optional[str], error: Exception) -> str:
    """Build a fallback answer when the LLM answer call fails."""
    if calc_result and not calc_result.startswith("Error"):
        return (
            f"FINAL_ANSWER: {calc_result}\n"
            "EXPLANATION: Computed from retrieved financial data (LLM answer node unavailable)."
        )
    prelim_match = re.search(r"PRELIMINARY_ANSWER:\s*(.+)", reasoning, re.DOTALL)
    prelim = prelim_match.group(1).strip()[:200] if prelim_match else "Unknown"
    return (
        f"FINAL_ANSWER: {prelim}\n"
        f"EXPLANATION: Derived from documents; LLM answer node failed: {error}"
    )


# ---------------------------------------------------------------------------
# Main — demo run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FinQA Agent — LangGraph Workflow Demo")
    print("=" * 80)

    agent = FinQAAgent()

    print("\nLoading hybrid retriever index...")
    if not agent.load_index():
        print("No saved index found. Building index first (this will take a few minutes)...")
        agent.retriever.build_index()
        print("Index built successfully.")
    else:
        print("Hybrid FAISS + BM25 index loaded successfully.")

    TEST_QUERY = "what is the interest expense in 2009?"
    print(f"\nTest Query: {TEST_QUERY}\n")

    result = agent.run(TEST_QUERY)

    # ── Print full reasoning trace ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FULL REASONING TRACE")
    print("=" * 80)

    for step in result["trace"]:
        node = step.get("node", "?")
        attempt = step.get("attempt", "")
        attempt_label = f" (attempt {attempt})" if attempt else ""
        print(f"\n{'─' * 60}")
        print(f"NODE: {node.upper()}{attempt_label}")
        print(f"{'─' * 60}")

        if node == "retrieve":
            print(f"  Documents retrieved : {step['num_docs']}")
            print(f"  Top hybrid score    : {step.get('top_hybrid_score', 'N/A'):.4f}")
            print("  Retrieved questions :")
            for q in step.get("retrieved_questions", []):
                print(f"    • {q}")

        elif node == "reason":
            print(f"  Is retry            : {step.get('is_retry', False)}")
            print(f"  Calculation expr    : {step.get('calculation_expression') or 'NONE'}")
            print("\n  Reasoning preview:\n")
            for line in step.get("reasoning_preview", "").split("\n"):
                print(f"    {line}")
            print("    ...")

        elif node == "calculator":
            if step.get("skipped"):
                print(f"  Skipped             : {step.get('reason', '')}")
            else:
                print(f"  Expression          : {step.get('expression')}")
                print(f"  Result              : {step.get('result')}")

        elif node == "verifier":
            status = step.get("status", "?")
            icon = {"PASS": "✓", "FAIL": "✗", "UNCERTAIN": "~"}.get(status, "?")
            print(f"  Status              : {icon} {status}")
            print(f"  Confidence          : {step.get('confidence', 'N/A')}")
            print(f"  Retry exhausted     : {step.get('retry_exhausted', False)}")
            issues = step.get("issues", [])
            if issues:
                print("  Issues detected     :")
                for issue in issues:
                    print(f"    ✗ {issue}")
            else:
                print("  Issues detected     : none")

        elif node == "answer":
            print(f"  Answer mode         : {step.get('answer_mode', 'N/A')}")
            print("\n  Final Answer:\n")
            for line in step.get("final_answer", "").split("\n"):
                print(f"    {line}")

    # ── Node execution timeline ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("NODE EXECUTION TIMELINE")
    print("=" * 80)
    print(f"  {'Node':<12} {'Attempt':<9} {'Duration':>10}    Status")
    print(f"  {'─'*12} {'─'*9} {'─'*10}    {'─'*10}")
    total_ms = 0.0
    for nt in result["node_traces"]:
        attempt_col = str(nt["attempt"]) if "attempt" in nt else "-"
        dur = nt["duration_ms"]
        total_ms += dur
        print(f"  {nt['node']:<12} {attempt_col:<9} {dur:>9.1f}ms   {nt['status']}")
    print(f"  {'─'*12} {'─'*9} {'─'*10}")
    print(f"  {'TOTAL':<12} {'':9} {total_ms:>9.1f}ms")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL ANSWER SUMMARY")
    print("=" * 80)
    print(f"\nQuestion             : {result['question']}")
    print(f"Retries used         : {result['retry_count']} / {MAX_RETRIES}")
    print(f"Retry exhausted      : {result['retry_exhausted']}")
    print(f"Verification Status  : {result['verification_status']} "
          f"(Confidence: {result['verification_confidence']})")
    if result["verification_issues"]:
        print("Verification Issues  :")
        for issue in result["verification_issues"]:
            print(f"  ✗ {issue}")
    print(f"\n{result['final_answer']}")
    if result.get("calculation_result"):
        print(f"\nCalculation : {result['calculation_expression']} = {result['calculation_result']}")

    print("\n" + "=" * 80)
    print("AGENT RUN COMPLETE")
    print("=" * 80 + "\n")
