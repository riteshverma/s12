from dataclasses import dataclass, asdict
from typing import Any, Literal, Optional
import uuid
import time
import json
from datetime import datetime

# Used by all snapshots
def current_utc_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

@dataclass
class ToolCode:
    tool_name: str
    tool_arguments: dict[str, Any]

    def to_dict(self):
        return {
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments
        }


@dataclass
class PerceptionSnapshot:
    run_id: str
    snapshot_type: str
    entities: list[str]
    result_requirement: str
    original_goal_achieved: bool
    reasoning: str
    local_goal_achieved: bool
    local_reasoning: str
    last_tooluse_summary: str
    solution_summary: str
    confidence: str
    route: str
    timestamp: str
    return_to: str = ""


@dataclass
class DecisionSnapshot:
    run_id: str
    input: dict
    output: dict
    next_step_id: str
    plan_graph: dict
    code_variants: dict
    timestamp: str = current_utc_ts()
    return_to: str = ""


@dataclass
class ExecutionSnapshot:
    run_id: str
    step_id: str
    variant_used: str
    code: str                     # NEW: full code string
    status: str                   # "success" or "error"
    result: Optional[Any]        # whatever run_user_code returns
    error: Optional[str]
    execution_time: str          # original start_timestamp
    total_time: str              # precomputed in run_user_code
    return_to: str = ""



@dataclass
class SummarizerSnapshot:
    run_id: str
    input: dict
    summary_output: str
    success: bool
    error: Optional[str]
    timestamp: str = current_utc_ts()
    return_to: str = ""


@dataclass
class BrowserSnapshot:
    run_id: str
    instruction: str
    steps: list[dict]
    success: bool
    final_message: str
    error: Optional[str]
    timestamp: str = current_utc_ts()
    return_to: str = ""



@dataclass
class Step:
    index: int
    description: str
    type: Literal["CODE", "CONCLUDE", "NOOP"]
    code: Optional[ToolCode] = None
    conclusion: Optional[str] = None
    execution_result: Optional[str] = None
    error: Optional[str] = None
    perception: Optional[PerceptionSnapshot] = None
    status: Literal["pending", "completed", "failed", "skipped"] = "pending"
    attempts: int = 0
    was_replanned: bool = False
    parent_index: Optional[int] = None
    generated_vars: list[str] = None

    def to_dict(self):
        return {
            "index": self.index,
            "description": self.description,
            "type": self.type,
            "code": self.code.to_dict() if self.code else None,
            "conclusion": self.conclusion,
            "execution_result": self.execution_result,
            "error": self.error,
            "perception": self.perception.__dict__ if self.perception else None,
            "status": self.status,
            "attempts": self.attempts,
            "was_replanned": self.was_replanned,
            "parent_index": self.parent_index,
            "generated_vars": self.generated_vars if self.generated_vars else [],
        }


class AgentSession:
    def __init__(self, session_id: str, original_query: str):
        self.session_id = session_id
        self.original_query = original_query
        # self.perception: Optional[PerceptionSnapshot] = None
        self.perception_snapshots: list[PerceptionSnapshot] = []
        self.decision_snapshots: list[DecisionSnapshot] = []
        self.execution_snapshots: list[ExecutionSnapshot] = []
        self.summarizer_snapshots: list[SummarizerSnapshot] = []
        self.browser_snapshots: list[BrowserSnapshot] = []
        self.final_summary = None
        self.status: str = "in_progress"
        self.completed_at: Optional[str] = None

        self.plan_versions: list[dict[str, Any]] = []
        self.state = {
            "original_goal_achieved": False,
            "final_answer": None,
            "confidence": 0.0,
            "reasoning_note": "",
            "solution_summary": ""
            
        }

    # def add_perception(self, snapshot: PerceptionSnapshot):
    #     self.perception = snapshot
    def add_perception_snapshot(self, snapshot: PerceptionSnapshot):
        self.perception_snapshots.append(snapshot)

    def add_decision_snapshot(self, snapshot: DecisionSnapshot):
        self.decision_snapshots.append(snapshot)

    def add_execution_snapshot(self, snapshot: ExecutionSnapshot):
        self.execution_snapshots.append(snapshot)

    def add_summarizer_snapshot(self, snapshot: SummarizerSnapshot):
        self.summarizer_snapshots.append(snapshot)

    def add_browser_snapshot(self, snapshot: BrowserSnapshot):
        self.browser_snapshots.append(snapshot)


    def add_plan_version(self, plan_texts: list[str], steps: list[Step]):
        plan = {
            "plan_text": plan_texts,
            "steps": steps.copy()
        }
        self.plan_versions.append(plan)
        return steps[0] if steps else None  # ✅ fix: return first Step

    def get_next_step_index(self) -> int:
        return sum(len(v["steps"]) for v in self.plan_versions)


    def to_json(self) -> dict:
        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "perception_snapshots": [asdict(p) for p in self.perception_snapshots],
            "decision_snapshots": [asdict(d) for d in self.decision_snapshots],
            "execution_snapshots": [asdict(e) for e in self.execution_snapshots],
            "summarizer_snapshots": [asdict(s) for s in self.summarizer_snapshots],
            "browser_snapshots": [asdict(b) for b in self.browser_snapshots],
            "final_summary": self.final_summary,
            "status": self.status,
            "completed_at": self.completed_at
        }

    def get_snapshot_summary(self):
        return {
            "session_id": self.session_id,
            "query": self.original_query,
            "final_plan": self.plan_versions[-1]["plan_text"] if self.plan_versions else [],
            "final_steps": [
                    asdict(s)
                    for version in self.plan_versions
                    for s in version["steps"]
                    if s.status == "completed"
                ],
            "final_executed_plan": [
                s.description
                for version in self.plan_versions
                for s in version["steps"]
                if s.status == "completed"
            ],
            "final_answer": self.state["final_answer"],
            "reasoning_note": self.state["reasoning_note"],
            "confidence": self.state["confidence"]
        }

    def mark_complete(self, perception: PerceptionSnapshot, final_answer: Optional[str] = None, fallback_confidence: float = 0.95):
        self.state.update({
            "original_goal_achieved": perception.original_goal_achieved,
            "final_answer": final_answer or perception.solution_summary,
            "confidence": perception.confidence or fallback_confidence,
            "reasoning_note": perception.reasoning,
            "solution_summary": perception.solution_summary,
        })
        # set the dedicated final_summary used in log
        self.final_summary = final_answer or perception.solution_summary



    def simulate_live(self, delay: float = 1.2):
        print("\n=== LIVE AGENT SESSION TRACE ===")
        print(f"Session ID: {self.session_id}")
        print(f"Query: {self.original_query}")
        time.sleep(delay)

        if self.perception_snapshots:
            print("\n[Perception 0] Initial ERORLL:")
            print(f"  {asdict(self.perception_snapshots)}")
            time.sleep(delay)

        for i, version in enumerate(self.plan_versions):
            print(f"\n[Decision Plan Text: V{i+1}]:")
            for j, p in enumerate(version["plan_text"]):
                print(f"  Step {j}: {p}")
            time.sleep(delay)

            for step in version["steps"]:
                print(f"\n[Step {step.index}] {step.description}")
                time.sleep(delay / 1.5)

                print(f"  Type: {step.type}")
                if step.code:
                    print(f"  Tool → {step.code.tool_name} | Args → {step.code.tool_arguments}")
                if step.execution_result:
                    print(f"  Execution Result: {step.execution_result}")
                if step.conclusion:
                    print(f"  Conclusion: {step.conclusion}")
                if step.error:
                    print(f"  Error: {step.error}")
                if step.perception:
                    print("  Perception ERORLL:")
                    for k, v in asdict(step.perception).items():
                        print(f"    {k}: {v}")
                print(f"  Status: {step.status}")
                if step.was_replanned:
                    print(f"  (Replanned from Step {step.parent_index})")
                if step.attempts > 1:
                    print(f"  Attempts: {step.attempts}")
                time.sleep(delay)

        print("\n[Session Snapshot]:")
        print(json.dumps(self.get_snapshot_summary(), indent=2))
