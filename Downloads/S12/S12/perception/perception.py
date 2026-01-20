import os
import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

from google.genai.errors import ServerError

from agent.agentSession import AgentSession, PerceptionSnapshot
from utils.utils import log_step, log_error, log_json_block
from utils.json_parser import parse_llm_json
from agent.model_manager import ModelManager



class Perception:
    def __init__(self, perception_prompt_path: str, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.perception_prompt_path = perception_prompt_path
        self.model = ModelManager()

    async def run(self, p_input: dict, session: Optional[AgentSession] = None) -> dict:
        prompt_template = Path(self.perception_prompt_path).read_text(encoding="utf-8")
        full_prompt = (
            f"{prompt_template.strip()}\n\n"
            "```json\n"
            f"{json.dumps(p_input, indent=2)}\n"
            "```"
        )

        try:
            log_step("[SENDING PROMPT TO PERCEPTION...]", symbol="→")
            # import pdb; pdb.set_trace()
            time.sleep(2)
            response = await self.model.generate_text(
                prompt=full_prompt
            )
        except ServerError as e:
            print(f"🚫 Perception LLM ServerError: {e}")
            return {
                "entities": [],
                "result_requirement": "Unavailable due to 503 error.",
                "original_goal_achieved": False,
                "reasoning": "Gemini model was not reachable.",
                "local_goal_achieved": False,
                "local_reasoning": "N/A",
                "last_tooluse_summary": "None",
                "solution_summary": "503 Unavailable. The service is currently unavailable.",
                "confidence": "0.0",
                "route": "decision"
            }
            
        log_step("[RECEIVED OUTPUT FROM PERCEPTION...]", symbol="←")

        try:
            output = parse_llm_json(response, required_keys=['entities', 'result_requirement', 'original_goal_achieved', 'reasoning', 'local_goal_achieved', 'local_reasoning', 'last_tooluse_summary', 'solution_summary', 'confidence', 'route'])
            if output.get("route") == "summarize" and "instruction_to_summarize" not in output:
                output["instruction_to_summarize"] = "Summarize the final results clearly. Format as plain text."
            if output.get("route") == "browser" and "instruction_to_browser" not in output:
                output["instruction_to_browser"] = output.get(
                    "result_requirement",
                    "Complete the browser task and confirm success."
                )
            # Success block
            if session:
                session.add_perception_snapshot(
                    PerceptionSnapshot(
                        run_id=p_input["run_id"],
                        snapshot_type=p_input["snapshot_type"],
                        entities=output.get("entities", []),
                        result_requirement=output.get("result_requirement", ""),
                        original_goal_achieved=output.get("original_goal_achieved", False),
                        reasoning=output.get("reasoning", ""),
                        local_goal_achieved=output.get("local_goal_achieved", False),
                        local_reasoning=output.get("local_reasoning", ""),
                        last_tooluse_summary=output.get("last_tooluse_summary", ""),
                        solution_summary=output.get("solution_summary", ""),
                        confidence=output.get("confidence", "0.0"),
                        route=output.get("route", "decision"),
                        timestamp=p_input.get("timestamp"),
                        return_to=""
                    )
                )
            return output

        except Exception as e:
            import pdb; pdb.set_trace()
            log_error("🛑 EXCEPTION IN PERCEPTION:", e)
            # Fallback output
            fallback_output = {
                "entities": [],
                "result_requirement": "N/A",
                "original_goal_achieved": False,
                "reasoning": "Perception failed to parse model output as JSON.",
                "local_goal_achieved": False,
                "local_reasoning": "Could not extract structured information.",
                "last_tooluse_summary": "None",
                "solution_summary": "Not ready yet",
                "confidence": "0.0",
                "route": "decision"
            }

            if session:
                session.add_perception_snapshot(
                    PerceptionSnapshot(
                        run_id=p_input.get("run_id", str(uuid.uuid4())),
                        snapshot_type=p_input.get("snapshot_type", "unknown"),
                        **fallback_output,
                        timestamp=p_input.get("timestamp"),
                        return_to=""
                    )
                )
            return fallback_output

def build_perception_input(query, memory, ctx, snapshot_type="user_query"):
    return {
        "current_time": datetime.utcnow().isoformat(),
        "run_id": f"{ctx.session_id}-P",
        "snapshot_type": snapshot_type,
        "original_query": query,
        "raw_input": query if snapshot_type == "user_query" else str(ctx.globals),
        "memory_excerpt": memory,
        "current_plan": getattr(ctx, "plan_graph", {}),
        "completed_steps": [ctx.graph.nodes[n]["data"].__dict__ for n in ctx.graph.nodes if ctx.graph.nodes[n]["data"].status == "completed"],
        "failed_steps": [ctx.graph.nodes[n]["data"].__dict__ for n in ctx.failed_nodes],
        "globals_schema": {
            k: (type(v).__name__, str(v)[:120]) for k, v in ctx.globals.items()
        },
        "timestamp": "...",
        "schema_version": 1
    }