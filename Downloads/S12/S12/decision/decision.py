import os
import json
from pathlib import Path
from utils.utils import log_step, log_error, log_json_block
from google.genai.errors import ServerError
import re
from utils.json_parser import parse_llm_json
from agent.agentSession import DecisionSnapshot
from mcp_servers.multiMCP import MultiMCP
import ast
import time
from utils.utils import log_step
import asyncio
from typing import Any, Literal, Optional
from agent.agentSession import AgentSession
import uuid
from datetime import datetime
from agent.model_manager import ModelManager

class Decision:
    def __init__(self, decision_prompt_path: str, multi_mcp: MultiMCP, api_key: str | None = None, model: str = "gemini-2.0-flash",  ):
        self.decision_prompt_path = decision_prompt_path
        self.multi_mcp = multi_mcp
        self.model = ModelManager()
        
    async def run(self, decision_input: dict, session: Optional[AgentSession] = None) -> dict:
        prompt_template = Path(self.decision_prompt_path).read_text(encoding="utf-8")
        function_list_text = self.multi_mcp.tool_description_wrapper()
        tool_descriptions = "\n".join(f"- `{desc.strip()}`" for desc in function_list_text)
        tool_descriptions = "\n\n### The ONLY Available Tools\n\n---\n\n" + tool_descriptions
        full_prompt = f"{prompt_template.strip()}\n{tool_descriptions}\n\n```json\n{json.dumps(decision_input, indent=2)}\n```"

        raw_text = ""
        try:
            log_step("[SENDING PROMPT TO DECISION...]", symbol="→")
            time.sleep(2)
            response = await self.model.generate_text(
                prompt=full_prompt
            )
            log_step("[RECEIVED OUTPUT FROM DECISION...]", symbol="←")

            output = parse_llm_json(response, required_keys=["plan_graph", "next_step_id", "code_variants"])
            output["code_variants"] = normalize_code_variants(
                output.get("next_step_id", ""),
                output.get("code_variants", {})
            )

            if session:
                session.add_decision_snapshot(
                    DecisionSnapshot(
                        run_id=decision_input.get("run_id", str(uuid.uuid4())),
                        input=decision_input,
                        plan_graph=output["plan_graph"],
                        next_step_id=output["next_step_id"],
                        code_variants=output["code_variants"],
                        output=output,
                        timestamp=decision_input.get("timestamp"),
                        return_to=""
                    )
                )

            return output

        except ServerError as e:
            log_error(f"🚫 DECISION LLM ServerError: {e}")
            if session:
                session.add_decision_snapshot(
                    DecisionSnapshot(
                        run_id=decision_input.get("run_id", str(uuid.uuid4())),
                        input=decision_input,
                        plan_graph={},
                        next_step_id="",
                        code_variants={},
                        output={"error": "ServerError", "message": str(e)},
                        timestamp=decision_input.get("timestamp"),
                        return_to=""
                    )
                )
            return {
                "plan_graph": {},
                "next_step_id": "",
                "code_variants": {},
                "error": "Decision ServerError: LLM unavailable"
            }

        except Exception as e:
            log_error(f"🛑 DECISION ERROR: {str(e)}")
            if session:
                session.add_decision_snapshot(
                    DecisionSnapshot(
                        run_id=decision_input.get("run_id", str(uuid.uuid4())),
                        input=decision_input,
                        plan_graph={},
                        next_step_id="",
                        code_variants={},
                        output={"error": str(e), "raw_text": raw_text},
                        timestamp=decision_input.get("timestamp"),
                        return_to=""
                    )
                )
            return {
                "plan_graph": {},
                "next_step_id": "",
                "code_variants": {},
                "error": "Decision failed due to malformed response"
            }


def normalize_code_variants(next_step_id: str, code_variants: dict) -> dict:
    if not next_step_id or not isinstance(code_variants, dict):
        return code_variants
    expected_prefix = f"CODE_{next_step_id}"
    if any(key.startswith(expected_prefix) for key in code_variants.keys()):
        return code_variants

    remapped = {}
    for suffix in ("A", "B", "C"):
        source_key = next((k for k in code_variants.keys() if k.endswith(suffix)), None)
        if source_key:
            remapped[f"{expected_prefix}{suffix}"] = code_variants[source_key]

    return remapped if len(remapped) == 3 else code_variants



def build_decision_input(ctx, query, p_out, strategy):
    return {
        "current_time": datetime.utcnow().isoformat(),
        "plan_mode": "initial",
        "planning_strategy": strategy,
        "original_query": query,
        "perception": p_out,
        "plan_graph": {},  # initially empty
        "completed_steps": [ctx.graph.nodes[n]["data"].__dict__ for n in ctx.graph.nodes if ctx.graph.nodes[n]["data"].status == "completed"],
        "failed_steps": [ctx.graph.nodes[n]["data"].__dict__ for n in ctx.failed_nodes],
        "globals_schema": {
            k: {
                "type": type(v).__name__,
                "preview": str(v)[:500] + ("…" if len(str(v)) > 500 else "")
            } for k, v in ctx.globals.items()
        }
    }