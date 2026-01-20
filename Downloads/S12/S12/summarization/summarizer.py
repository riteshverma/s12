
import os
import json
from pathlib import Path
from utils.utils import log_step, log_error, log_json_block
from google.genai.errors import ServerError
import time
from utils.utils import log_step, log_error, save_final_plan
from agent.agentSession import SummarizerSnapshot
from typing import Any, Literal, Optional
from agent.agentSession import AgentSession
import uuid
from datetime import datetime
from agent.model_manager import ModelManager


class Summarizer:
    def __init__(self, summarizer_prompt_path: str, api_key: str | None = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or explicitly provided.")
        self.model = ModelManager()
        self.summarizer_prompt_path = summarizer_prompt_path

    async def run(self, s_input: dict, session: Optional[AgentSession] = None) -> str:
        try:
            prompt_template = Path(self.summarizer_prompt_path).read_text(encoding="utf-8")
            full_prompt = (
                f"Current Time: {datetime.utcnow().isoformat()}\n\n"
                f"{prompt_template.strip()}\n\n"
                f"{json.dumps(s_input, indent=2)}"
            )

            log_step("[SENDING PROMPT TO SUMMARIZER...]")
            time.sleep(2)
            response = await self.model.generate_text(
                prompt=full_prompt
            )
            session.add_summarizer_snapshot(
                SummarizerSnapshot(
                    run_id=str(uuid.uuid4()),
                    input=s_input,
                    summary_output=response,
                    success="Summary unavailable" not in response,
                    error=None if "Summary unavailable" not in response else response
                )
            )
            return response
        except ServerError as e:
            print(f"🚫 Summarizer LLM ServerError: {e}")
            if session:
                session.add_summarizer_snapshot(
                    SummarizerSnapshot(
                        run_id=str(uuid.uuid4()),
                        input=s_input,
                        summary_output="",
                        success=False,
                        error=str(e)
                    )
                )

            return "Summary unavailable due to model error (503)."
        except Exception as e:
            print(f"❌ Unexpected Summarizer Exception: {e}")
            if session:
                session.add_summarizer_snapshot(
                    SummarizerSnapshot(
                        run_id=str(uuid.uuid4()),
                        input=s_input,
                        summary_output="",
                        success=False,
                        error=str(e)
                    )
                )

            return "Summary generation failed due to internal error."

    async def summarize(self, query, ctx, latest_perception, session: AgentSession) -> str:
        # ✅ Mark all remaining pending steps as "Skipped"
        for node_id in ctx.graph.nodes:
            node = ctx.graph.nodes[node_id]["data"]
            if node.status == "pending":
                node.status = "Skipped"

        s_input = {
            "original_query": query,
            "globals_schema": ctx.globals,
            "plan_graph": ctx.get_context_snapshot()["graph"],
            "perception": latest_perception
        }

        summary = await self.run(s_input, session=session)
        ctx.attach_summary({"summarizer_summary": summary})

        session.status = "success"
        session.completed_at = datetime.utcnow().isoformat()

        print("\n🔚 Final Summary:\n", summary)
        session.mark_complete(session.perception_snapshots[-1], final_answer=summary)
        save_final_plan(ctx.session_id, {
            "context": ctx.get_context_snapshot(),
            "session": session.to_json(),
            "status": "success",
            "final_step_id": ctx.get_latest_node(),
            "reason": "Summarized successfully",
            "timestamp": datetime.utcnow().isoformat(),
            "original_query": ctx.original_query,
            "final_summary": session.final_summary,
        })

        return summary