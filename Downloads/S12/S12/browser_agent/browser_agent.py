import json
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google.genai.errors import ServerError

from agent.agentSession import AgentSession, BrowserSnapshot
from agent.model_manager import ModelManager
from memory.browser_agent_memory import BrowserAgentMemoryStore, BrowserAgentMemoryState
from utils.json_parser import parse_llm_json
from utils.utils import log_error, log_step


@dataclass
class BrowserActionStep:
    index: int
    action: dict
    result: str
    error: Optional[str] = None
    ranking: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "action": self.action,
            "result": self.result,
            "error": self.error,
            "ranking": self.ranking
        }


class BrowserAgent:
    def __init__(
        self,
        browser_prompt_path: str,
        multi_mcp,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
        max_steps: int = 25
    ):
        self.browser_prompt_path = browser_prompt_path
        self.multi_mcp = multi_mcp
        self.model = ModelManager()
        self.max_steps = max_steps
        self.memory_store = BrowserAgentMemoryStore()

    async def run(self, instruction: str, session: Optional[AgentSession] = None) -> dict:
        steps: list[BrowserActionStep] = []
        last_action = None
        last_result = ""
        run_id = str(uuid.uuid4())
        memory_key = session.session_id if session else run_id
        memory_state = self.memory_store.load_state(memory_key)

        try:
            ready, ready_msg = await self._ensure_browser_ready()
            if not ready:
                return self._finalize(
                    run_id,
                    instruction,
                    steps,
                    success=False,
                    final_message=ready_msg,
                    session=session
                )

            for step_idx in range(1, self.max_steps + 1):
                browser_state = await self._get_browser_state()
                page_hash = memory_state.record_page(browser_state)
                memory_state.record_redirect_if_needed(page_hash)

                prompt = self._build_prompt(
                    instruction=instruction,
                    step_idx=step_idx,
                    last_action=last_action,
                    last_result=last_result,
                    browser_state=browser_state,
                    history=steps,
                    memory_state=memory_state
                )

                log_step(f"[BROWSER AGENT] Step {step_idx}", symbol="🌐")
                time.sleep(1)
                response = await self.model.generate_text(prompt=prompt)
                decision = parse_llm_json(
                    response,
                    required_keys=["done", "success", "action", "final_message"]
                )

                if decision.get("done"):
                    return self._finalize(
                        run_id,
                        instruction,
                        steps,
                        success=bool(decision.get("success")),
                        final_message=decision.get("final_message", ""),
                        session=session
                    )

                action = decision.get("action") or {}
                tool_name = action.get("tool_name")
                arguments = dict(action.get("arguments", {}) or {})
                ranking_info = None
                rank_request = decision.get("rank_candidates")
                if not tool_name:
                    return self._finalize(
                        run_id,
                        instruction,
                        steps,
                        success=False,
                        final_message="BrowserAgent failed: missing tool_name in action.",
                        session=session,
                        memory_state=memory_state
                    )

                if rank_request and tool_name in {
                    "click_element_by_index",
                    "input_text",
                    "select_dropdown_option"
                }:
                    ranked_index, ranking_info = await self._rank_interactive_elements(
                        rank_request=rank_request,
                        instruction=instruction,
                        browser_state=browser_state
                    )
                    if ranked_index is not None:
                        arguments["index"] = ranked_index

                result_text, error = await self._call_tool(tool_name, arguments)
                memory_state.record_action(
                    {"tool_name": tool_name, "arguments": arguments},
                    result_text,
                    error,
                    step_idx
                )
                memory_state.record_form_interaction(
                    {"tool_name": tool_name, "arguments": arguments},
                    result_text,
                    step_idx
                )
                if tool_name in {"open_tab", "go_to_url"}:
                    memory_state.record_page(browser_state, url=arguments.get("url"))
                    memory_state.record_tab_event(
                        {"tool_name": tool_name, "url": arguments.get("url"), "step_index": step_idx}
                    )
                if tool_name in {"switch_tab", "close_tab"}:
                    memory_state.record_tab_event(
                        {"tool_name": tool_name, "tab_id": arguments.get("tab_id"), "step_index": step_idx}
                    )
                self.memory_store.save_state(memory_key, memory_state)
                steps.append(
                    BrowserActionStep(
                        index=step_idx,
                        action={"tool_name": tool_name, "arguments": arguments},
                        result=result_text,
                        error=error,
                        ranking=ranking_info
                    )
                )
                last_action = {"tool_name": tool_name, "arguments": arguments}
                last_result = result_text

            return self._finalize(
                run_id,
                instruction,
                steps,
                success=False,
                final_message="BrowserAgent stopped after max steps without completing the task.",
                session=session,
                memory_state=memory_state
            )

        except ServerError as e:
            log_error(f"🚫 BrowserAgent LLM ServerError: {e}")
            return self._finalize(
                run_id,
                instruction,
                steps,
                success=False,
                final_message="BrowserAgent failed due to model error (503).",
                session=session,
                error=str(e),
                memory_state=memory_state
            )
        except Exception as e:
            log_error(f"🛑 BrowserAgent Error: {e}")
            return self._finalize(
                run_id,
                instruction,
                steps,
                success=False,
                final_message="BrowserAgent failed due to an internal error.",
                session=session,
                error=str(e),
                memory_state=memory_state
            )

    async def run_form_submission(self, form_data: dict[str, str], session: Optional[AgentSession] = None) -> dict:
        from agent_3 import build_form_prompt

        instruction = build_form_prompt(form_data)
        return await self.run(instruction, session=session)

    async def _get_browser_state(self) -> str:
        try:
            result = await self.multi_mcp.call_tool("get_session_snapshot", {})
            text = self._extract_text_content(result)
            if self._looks_like_tool_error(text):
                return f"Browser state unavailable: {text}"
            return text
        except Exception as e:
            return f"Browser state unavailable: {e}"

    async def _ensure_browser_ready(self) -> tuple[bool, str]:
        result_text, error = await self._call_tool("get_session_snapshot", {})
        if not error and not self._looks_like_tool_error(result_text):
            return True, ""

        await self._call_tool("close_browser_session", {})
        await self._call_tool("open_tab", {"url": "about:blank"})
        result_text, error = await self._call_tool("get_session_snapshot", {})

        if error or self._looks_like_tool_error(result_text):
            return False, (
                "Browser session unavailable. Make sure the BrowserMCP server is running: "
                "uv run browserMCP/browser_mcp_sse.py"
            )
        return True, ""

    def _looks_like_tool_error(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(marker in lowered for marker in [
            "error executing tool",
            "error getting",
            "browser state unavailable",
            "browser session closed",
            "not found",
            "failed to",
            "navigation failed",
            "chrome-error://",
            "about:neterror",
            "about:blank",
        ])

    async def _call_tool(self, tool_name: str, arguments: dict) -> tuple[str, Optional[str]]:
        try:
            result = await self.multi_mcp.call_tool(tool_name, arguments)
            return self._extract_text_content(result), None
        except Exception as e:
            return f"Tool error: {e}", str(e)

    def _extract_text_content(self, result: Any) -> str:
        if hasattr(result, "content"):
            content = getattr(result, "content", [])
            texts = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    texts.append(text)
                elif isinstance(item, dict) and "text" in item:
                    texts.append(str(item["text"]))
            if texts:
                return "\n".join(texts)

        if isinstance(result, list):
            texts = []
            for item in result:
                if isinstance(item, dict) and "text" in item:
                    texts.append(str(item["text"]))
                else:
                    texts.append(str(item))
            return "\n".join(texts)

        return str(result)

    def _build_prompt(
        self,
        instruction: str,
        step_idx: int,
        last_action: Optional[dict],
        last_result: str,
        browser_state: str,
        history: list[BrowserActionStep],
        memory_state: BrowserAgentMemoryState
    ) -> str:
        prompt_template = Path(self.browser_prompt_path).read_text(encoding="utf-8")
        tool_descriptions = self._build_tool_descriptions()
        history_payload = [step.to_dict() for step in history][-5:]

        payload = {
            "instruction": instruction,
            "step_index": step_idx,
            "max_steps": self.max_steps,
            "last_action": last_action,
            "last_result": last_result,
            "browser_state": browser_state,
            "recent_steps": history_payload,
            "browser_memory": memory_state.to_dict()
        }

        return (
            f"{prompt_template.strip()}\n"
            f"{tool_descriptions}\n\n"
            f"```json\n{json.dumps(payload, indent=2)}\n```"
        )

    def _build_tool_descriptions(self) -> str:
        tools = self.multi_mcp.get_tools_from_servers(["webbrowsing"])
        descriptions = []
        for tool in tools:
            schema = tool.inputSchema
            if "input" in schema.get("properties", {}):
                inner_key = next(iter(schema.get("$defs", {})), None)
                props = schema["$defs"][inner_key]["properties"]
            else:
                props = schema.get("properties", {})
            arg_types = []
            for _, spec in props.items():
                arg_types.append(spec.get("type", "any"))
            signature_str = ", ".join(arg_types)
            descriptions.append(f"- `{tool.name}({signature_str})`  # {tool.description}")
        return "\n\n### The ONLY Available Tools\n\n---\n\n" + "\n".join(descriptions)

    async def _rank_interactive_elements(
        self,
        rank_request: dict,
        instruction: str,
        browser_state: str
    ) -> tuple[Optional[int], Optional[dict]]:
        query = (rank_request or {}).get("query") or instruction
        categories = (rank_request or {}).get("categories")
        candidate_ids = set((rank_request or {}).get("candidate_ids", []))
        strict_mode = (rank_request or {}).get("strict_mode", True)

        elements_text, error = await self._call_tool(
            "get_interactive_elements",
            {"structured_output": True, "strict_mode": strict_mode}
        )
        if error:
            return None, None

        try:
            elements_payload = json.loads(elements_text)
        except Exception:
            return None, None

        candidates = []
        for group_name in ("nav", "forms", "buttons"):
            if categories and group_name not in categories:
                continue
            for item in elements_payload.get(group_name, []):
                element_id = item.get("id")
                if candidate_ids and element_id not in candidate_ids:
                    continue
                candidates.append(
                    {
                        "id": element_id,
                        "desc": item.get("desc", ""),
                        "action": item.get("action", ""),
                        "category": group_name
                    }
                )

        if not candidates:
            return None, None

        ranking_prompt = (
            "Select the best element ID for the user intent.\n"
            f"Instruction: {instruction}\n"
            f"Query: {query}\n"
            f"Browser state excerpt: {browser_state[:500]}\n"
            "Candidates:\n"
            + "\n".join(
                f"- id: {c['id']}, category: {c['category']}, desc: {c['desc']}, action: {c['action']}"
                for c in candidates[:40]
            )
            + "\nReturn STRICT JSON: {\"index\": <id>, \"reason\": \"...\", \"alternates\": [..]}"
        )

        response = await self.model.generate_text(prompt=ranking_prompt)
        try:
            ranking = parse_llm_json(response, required_keys=["index"])
        except Exception:
            return None, None

        selected = ranking.get("index")
        candidate_ids_set = {c["id"] for c in candidates}
        if selected not in candidate_ids_set:
            selected = candidates[0]["id"]

        ranking_info = {
            "query": query,
            "selected_index": selected,
            "reason": ranking.get("reason", ""),
            "alternates": ranking.get("alternates", []),
            "candidate_count": len(candidates)
        }
        return selected, ranking_info

    def _finalize(
        self,
        run_id: str,
        instruction: str,
        steps: list[BrowserActionStep],
        success: bool,
        final_message: str,
        session: Optional[AgentSession],
        error: Optional[str] = None,
        memory_state: Optional[BrowserAgentMemoryState] = None
    ) -> dict:
        output = {
            "run_id": run_id,
            "instruction": instruction,
            "success": success,
            "final_message": final_message,
            "steps": [step.to_dict() for step in steps],
            "error": error,
            "memory_state": memory_state.to_dict() if memory_state else None,
            "timestamp": datetime.utcnow().isoformat()
        }

        if session:
            session.add_browser_snapshot(
                BrowserSnapshot(
                    run_id=run_id,
                    instruction=instruction,
                    steps=output["steps"],
                    success=success,
                    final_message=final_message,
                    error=error,
                    memory_state=output["memory_state"]
                )
            )

        return output

