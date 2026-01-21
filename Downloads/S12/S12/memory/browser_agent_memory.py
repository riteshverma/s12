from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


STATE_DIR = Path("memory/browser_agent_state")
MAX_PAGE_HISTORY = 15
MAX_ACTION_HISTORY = 25
MAX_TAB_HISTORY = 20
MAX_REDIRECT_HISTORY = 15
MAX_FORM_FIELDS = 15
MAX_FAILURES = 20


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _compact(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _hash_text(text: str) -> str:
    normalized = " ".join((text or "").split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest[:16]


def _safe_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _action_signature(action: dict) -> str:
    tool_name = action.get("tool_name", "unknown")
    args = action.get("arguments", {})
    return f"{tool_name}:{_safe_json(args)}"


def _sanitize_action(action: dict) -> dict:
    tool_name = action.get("tool_name")
    args = action.get("arguments", {}) or {}
    sanitized_args = {}
    for key, value in args.items():
        if isinstance(value, str):
            sanitized_args[key] = _compact(value, 80)
            if len(value) > 80:
                sanitized_args[f"{key}_len"] = len(value)
        else:
            sanitized_args[key] = value
    return {"tool_name": tool_name, "arguments": sanitized_args}


def _default_form_flow() -> dict:
    return {
        "fields": [],
        "last_action": None,
        "last_submit_step": None,
        "submit_attempts": 0,
        "submitted": False,
        "submit_confirmed": False
    }


@dataclass
class BrowserAgentMemoryState:
    session_id: str
    page_history: list[dict] = field(default_factory=list)
    page_counts: dict[str, int] = field(default_factory=dict)
    redirects: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    failed_actions: dict[str, dict] = field(default_factory=dict)
    form_flow: dict = field(default_factory=_default_form_flow)
    tab_events: list[dict] = field(default_factory=list)
    last_page_hash: Optional[str] = None
    last_page_url: Optional[str] = None
    previous_page_hash: Optional[str] = None
    previous_page_url: Optional[str] = None

    def record_page(self, browser_state: str, url: Optional[str] = None) -> str:
        page_hash = _hash_text(browser_state)
        timestamp = _utc_ts()
        snippet = _compact(browser_state, 180)

        self.previous_page_hash = self.last_page_hash
        self.previous_page_url = self.last_page_url
        if url:
            self.last_page_url = url
        self.last_page_hash = page_hash

        self.page_counts[page_hash] = self.page_counts.get(page_hash, 0) + 1

        entry = {
            "hash": page_hash,
            "url": self.last_page_url,
            "snippet": snippet,
            "first_seen": timestamp,
            "last_seen": timestamp
        }

        if self.page_history and self.page_history[-1]["hash"] == page_hash:
            self.page_history[-1]["last_seen"] = timestamp
            if url:
                self.page_history[-1]["url"] = url
            if snippet:
                self.page_history[-1]["snippet"] = snippet
        else:
            self.page_history.append(entry)

        self._trim()
        return page_hash

    def record_redirect_if_needed(self, page_hash: str) -> None:
        if not self.previous_page_hash or self.previous_page_hash == page_hash:
            return

        if self.previous_page_url == self.last_page_url and self.previous_page_url:
            return

        self.redirects.append({
            "from_hash": self.previous_page_hash,
            "to_hash": page_hash,
            "from_url": self.previous_page_url,
            "to_url": self.last_page_url,
            "timestamp": _utc_ts()
        })
        self._trim()

    def record_action(self, action: dict, result: str, error: Optional[str], step_index: int) -> None:
        signature = _action_signature(action)
        entry = {
            "step_index": step_index,
            "action": _sanitize_action(action),
            "result": _compact(result, 220),
            "error": _compact(error or "", 200) if error else None
        }
        self.actions.append(entry)

        if error or self._looks_like_error(result):
            failure = self.failed_actions.get(signature, {
                "action": _sanitize_action(action),
                "count": 0,
                "last_error": "",
                "last_step": None
            })
            failure["count"] += 1
            failure["last_error"] = _compact(error or result or "", 200)
            failure["last_step"] = step_index
            self.failed_actions[signature] = failure

        self._trim()

    def record_form_interaction(self, action: dict, result: str, step_index: int) -> None:
        tool_name = action.get("tool_name")
        args = action.get("arguments", {}) or {}
        value = args.get("text") or args.get("value") or args.get("option")
        if tool_name in {"input_text", "select_dropdown_option"}:
            field_entry = {
                "step_index": step_index,
                "tool": tool_name,
                "index": args.get("index"),
                "value_preview": _compact(str(value), 32) if value is not None else "",
                "value_len": len(str(value)) if value is not None else 0
            }
            self.form_flow["fields"].append(field_entry)

        if tool_name in {"click_element_by_index", "send_keys"}:
            if self._looks_like_submit(result, args):
                self.form_flow["submit_attempts"] += 1
                self.form_flow["last_submit_step"] = step_index
                self.form_flow["submitted"] = True
            if self._looks_like_submit_confirmation(result):
                self.form_flow["submit_confirmed"] = True

        self.form_flow["last_action"] = {
            "step_index": step_index,
            "tool": tool_name,
            "index": args.get("index")
        }
        self._trim()

    def record_tab_event(self, event: dict) -> None:
        payload = dict(event)
        payload.setdefault("timestamp", _utc_ts())
        self.tab_events.append(payload)
        self._trim()

    def to_dict(self) -> dict:
        failed = sorted(
            self.failed_actions.values(),
            key=lambda item: item.get("count", 0),
            reverse=True
        )
        repeat_pages = [
            {"hash": page_hash, "count": count}
            for page_hash, count in self.page_counts.items()
            if count > 1
        ]
        repeat_pages.sort(key=lambda item: item["count"], reverse=True)

        return {
            "last_page": {
                "hash": self.last_page_hash,
                "url": self.last_page_url,
                "visit_count": self.page_counts.get(self.last_page_hash or "", 0)
            },
            "recent_pages": self.page_history[-5:],
            "repeat_pages": repeat_pages[:5],
            "recent_actions": self.actions[-6:],
            "failed_actions": failed[:5],
            "form_flow": {
                "fields": self.form_flow.get("fields", [])[-5:],
                "last_action": self.form_flow.get("last_action"),
                "submit_attempts": self.form_flow.get("submit_attempts", 0),
                "submitted": self.form_flow.get("submitted", False),
                "submit_confirmed": self.form_flow.get("submit_confirmed", False),
                "last_submit_step": self.form_flow.get("last_submit_step")
            },
            "tab_events": self.tab_events[-5:],
            "redirects": self.redirects[-5:]
        }

    def _looks_like_error(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(marker in lowered for marker in [
            "error",
            "failed",
            "exception",
            "timed out",
            "timeout",
            "not found",
            "denied"
        ])

    def _looks_like_submit(self, result: str, args: dict) -> bool:
        keys = str(args.get("keys") or "").lower()
        if "enter" in keys or "return" in keys:
            return True
        lowered = (result or "").lower()
        return any(marker in lowered for marker in [
            "submit",
            "submitted",
            "sending",
            "form"
        ])

    def _looks_like_submit_confirmation(self, result: str) -> bool:
        lowered = (result or "").lower()
        return any(marker in lowered for marker in [
            "thank you",
            "success",
            "submitted",
            "confirmation",
            "we received"
        ])

    def _trim(self) -> None:
        self.page_history = self.page_history[-MAX_PAGE_HISTORY:]
        self.actions = self.actions[-MAX_ACTION_HISTORY:]
        self.tab_events = self.tab_events[-MAX_TAB_HISTORY:]
        self.redirects = self.redirects[-MAX_REDIRECT_HISTORY:]
        self.form_flow["fields"] = self.form_flow.get("fields", [])[-MAX_FORM_FIELDS:]
        if len(self.failed_actions) > MAX_FAILURES:
            ordered = sorted(
                self.failed_actions.items(),
                key=lambda item: item[1].get("count", 0),
                reverse=True
            )
            self.failed_actions = dict(ordered[:MAX_FAILURES])


class BrowserAgentMemoryStore:
    def __init__(self, base_dir: Path | str = STATE_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _state_path(self, key: str) -> Path:
        safe_key = key.replace("/", "_")
        return self.base_dir / f"{safe_key}.json"

    def load_state(self, key: str) -> BrowserAgentMemoryState:
        path = self._state_path(key)
        if not path.exists():
            return BrowserAgentMemoryState(session_id=key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return BrowserAgentMemoryState(session_id=key)

        state = BrowserAgentMemoryState(session_id=data.get("session_id", key))
        state.page_history = data.get("page_history", [])
        state.page_counts = data.get("page_counts", {})
        state.redirects = data.get("redirects", [])
        state.actions = data.get("actions", [])
        state.failed_actions = data.get("failed_actions", {})
        state.form_flow = data.get("form_flow", _default_form_flow())
        state.tab_events = data.get("tab_events", [])
        state.last_page_hash = data.get("last_page_hash")
        state.last_page_url = data.get("last_page_url")
        state.previous_page_hash = data.get("previous_page_hash")
        state.previous_page_url = data.get("previous_page_url")
        state._trim()
        return state

    def save_state(self, key: str, state: BrowserAgentMemoryState) -> None:
        path = self._state_path(key)
        state._trim()
        payload = {
            "session_id": state.session_id,
            "page_history": state.page_history,
            "page_counts": state.page_counts,
            "redirects": state.redirects,
            "actions": state.actions,
            "failed_actions": state.failed_actions,
            "form_flow": state.form_flow,
            "tab_events": state.tab_events,
            "last_page_hash": state.last_page_hash,
            "last_page_url": state.last_page_url,
            "previous_page_hash": state.previous_page_hash,
            "previous_page_url": state.previous_page_url
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
