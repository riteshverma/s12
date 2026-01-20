
from utils.utils import log_step, log_error, render_graph
import networkx as nx
from typing import Any, Dict, Optional
from dataclasses import dataclass
import json
from collections import defaultdict


@dataclass
class StepNode:
    index: str  # Now string to support labels like "0A", "0B"
    description: str
    type: str  # CODE, CONCLUDE, NOP
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    conclusion: Optional[str] = None
    error: Optional[str] = None
    perception: Optional[Dict[str, Any]] = None
    from_step: Optional[str] = None  # for debugging lineage


class ContextManager:
    def __init__(self, session_id: str, original_query: str):
        self.session_id = session_id
        self.original_query = original_query
        self.globals: Dict[str, Any] = {}
        self.session_memory: list[dict] = []  # Full memory, not compressed
        self.failed_nodes: list[str] = []     # Node labels of failed steps
        self.graph = nx.DiGraph()
        self.latest_node_id: Optional[str] = None
        self.executed_variants: Dict[str, Set[str]] = defaultdict(set)


        root_node = StepNode(index="ROOT", description=original_query, type="ROOT", status="completed")
        self.graph.add_node("ROOT", data=root_node)

    def add_step(self, step_id: str, description: str, step_type: str, from_node: Optional[str] = None, edge_type: str = "normal") -> str:
        step_node = StepNode(index=step_id, description=description, type=step_type, from_step=from_node)
        self.graph.add_node(step_id, data=step_node)
        if from_node:
            self.graph.add_edge(from_node, step_id, type=edge_type)
        self.latest_node_id = step_id
        # self._print_graph(depth=1)
        return step_id

    def is_step_completed(self, step_id: str) -> bool:
        node = self.graph.nodes.get(step_id, {}).get("data")
        return node is not None and node.status == "completed"


    def update_step_result(self, step_id: str, result: dict):
        node: StepNode = self.graph.nodes[step_id]["data"]
        node.result = result
        node.status = "completed"
        self._update_globals(result)
        # self._print_graph(depth=2)

    def mark_step_completed(self, step_id: str):
        if step_id in self.graph:
            node: StepNode = self.graph.nodes[step_id]["data"]
            node.status = "completed"


    def mark_step_failed(self, step_id: str, error_msg: str):
        node: StepNode = self.graph.nodes[step_id]["data"]
        node.status = "failed"
        node.error = error_msg
        self.failed_nodes.append(step_id)
        self.session_memory.append({
            "query": node.description,
            "result_requirement": "Tool failed",
            "solution_summary": str(error_msg)[:300]
        })
        # self._print_graph(depth=2)

    def attach_perception(self, step_id: str, perception: dict):
        if step_id not in self.graph.nodes:
            fallback_node = StepNode(index=step_id, description="Perception-only node", type="PERCEPTION")
            self.graph.add_node(step_id, data=fallback_node)
        node: StepNode = self.graph.nodes[step_id]["data"]
        node.perception = perception
        if not perception.get("local_goal_achieved", True):
            self.failed_nodes.append(step_id)
        # self._print_graph(depth=2)

    def conclude(self, step_id: str, conclusion: str):
        node: StepNode = self.graph.nodes[step_id]["data"]
        node.status = "completed"
        node.conclusion = conclusion
        # self._print_graph(depth=2)

    def get_latest_node(self) -> Optional[str]:
        return self.latest_node_id

    def _update_globals(self, new_vars: Dict[str, Any]):
        for k, v in new_vars.items():
            if k in self.globals:
                versioned_key = f"{k}__{self.latest_node_id}"
                self.globals[versioned_key] = v
            else:
                self.globals[k] = v

    def _print_graph(self, depth: int = 1, only_if: bool = True):
        if only_if:
            render_graph(self.graph, depth=depth)

    def get_context_snapshot(self):
        def serialize_node_data(data):
            if hasattr(data, '__dict__'):
                return data.__dict__
            return data
        graph_data = nx.readwrite.json_graph.node_link_data(self.graph, edges="links")

        for node in graph_data["nodes"]:
            if "data" in node:
                node["data"] = serialize_node_data(node["data"])

        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "globals": self.globals,
            "memory": self.session_memory,
            "graph": graph_data,
        }

    def get_latest_perception(self) -> Optional[Dict[str, Any]]:
        """Return the most recent perception snapshot in the graph."""
        for node_id in reversed(list(self.graph.nodes)):
            node = self.graph.nodes.get(node_id, {}).get("data")
            if node and getattr(node, "perception", None):
                return node.perception
        return None

    def rename_subtree_from(self, from_step_id: str, suffix: str):
        if from_step_id not in self.graph:
            return
        to_rename = [from_step_id] + list(nx.descendants(self.graph, from_step_id))
        mapping = {}
        for old_id in to_rename:
            new_id = f"{old_id}{suffix}"
            node = self.graph.nodes[old_id]["data"]
            node.index = new_id
            self.graph.add_node(new_id, data=node)
            mapping[old_id] = new_id

        # Reconnect edges
        for old_src, old_tgt, attr in list(self.graph.edges(data=True)):
            new_src = mapping.get(old_src, old_src)
            new_tgt = mapping.get(old_tgt, old_tgt)
            if new_src != old_src or new_tgt != old_tgt:
                self.graph.add_edge(new_src, new_tgt, **attr)

        # Remove old nodes
        for old_id in to_rename:
            self.graph.remove_node(old_id)

        # Update failed_nodes list
        self.failed_nodes = [mapping.get(x, x) for x in self.failed_nodes]

    def attach_summary(self, summary: dict):
        """Attach summarizer output to session memory."""
        self.session_memory.append({
            "original_query": self.original_query,
            "result_requirement": "Final summary",
            "summarizer_summary": summary.get("summarizer_summary", summary if isinstance(summary, str) else ""),
            "confidence": summary.get("confidence", 0.95),
            "original_goal_achieved": True,
            "route": "summarize"
        })

