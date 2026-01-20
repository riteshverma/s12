import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from decision import Decision, build_decision_input
from mcp_servers.multiMCP import MultiMCP
from utils.utils import log_step
import os
import asyncio

async def test_decision():
    decision_prompt_path = "prompts/decision_prompt.txt"
    multi_mcp = MultiMCP(server_configs=[])
    decision = Decision(decision_prompt_path, multi_mcp)
    dummy_ctx = type("DummyCtx", (), {
        "globals": {},
        "graph": type("G", (), {"nodes": {}, "edges": []})(),
        "failed_nodes": [],
    })()
    perception_result = {"result_requirement": "Calculate 4 + 4"}
    input_data = build_decision_input(dummy_ctx, "What is 4 + 4?", perception_result, "exploratory")
    log_step("Decision Input", input_data)
    result = decision.run(input_data)
    log_step("Decision Output", result)

if __name__ == "__main__":
    asyncio.run(test_decision())