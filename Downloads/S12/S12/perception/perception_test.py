
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from perception import Perception, build_perception_input
from utils.utils import log_step
import os

def test_perception():
    perception_prompt_path = "prompts/perception_prompt.txt"
    perception = Perception(perception_prompt_path)
    dummy_ctx = type("DummyCtx", (), {
        "session_id": "test123",
        "globals": {},
        "graph": type("G", (), {"nodes": {}, "edges": []})(),
        "failed_nodes": []
    })()
    memory = []
    input_data = build_perception_input("What is 4 + 4?", memory, dummy_ctx)
    log_step("Perception Input", input_data)
    result = perception.run(input_data)
    log_step("Perception Output", result)

if __name__ == "__main__":
    test_perception()