import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from summarizer import Summarizer
from utils.utils import log_step

def test_summarizer():
    summarizer_prompt_path = "prompts/summarizer_prompt.txt"
    summarizer = Summarizer(summarizer_prompt_path)
    input_data = {
        "original_query": "What is 4 + 4?",
        "globals_schema": {"result": 8},
        "plan_graph": {"nodes": [], "edges": []},
        "perception": {
            "entities": ["math"],
            "result_requirement": "Sum of 4 and 4",
            "original_goal_achieved": True,
            "local_goal_achieved": True,
            "confidence": "0.99",
            "reasoning": "Basic arithmetic.",
            "local_reasoning": "Simple addition",
            "last_tooluse_summary": "None",
            "solution_summary": "8",
            "route": "summarize",
            "instruction_to_summarize": "Summarize the final tool results for the user in plain language."
        }
    }
    log_step("Summarizer Input", input_data)
    result = summarizer.run(input_data)
    log_step("Summarizer Output", result)

if __name__ == "__main__":
    test_summarizer()