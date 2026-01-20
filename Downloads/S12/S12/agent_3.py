import asyncio
import yaml
from dotenv import load_dotenv

from mcp_servers.multiMCP import MultiMCP
from agent.agent_loop3 import AgentLoop
from utils.utils import log_step, log_error

FORM_URL = "https://forms.gle/6Nc6QaaJyDvePxLv7"
VALID_COURSES = {"ERA", "EAG", "EPAi"}
VALID_MARITAL = {"Yes", "No", "Maybe"}


def _prompt_non_empty(label: str) -> str:
    while True:
        value = input(f"{label}: ").strip()
        if value:
            return value
        log_error(f"{label} is required.")


def _prompt_choice(label: str, choices: set[str]) -> str:
    choices_display = "/".join(sorted(choices))
    while True:
        value = input(f"{label} ({choices_display}): ").strip()
        if value in choices:
            return value
        log_error(f"Please enter one of: {choices_display}")


def build_form_prompt(form_data: dict[str, str]) -> str:
    return (
        f"Open {FORM_URL} and fill the form with the following details:\n"
        f"- Course (what course is he/she in?): {form_data['course']}\n"
        f"- Email ID: {form_data['email']}\n"
        f"- Master's name: {form_data['master']}\n"
        f"- Which course is he/she taking?: {form_data['course_taking']}\n"
        f"- Date of birth: {form_data['dob']}\n"
        f"- Marital status: {form_data['marital_status']}\n\n"
        "Submit the form after all required fields are filled. "
        "After submission, verify success by locating a confirmation message "
        "or a URL change that indicates the form was submitted. "
        "Report the exact confirmation text or resulting URL."
    )


async def run_form_agent() -> None:
    log_step("Form Agent: collect details", symbol="📝")
    form_data = {
        "course": _prompt_non_empty("Course (what course is he/she in?)"),
        "email": _prompt_non_empty("Email ID"),
        "master": _prompt_non_empty("Master's name"),
        "course_taking": _prompt_choice("Which course is he/she taking?", VALID_COURSES),
        "dob": _prompt_non_empty("Date of birth"),
        "marital_status": _prompt_choice("Marital status", VALID_MARITAL),
    }

    log_step("Loading MCP Servers...", symbol="📥")
    with open("config/mcp_server_config.yaml", "r") as f:
        profile = yaml.safe_load(f)
        mcp_servers_list = profile.get("mcp_servers", [])
        configs = list(mcp_servers_list)

    multi_mcp = MultiMCP(server_configs=configs)
    await multi_mcp.initialize()

    try:
        loop = AgentLoop(
            perception_prompt="prompts/perception_prompt.txt",
            decision_prompt="prompts/decision_prompt.txt",
            summarizer_prompt="prompts/summarizer_prompt.txt",
            multi_mcp=multi_mcp,
            strategy="exploratory",
        )
        query = build_form_prompt(form_data)
        await loop.run(query)
        log_step("Form submission attempt completed.", symbol="✅")
    finally:
        await multi_mcp.shutdown()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(run_form_agent())
