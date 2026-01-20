"""
Small simulation harness for timeout handling policy.
Run: python -m agent.timeout_simulation
"""

from agent.agent_loop3 import StepExecutionTracker, StepType


class TimeoutPolicySimulator:
    def __init__(self, max_timeout_retries: int = 1):
        self.max_timeout_retries = max_timeout_retries
        self.tracker = StepExecutionTracker(max_steps=5, max_retries=1)
        self.next_step_id = "0"
        self.replans = 0

    def handle_timeout(self, error_message: str) -> str:
        message = (error_message or "").lower()
        if "timed out" not in message and "timeout" not in message:
            return "no-timeout"

        timeout_count = self.tracker.record_timeout(self.next_step_id)
        if timeout_count <= self.max_timeout_retries:
            return f"retry ({timeout_count}/{self.max_timeout_retries})"

        self.next_step_id = StepType.ROOT
        self.replans += 1
        return "replan-to-root"


def run_simulation():
    simulator = TimeoutPolicySimulator(max_timeout_retries=1)
    errors = [
        "Execution timed out after 50 seconds",
        "Execution timed out after 50 seconds",
        "Some other error"
    ]

    print("Timeout policy simulation:")
    for idx, error in enumerate(errors, start=1):
        action = simulator.handle_timeout(error)
        print(f"  Event {idx}: {error} -> {action}")

    print(f"Final next_step_id: {simulator.next_step_id}")
    print(f"Total replans: {simulator.replans}")


if __name__ == "__main__":
    run_simulation()
