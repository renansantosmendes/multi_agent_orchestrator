from deepagents.middleware.subagents import SubAgent

from src.core.prompts.agent_prompts import DRIFT_AGENT_DESCRIPTION, DRIFT_AGENT_SYSTEM_PROMPT
from src.core.tools.drift import detect_data_drift

drift_agent: SubAgent = {
    "name": "drift_detector",
    "description": DRIFT_AGENT_DESCRIPTION,
    "system_prompt": DRIFT_AGENT_SYSTEM_PROMPT,
    "tools": [detect_data_drift],
}
