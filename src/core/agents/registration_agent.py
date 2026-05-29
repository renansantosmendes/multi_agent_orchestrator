from deepagents.middleware.subagents import SubAgent

from src.core.prompts.agent_prompts import (
    REGISTRATION_AGENT_DESCRIPTION,
    REGISTRATION_AGENT_SYSTEM_PROMPT,
)
from src.core.tools.registration import register_model

registration_agent: SubAgent = {
    "name": "registrar",
    "description": REGISTRATION_AGENT_DESCRIPTION,
    "system_prompt": REGISTRATION_AGENT_SYSTEM_PROMPT,
    "tools": [register_model],
}
