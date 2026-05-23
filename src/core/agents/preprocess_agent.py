from deepagents.middleware.subagents import SubAgent

from src.core.prompts.agent_prompts import PREPROCESS_AGENT_DESCRIPTION, PREPROCESS_AGENT_SYSTEM_PROMPT
from src.core.tools.preprocessing import preprocess_data

preprocess_agent: SubAgent = {
    "name": "preprocessor",
    "description": PREPROCESS_AGENT_DESCRIPTION,
    "system_prompt": PREPROCESS_AGENT_SYSTEM_PROMPT,
    "tools": [preprocess_data],
}
