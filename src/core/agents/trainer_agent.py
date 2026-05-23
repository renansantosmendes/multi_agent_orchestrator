from deepagents.middleware.subagents import SubAgent

from src.core.prompts.agent_prompts import TRAINER_AGENT_DESCRIPTION, TRAINER_AGENT_SYSTEM_PROMPT
from src.core.tools.training import train_model

trainer_agent: SubAgent = {
    "name": "trainer",
    "description": TRAINER_AGENT_DESCRIPTION,
    "system_prompt": TRAINER_AGENT_SYSTEM_PROMPT,
    "tools": [train_model],
}
