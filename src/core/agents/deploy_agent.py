from deepagents.middleware.subagents import SubAgent

from src.core.prompts.agent_prompts import DEPLOY_AGENT_DESCRIPTION, DEPLOY_AGENT_SYSTEM_PROMPT
from src.core.tools.deployment import deploy_model

deploy_agent: SubAgent = {
    "name": "deployer",
    "description": DEPLOY_AGENT_DESCRIPTION,
    "system_prompt": DEPLOY_AGENT_SYSTEM_PROMPT,
    "tools": [deploy_model],
}
