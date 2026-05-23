from deepagents.middleware.subagents import SubAgent

from src.core.prompts.agent_prompts import ANALYZER_AGENT_DESCRIPTION, ANALYZER_AGENT_SYSTEM_PROMPT
from src.core.tools.analysis import analyze_results

analyzer_agent: SubAgent = {
    "name": "result_analyzer",
    "description": ANALYZER_AGENT_DESCRIPTION,
    "system_prompt": ANALYZER_AGENT_SYSTEM_PROMPT,
    "tools": [analyze_results],
}
