from __future__ import annotations

import pytest

from src.core.prompts.agent_prompts import (
    ANALYZER_AGENT_DESCRIPTION,
    ANALYZER_AGENT_SYSTEM_PROMPT,
    DEPLOY_AGENT_DESCRIPTION,
    DEPLOY_AGENT_SYSTEM_PROMPT,
    DRIFT_AGENT_DESCRIPTION,
    DRIFT_AGENT_SYSTEM_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    PREPROCESS_AGENT_DESCRIPTION,
    PREPROCESS_AGENT_SYSTEM_PROMPT,
    TRAINER_AGENT_DESCRIPTION,
    TRAINER_AGENT_SYSTEM_PROMPT,
)

ALL_PROMPTS = [
    ("DRIFT_AGENT_DESCRIPTION", DRIFT_AGENT_DESCRIPTION),
    ("DRIFT_AGENT_SYSTEM_PROMPT", DRIFT_AGENT_SYSTEM_PROMPT),
    ("PREPROCESS_AGENT_DESCRIPTION", PREPROCESS_AGENT_DESCRIPTION),
    ("PREPROCESS_AGENT_SYSTEM_PROMPT", PREPROCESS_AGENT_SYSTEM_PROMPT),
    ("TRAINER_AGENT_DESCRIPTION", TRAINER_AGENT_DESCRIPTION),
    ("TRAINER_AGENT_SYSTEM_PROMPT", TRAINER_AGENT_SYSTEM_PROMPT),
    ("ANALYZER_AGENT_DESCRIPTION", ANALYZER_AGENT_DESCRIPTION),
    ("ANALYZER_AGENT_SYSTEM_PROMPT", ANALYZER_AGENT_SYSTEM_PROMPT),
    ("DEPLOY_AGENT_DESCRIPTION", DEPLOY_AGENT_DESCRIPTION),
    ("DEPLOY_AGENT_SYSTEM_PROMPT", DEPLOY_AGENT_SYSTEM_PROMPT),
    ("ORCHESTRATOR_SYSTEM_PROMPT", ORCHESTRATOR_SYSTEM_PROMPT),
]


class TestAgentPrompts:
    """Tests for the agent_prompts constants."""

    @pytest.mark.parametrize("name,value", ALL_PROMPTS)
    def test_prompt_is_non_empty_string(self, name: str, value: str) -> None:
        """Verifies every prompt constant is a non-empty string."""
        assert isinstance(value, str), f"{name} must be a string"
        assert len(value.strip()) > 0, f"{name} must not be empty"

    def test_orchestrator_prompt_lists_all_subagents(self) -> None:
        """Verifies the orchestrator prompt references every subagent name."""
        for subagent in ("drift_detector", "preprocessor", "trainer", "result_analyzer", "deployer"):
            assert subagent in ORCHESTRATOR_SYSTEM_PROMPT

    def test_system_prompts_contain_responsibilities_section(self) -> None:
        """Verifies each agent system prompt defines a responsibilities section."""
        system_prompts = [
            DRIFT_AGENT_SYSTEM_PROMPT,
            PREPROCESS_AGENT_SYSTEM_PROMPT,
            TRAINER_AGENT_SYSTEM_PROMPT,
            ANALYZER_AGENT_SYSTEM_PROMPT,
            DEPLOY_AGENT_SYSTEM_PROMPT,
        ]
        for prompt in system_prompts:
            assert "RESPONSIBILITIES" in prompt
