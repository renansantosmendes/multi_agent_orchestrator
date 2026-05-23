from __future__ import annotations

from src.core.agents.trainer_agent import trainer_agent


class TestTrainerAgent:
    """Tests for the trainer_agent SubAgent definition."""

    def test_has_required_keys(self) -> None:
        """Verifies the SubAgent dict contains all required keys."""
        assert "name" in trainer_agent
        assert "description" in trainer_agent
        assert "system_prompt" in trainer_agent
        assert "tools" in trainer_agent

    def test_name_is_correct(self) -> None:
        """Verifies the agent name matches the expected identifier."""
        assert trainer_agent["name"] == "trainer"

    def test_has_one_tool(self) -> None:
        """Verifies the agent exposes exactly one tool."""
        assert len(trainer_agent["tools"]) == 1

    def test_tool_name_is_train_model(self) -> None:
        """Verifies the registered tool is train_model."""
        assert trainer_agent["tools"][0].name == "train_model"
