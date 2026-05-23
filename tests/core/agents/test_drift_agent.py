from __future__ import annotations

from src.core.agents.drift_agent import drift_agent


class TestDriftAgent:
    """Tests for the drift_agent SubAgent definition."""

    def test_has_required_keys(self) -> None:
        """Verifies the SubAgent dict contains all required keys."""
        assert "name" in drift_agent
        assert "description" in drift_agent
        assert "system_prompt" in drift_agent
        assert "tools" in drift_agent

    def test_name_is_correct(self) -> None:
        """Verifies the agent name matches the expected identifier."""
        assert drift_agent["name"] == "drift_detector"

    def test_has_one_tool(self) -> None:
        """Verifies the agent exposes exactly one tool."""
        assert len(drift_agent["tools"]) == 1

    def test_tool_name_is_detect_data_drift(self) -> None:
        """Verifies the registered tool is detect_data_drift."""
        assert drift_agent["tools"][0].name == "detect_data_drift"

    def test_description_is_non_empty(self) -> None:
        """Verifies the agent description is a non-empty string."""
        assert isinstance(drift_agent["description"], str)
        assert len(drift_agent["description"]) > 0

    def test_system_prompt_is_non_empty(self) -> None:
        """Verifies the agent system prompt is a non-empty string."""
        assert isinstance(drift_agent["system_prompt"], str)
        assert len(drift_agent["system_prompt"]) > 0
