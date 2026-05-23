from __future__ import annotations

from src.core.agents.preprocess_agent import preprocess_agent


class TestPreprocessAgent:
    """Tests for the preprocess_agent SubAgent definition."""

    def test_has_required_keys(self) -> None:
        """Verifies the SubAgent dict contains all required keys."""
        assert "name" in preprocess_agent
        assert "description" in preprocess_agent
        assert "system_prompt" in preprocess_agent
        assert "tools" in preprocess_agent

    def test_name_is_correct(self) -> None:
        """Verifies the agent name matches the expected identifier."""
        assert preprocess_agent["name"] == "preprocessor"

    def test_has_one_tool(self) -> None:
        """Verifies the agent exposes exactly one tool."""
        assert len(preprocess_agent["tools"]) == 1

    def test_tool_name_is_preprocess_data(self) -> None:
        """Verifies the registered tool is preprocess_data."""
        assert preprocess_agent["tools"][0].name == "preprocess_data"
