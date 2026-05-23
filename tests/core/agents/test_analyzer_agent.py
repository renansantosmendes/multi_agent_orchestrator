from __future__ import annotations

from src.core.agents.analyzer_agent import analyzer_agent


class TestAnalyzerAgent:
    """Tests for the analyzer_agent SubAgent definition."""

    def test_has_required_keys(self) -> None:
        """Verifies the SubAgent dict contains all required keys."""
        assert "name" in analyzer_agent
        assert "description" in analyzer_agent
        assert "system_prompt" in analyzer_agent
        assert "tools" in analyzer_agent

    def test_name_is_correct(self) -> None:
        """Verifies the agent name matches the expected identifier."""
        assert analyzer_agent["name"] == "result_analyzer"

    def test_has_one_tool(self) -> None:
        """Verifies the agent exposes exactly one tool."""
        assert len(analyzer_agent["tools"]) == 1

    def test_tool_name_is_analyze_results(self) -> None:
        """Verifies the registered tool is analyze_results."""
        assert analyzer_agent["tools"][0].name == "analyze_results"
