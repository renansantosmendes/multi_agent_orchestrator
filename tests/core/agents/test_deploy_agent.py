from __future__ import annotations

from src.core.agents.deploy_agent import deploy_agent


class TestDeployAgent:
    """Tests for the deploy_agent SubAgent definition."""

    def test_has_required_keys(self) -> None:
        """Verifies the SubAgent dict contains all required keys."""
        assert "name" in deploy_agent
        assert "description" in deploy_agent
        assert "system_prompt" in deploy_agent
        assert "tools" in deploy_agent

    def test_name_is_correct(self) -> None:
        """Verifies the agent name matches the expected identifier."""
        assert deploy_agent["name"] == "deployer"

    def test_has_one_tool(self) -> None:
        """Verifies the agent exposes exactly one tool."""
        assert len(deploy_agent["tools"]) == 1

    def test_tool_name_is_deploy_model(self) -> None:
        """Verifies the registered tool is deploy_model."""
        assert deploy_agent["tools"][0].name == "deploy_model"
