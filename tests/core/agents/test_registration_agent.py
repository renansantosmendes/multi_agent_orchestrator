from src.core.agents.registration_agent import registration_agent
from src.core.tools.registration import register_model


def test_registration_agent_name():
    """registration_agent must be identified as 'registrar'."""
    assert registration_agent["name"] == "registrar"


def test_registration_agent_has_register_model_tool():
    """registration_agent must expose the register_model tool."""
    assert register_model in registration_agent["tools"]


def test_registration_agent_has_description():
    """registration_agent description must be a non-empty string."""
    assert isinstance(registration_agent["description"], str)
    assert len(registration_agent["description"]) > 0


def test_registration_agent_has_system_prompt():
    """registration_agent system_prompt must reference MLflow registry responsibilities."""
    prompt = registration_agent["system_prompt"]
    assert "RegistrationAgent" in prompt
    assert "register_model" in prompt
