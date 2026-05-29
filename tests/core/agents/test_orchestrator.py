from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.core.agents.orchestrator import create_ml_orchestrator, run_pipeline


class TestCreateMlOrchestrator:
    """Tests for the create_ml_orchestrator factory function."""

    @patch("src.core.agents.orchestrator.create_deep_agent")
    @patch("src.core.agents.orchestrator.ChatOpenAI")
    def test_returns_orchestrator_object(
        self, mock_chat_openai: MagicMock, mock_create_deep_agent: MagicMock
    ) -> None:
        """Verifies the function returns whatever create_deep_agent produces."""
        expected = MagicMock()
        mock_create_deep_agent.return_value = expected

        result = create_ml_orchestrator()

        assert result is expected

    @patch("src.core.agents.orchestrator.create_deep_agent")
    @patch("src.core.agents.orchestrator.ChatOpenAI")
    def test_uses_provided_model_name(
        self, mock_chat_openai: MagicMock, mock_create_deep_agent: MagicMock
    ) -> None:
        """Verifies the LLM is instantiated with the given model name."""
        create_ml_orchestrator(model_name="gpt-4o")

        mock_chat_openai.assert_called_once_with(model="gpt-4o", temperature=0)

    @patch("src.core.agents.orchestrator.create_deep_agent")
    @patch("src.core.agents.orchestrator.ChatOpenAI")
    def test_passes_six_subagents(
        self, mock_chat_openai: MagicMock, mock_create_deep_agent: MagicMock
    ) -> None:
        """Verifies that all six subagents are passed to create_deep_agent."""
        create_ml_orchestrator()

        _, kwargs = mock_create_deep_agent.call_args
        assert len(kwargs["subagents"]) == 6

    @patch("src.core.agents.orchestrator.create_deep_agent")
    @patch("src.core.agents.orchestrator.ChatOpenAI")
    def test_orchestrator_has_no_direct_tools(
        self, mock_chat_openai: MagicMock, mock_create_deep_agent: MagicMock
    ) -> None:
        """Verifies the orchestrator has no direct tools (delegation only)."""
        create_ml_orchestrator()

        _, kwargs = mock_create_deep_agent.call_args
        assert kwargs["tools"] == []


class TestRunPipeline:
    """Tests for the run_pipeline execution function."""

    @patch("src.core.agents.orchestrator.create_ml_orchestrator")
    def test_creates_orchestrator(self, mock_create: MagicMock) -> None:
        """Verifies that run_pipeline creates an orchestrator before streaming."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.stream.return_value = []
        mock_create.return_value = mock_orchestrator

        run_pipeline()

        mock_create.assert_called_once()

    @patch("src.core.agents.orchestrator.create_ml_orchestrator")
    def test_streams_with_values_mode(self, mock_create: MagicMock) -> None:
        """Verifies the orchestrator is streamed in 'values' mode."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.stream.return_value = []
        mock_create.return_value = mock_orchestrator

        run_pipeline()

        mock_orchestrator.stream.assert_called_once()
        _, kwargs = mock_orchestrator.stream.call_args
        assert kwargs.get("stream_mode") == "values"

    @patch("src.core.agents.orchestrator.create_ml_orchestrator")
    def test_initial_state_has_required_keys(self, mock_create: MagicMock) -> None:
        """Verifies the initial state passed to the orchestrator has all required keys."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.stream.return_value = []
        mock_create.return_value = mock_orchestrator

        run_pipeline()

        initial_state = mock_orchestrator.stream.call_args.args[0]
        for key in ("messages", "drift_detected", "preprocessing_done",
                    "accuracy_history", "model_path", "current_stage"):
            assert key in initial_state

    @patch("src.core.agents.orchestrator.create_ml_orchestrator")
    def test_handles_stream_exception_without_raising(self, mock_create: MagicMock) -> None:
        """Verifies that a streaming exception is caught and does not propagate."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.stream.side_effect = RuntimeError("connection error")
        mock_create.return_value = mock_orchestrator

        run_pipeline()
