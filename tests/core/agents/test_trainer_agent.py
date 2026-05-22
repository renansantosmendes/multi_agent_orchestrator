from __future__ import annotations

import numpy as np
import pytest
from langgraph.graph import END
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.core.agents.trainer_agent import AgentState, TrainerAgent


def make_classification_data() -> tuple[np.ndarray, np.ndarray]:
    input_data, output_data = make_classification(
        n_samples=100, n_features=4, random_state=42
    )
    return input_data, output_data


def make_regression_data() -> tuple[np.ndarray, np.ndarray]:
    input_data, output_data = make_regression(
        n_samples=100, n_features=4, random_state=42
    )
    return input_data, output_data


class TestTrainerAgentInit:
    def test_stores_model(self) -> None:
        model = RandomForestClassifier()
        agent = TrainerAgent(model=model)
        assert agent.model is model

    def test_compiles_graph(self) -> None:
        agent = TrainerAgent(model=RandomForestClassifier())
        assert agent.graph is not None

    def test_accepts_regressor(self) -> None:
        model = RandomForestRegressor()
        agent = TrainerAgent(model=model)
        assert agent.model is model


class TestTrainingNode:
    def test_training_only_routes_to_end(self) -> None:
        input_data, output_data = make_classification_data()
        model = RandomForestClassifier(random_state=42)
        agent = TrainerAgent(model=model)
        state: AgentState = {
            "train_input_data": input_data,
            "train_output_data": output_data,
            "model": model,
            "run_test": False,
        }
        command = agent._training_node(state)
        assert command.goto == END

    def test_training_with_test_routes_to_testing_node(self) -> None:
        input_data, output_data = make_classification_data()
        model = RandomForestClassifier(random_state=42)
        agent = TrainerAgent(model=model)
        state: AgentState = {
            "train_input_data": input_data,
            "train_output_data": output_data,
            "model": model,
            "run_test": True,
        }
        command = agent._training_node(state)
        assert command.goto == "testing_node"

    def test_training_node_updates_model(self) -> None:
        input_data, output_data = make_classification_data()
        model = RandomForestClassifier(random_state=42)
        agent = TrainerAgent(model=model)
        state: AgentState = {
            "train_input_data": input_data,
            "train_output_data": output_data,
            "model": model,
            "run_test": False,
        }
        command = agent._training_node(state)
        assert command.update["model"] is model


class TestTestingNode:
    def test_returns_predictions(self) -> None:
        input_data, output_data = make_classification_data()
        model = RandomForestClassifier(random_state=42)
        model.fit(input_data[:80], output_data[:80])
        agent = TrainerAgent(model=model)
        state: AgentState = {
            "test_input_data": input_data[80:],
            "test_output_data": output_data[80:],
            "model": model,
        }
        command = agent._testing_node(state)
        assert "predictions" in command.update
        assert command.update["predictions"].shape == output_data[80:].shape

    def test_returns_metrics(self) -> None:
        input_data, output_data = make_classification_data()
        model = RandomForestClassifier(random_state=42)
        model.fit(input_data[:80], output_data[:80])
        agent = TrainerAgent(model=model)
        state: AgentState = {
            "test_input_data": input_data[80:],
            "test_output_data": output_data[80:],
            "model": model,
        }
        command = agent._testing_node(state)
        assert "metrics" in command.update

    def test_routes_to_end(self) -> None:
        input_data, output_data = make_classification_data()
        model = RandomForestClassifier(random_state=42)
        model.fit(input_data[:80], output_data[:80])
        agent = TrainerAgent(model=model)
        state: AgentState = {
            "test_input_data": input_data[80:],
            "test_output_data": output_data[80:],
            "model": model,
        }
        command = agent._testing_node(state)
        assert command.goto == END


class TestComputeMetricsBinaryClassification:
    def setup_method(self) -> None:
        input_data, output_data = make_classification(n_samples=100, random_state=42)
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(input_data, output_data)
        self.predictions = self.model.predict(input_data)
        self.output_data = output_data

    def test_returns_classification_keys(self) -> None:
        metrics = TrainerAgent._compute_metrics(self.model, self.output_data, self.predictions)
        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}

    def test_accuracy_is_float(self) -> None:
        metrics = TrainerAgent._compute_metrics(self.model, self.output_data, self.predictions)
        assert isinstance(metrics["accuracy"], float)

    def test_accuracy_in_valid_range(self) -> None:
        metrics = TrainerAgent._compute_metrics(self.model, self.output_data, self.predictions)
        assert 0.0 <= metrics["accuracy"] <= 1.0


class TestComputeMetricsMulticlassClassification:
    def setup_method(self) -> None:
        input_data, output_data = make_classification(
            n_samples=150, n_classes=3, n_informative=4, random_state=42
        )
        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(input_data, output_data)
        self.predictions = self.model.predict(input_data)
        self.output_data = output_data

    def test_returns_classification_keys(self) -> None:
        metrics = TrainerAgent._compute_metrics(self.model, self.output_data, self.predictions)
        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}

    def test_all_values_are_floats(self) -> None:
        metrics = TrainerAgent._compute_metrics(self.model, self.output_data, self.predictions)
        for value in metrics.values():
            assert isinstance(value, float)


class TestComputeMetricsRegression:
    def setup_method(self) -> None:
        input_data, output_data = make_regression(n_samples=100, random_state=42)
        self.model = LinearRegression()
        self.model.fit(input_data, output_data)
        self.predictions = self.model.predict(input_data)
        self.output_data = output_data

    def test_returns_regression_keys(self) -> None:
        metrics = TrainerAgent._compute_metrics(self.model, self.output_data, self.predictions)
        assert set(metrics.keys()) == {"mean_squared_error", "mean_absolute_error", "r2"}

    def test_all_values_are_floats(self) -> None:
        metrics = TrainerAgent._compute_metrics(self.model, self.output_data, self.predictions)
        for value in metrics.values():
            assert isinstance(value, float)

    def test_mse_is_non_negative(self) -> None:
        metrics = TrainerAgent._compute_metrics(self.model, self.output_data, self.predictions)
        assert metrics["mean_squared_error"] >= 0.0

    def test_mae_is_non_negative(self) -> None:
        metrics = TrainerAgent._compute_metrics(self.model, self.output_data, self.predictions)
        assert metrics["mean_absolute_error"] >= 0.0


class TestTrainerAgentRunClassification:
    def setup_method(self) -> None:
        input_data, output_data = make_classification(n_samples=100, random_state=42)
        self.train_input = input_data[:80]
        self.train_output = output_data[:80]
        self.test_input = input_data[80:]
        self.test_output = output_data[80:]

    def test_run_training_only_returns_model(self) -> None:
        agent = TrainerAgent(model=RandomForestClassifier(random_state=42))
        result = agent.run(self.train_input, self.train_output)
        assert result["model"] is not None

    def test_run_training_only_has_no_metrics(self) -> None:
        agent = TrainerAgent(model=RandomForestClassifier(random_state=42))
        result = agent.run(self.train_input, self.train_output)
        assert result.get("metrics") is None

    def test_run_with_testing_returns_metrics(self) -> None:
        agent = TrainerAgent(model=RandomForestClassifier(random_state=42))
        result = agent.run(self.train_input, self.train_output, self.test_input, self.test_output)
        assert "metrics" in result
        assert set(result["metrics"].keys()) == {"accuracy", "precision", "recall", "f1"}

    def test_run_with_testing_returns_predictions(self) -> None:
        agent = TrainerAgent(model=RandomForestClassifier(random_state=42))
        result = agent.run(self.train_input, self.train_output, self.test_input, self.test_output)
        assert result["predictions"].shape == self.test_output.shape

    def test_only_train_input_provided_skips_test(self) -> None:
        agent = TrainerAgent(model=RandomForestClassifier(random_state=42))
        result = agent.run(self.train_input, self.train_output, self.test_input, None)
        assert result.get("metrics") is None


class TestTrainerAgentRunRegression:
    def setup_method(self) -> None:
        input_data, output_data = make_regression(n_samples=100, random_state=42)
        self.train_input = input_data[:80]
        self.train_output = output_data[:80]
        self.test_input = input_data[80:]
        self.test_output = output_data[80:]

    def test_run_with_testing_returns_regression_metrics(self) -> None:
        agent = TrainerAgent(model=RandomForestRegressor(random_state=42))
        result = agent.run(self.train_input, self.train_output, self.test_input, self.test_output)
        assert set(result["metrics"].keys()) == {"mean_squared_error", "mean_absolute_error", "r2"}
