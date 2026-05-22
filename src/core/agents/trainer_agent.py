from __future__ import annotations

from typing import Any, Literal, Optional, Union

import numpy as np
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from sklearn.base import BaseEstimator, is_classifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.utils.multiclass import type_of_target
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Shared state flowing through the LangGraph agent."""

    train_input_data: np.ndarray
    train_output_data: np.ndarray
    test_input_data: np.ndarray
    test_output_data: np.ndarray
    model: BaseEstimator
    predictions: np.ndarray
    metrics: dict[str, float]
    run_test: bool


class TrainerAgent:
    """A LangGraph-based agent that trains and evaluates any scikit-learn estimator.

    The agent exposes a two-node graph (training -> testing) whose routing is
    controlled via ``Command`` objects.  Each node both updates the shared state
    and decides the next destination, removing the need for separate conditional
    edge functions.

    Attributes:
        model: The scikit-learn estimator used for training and prediction.
        graph: The compiled LangGraph ``StateGraph`` ready for invocation.
    """

    def __init__(self, model: BaseEstimator) -> None:
        """Initialise the agent with a scikit-learn compatible estimator.

        Args:
            model: An already-instantiated estimator that implements ``fit``
                and ``predict`` (e.g. ``RandomForestClassifier()``).
        """
        self.model = model
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct and compile the LangGraph state graph.

        Nodes are connected exclusively through ``Command`` objects returned
        by each node function — no ``add_edge`` or ``add_conditional_edges``
        calls are used between internal nodes.

        Returns:
            A compiled ``StateGraph`` with training and testing nodes.
        """
        builder = StateGraph(AgentState)

        builder.add_node("training_node", self._training_node)
        builder.add_node("testing_node", self._testing_node)

        builder.add_edge(START, "training_node")

        return builder.compile()

    def _training_node(
        self, state: AgentState
    ) -> Command[Literal["testing_node", "__end__"]]:
        """Fit the model and route to testing or end.

        After fitting, the node inspects ``run_test`` to decide whether to
        proceed to the testing node or finish the graph execution.

        Args:
            state: Must contain ``train_input_data`` and ``train_output_data``.

        Returns:
            A ``Command`` that updates the state with the fitted model and
            navigates to ``testing_node`` or ``END``.
        """
        self.model.fit(state["train_input_data"], state["train_output_data"])

        next_node: str = "testing_node" if state.get("run_test", False) else END

        return Command(
            update={"model": self.model},
            goto=next_node,
        )

    def _testing_node(
        self, state: AgentState
    ) -> Command[Literal["__end__"]]:
        """Generate predictions, compute metrics, and finish.

        Automatically detects whether the task is classification or regression
        and returns the appropriate metric set.

        Args:
            state: Must contain ``test_input_data``, ``test_output_data``, and
                a fitted ``model``.

        Returns:
            A ``Command`` that updates the state with predictions and metrics,
            then navigates to ``END``.
        """
        predictions = self.model.predict(state["test_input_data"])
        metrics = self._compute_metrics(
            self.model, state["test_output_data"], predictions
        )

        return Command(
            update={"predictions": predictions, "metrics": metrics},
            goto=END,
        )

    @staticmethod
    def _compute_metrics(
        model: BaseEstimator,
        true_values: np.ndarray,
        predicted_values: np.ndarray,
    ) -> dict[str, float]:
        """Compute evaluation metrics appropriate to the estimator type.

        Uses ``sklearn.base.is_classifier`` to decide between classification
        and regression metrics, which is more reliable than inspecting the
        target array alone.

        Args:
            model: The fitted estimator (used only for type detection).
            true_values: Ground-truth labels or values.
            predicted_values: Model predictions.

        Returns:
            A dictionary of metric names to their computed values.
        """
        if is_classifier(model):
            target_kind = type_of_target(true_values)
            average = "binary" if target_kind == "binary" else "weighted"
            return {
                "accuracy": float(accuracy_score(true_values, predicted_values)),
                "precision": float(
                    precision_score(true_values, predicted_values, average=average)
                ),
                "recall": float(
                    recall_score(true_values, predicted_values, average=average)
                ),
                "f1": float(
                    f1_score(true_values, predicted_values, average=average)
                ),
            }

        return {
            "mean_squared_error": float(
                mean_squared_error(true_values, predicted_values)
            ),
            "mean_absolute_error": float(
                mean_absolute_error(true_values, predicted_values)
            ),
            "r2": float(r2_score(true_values, predicted_values)),
        }

    def run(
        self,
        train_input_data: np.ndarray,
        train_output_data: np.ndarray,
        test_input_data: Optional[np.ndarray] = None,
        test_output_data: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Execute the agent graph (training and, optionally, testing).

        When ``test_input_data`` and ``test_output_data`` are provided the full
        pipeline runs (train -> test).  Otherwise only training is executed.

        Args:
            train_input_data: Feature matrix for training.
            train_output_data: Target vector for training.
            test_input_data: Feature matrix for testing (optional).
            test_output_data: Target vector for testing (optional).

        Returns:
            The final ``AgentState`` dictionary produced by the graph.
        """
        has_test_data = test_input_data is not None and test_output_data is not None

        initial_state: AgentState = {
            "train_input_data": train_input_data,
            "train_output_data": train_output_data,
            "model": self.model,
            "run_test": has_test_data,
        }

        if has_test_data:
            initial_state["test_input_data"] = test_input_data
            initial_state["test_output_data"] = test_output_data

        return self.graph.invoke(initial_state)