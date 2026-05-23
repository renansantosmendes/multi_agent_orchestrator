from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.core.agents.analyzer_agent import analyzer_agent
from src.core.agents.deploy_agent import deploy_agent
from src.core.agents.drift_agent import drift_agent
from src.core.agents.preprocess_agent import preprocess_agent
from src.core.agents.trainer_agent import trainer_agent
from src.core.prompts.agent_prompts import ORCHESTRATOR_SYSTEM_PROMPT
from src.core.schema import MLPipelineContext


def create_ml_orchestrator(model_name: str = "gpt-4o-mini", temperature: float = 0) -> object:
    """Creates the ML pipeline orchestrator with a subagent architecture.

    Args:
        model_name: OpenAI model identifier to use for the orchestrator LLM.
        temperature: Sampling temperature for the LLM (0 = deterministic).

    Returns:
        A compiled LangGraph agent ready to stream pipeline events.
    """
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    return create_deep_agent(
        model=llm,
        tools=[],
        subagents=[
            drift_agent,
            preprocess_agent,
            trainer_agent,
            analyzer_agent,
            deploy_agent,
        ],
        context_schema=MLPipelineContext,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
    )


def run_pipeline() -> None:
    """Runs the full ML pipeline using the multi-agent orchestrator."""
    orchestrator = create_ml_orchestrator()

    initial_state = {
        "messages": [
            HumanMessage(
                content=(
                    "Run the full ML pipeline. "
                    "Check for drift, preprocess the data, "
                    "train a RandomForest and a LogisticRegression, "
                    "analyse the results, and deploy the best model."
                )
            )
        ],
        "drift_detected": False,
        "drift_summary": "",
        "drifted_columns": [],
        "preprocessing_done": False,
        "feature_columns": [],
        "target_column": "",
        "accuracy_history": [],
        "trained_models": [],
        "best_model_name": "",
        "best_accuracy": 0.0,
        "model_path": "",
        "current_stage": "start",
    }

    print("🚀 Starting ML Pipeline with SubAgents...\n")

    try:
        for event in orchestrator.stream(initial_state, stream_mode="values"):
            if "messages" in event:
                last_msg = event["messages"][-1]
                if hasattr(last_msg, "content") and last_msg.content:
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            agent_name = tc.get("args", {}).get("subagent_type", tc.get("name", "?"))
                            print(f"🛠️  Delegating to: {agent_name}")
                    elif isinstance(last_msg, ToolMessage):
                        print(f"📦 Result:\n{last_msg.content}\n")
                    else:
                        print(f"🤖 Orchestrator:\n{last_msg.content}\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
