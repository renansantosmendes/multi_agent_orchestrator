from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.core.agents.tools import list_github_workflows, trigger_github_action

load_dotenv()


def build_agent():
    """Construct and return the LangGraph react agent.

    Returns:
        A compiled LangGraph agent with GitHub Actions tools.
    """
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    return create_react_agent(
        model=llm,
        tools=[list_github_workflows, trigger_github_action],
        prompt=(
            "You are an agent that manages GitHub Actions workflows. "
            "When asked to trigger a workflow, first list the available "
            "workflows to confirm the correct file name, then trigger it."
        ),
    )


def run_with_streaming(user_input: str) -> None:
    """Invoke the agent and print each reasoning step as it happens.

    Args:
        user_input: The natural-language instruction for the agent.
    """
    agent = build_agent()

    print("=" * 60)
    print(f"INPUT: {user_input}")
    print("=" * 60)

    for chunk in agent.stream({"messages": [("human", user_input)]}):
        for node_name, node_output in chunk.items():
            for message in node_output.get("messages", []):
                _print_message(node_name, message)

    print("=" * 60)


def _print_message(node_name: str, message: object) -> None:
    """Print a single agent message with its role and content.

    Args:
        node_name: Name of the LangGraph node that produced the message.
        message: A LangChain message object.
    """
    role = type(message).__name__.replace("Message", "").upper()
    print(f"\n[{node_name}] {role}")
    print("-" * 40)

    if hasattr(message, "tool_calls") and message.tool_calls:
        for call in message.tool_calls:
            print(f"  TOOL CALL → {call['name']}")
            for key, value in call["args"].items():
                print(f"    {key}: {value}")

    elif hasattr(message, "name") and message.name:
        print(f"  TOOL RESULT ({message.name}):")
        print(f"  {message.content}")

    else:
        print(f"  {message.content}")


def main() -> None:
    """Run the GitHub Actions agent with a sample instruction."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in the .env file")
    if not os.environ.get("GITHUB_TOKEN"):
        raise ValueError("Set GITHUB_TOKEN in the .env file")

    run_with_streaming(
        "Liste os workflows disponíveis no repositório multi_agent_orchestrator "
        "do owner renansantosmendes e dispare o workflow de CI na branch main."
    )


if __name__ == "__main__":
    main()
