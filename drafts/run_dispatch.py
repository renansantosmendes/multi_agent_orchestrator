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


def main() -> None:
    """Run the GitHub Actions dispatch agent."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in the .env file")
    if not os.environ.get("GITHUB_TOKEN"):
        raise ValueError("Set GITHUB_TOKEN in the .env file")

    agent = build_agent()
    result = agent.invoke({
        "messages": [(
            "human",
            "Execute o workflow ci.yml"
            "do repositório multi_agent_orchestrator "
            "na branch main "
            "do owner renansantosmendes",
        )]
    })
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
