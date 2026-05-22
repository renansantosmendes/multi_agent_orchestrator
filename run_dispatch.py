from __future__ import annotations

import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()


@tool
def trigger_github_action(
    owner: str,
    repo: str,
    workflow: str,
    branch: str = "main",
) -> dict:
    """Trigger a GitHub Actions workflow via the GitHub REST API.

    Args:
        owner: GitHub organization or user that owns the repository.
        repo: Repository name.
        workflow: Workflow file name (e.g. ``deploy.yml``).
        branch: Branch or tag ref to run the workflow on.

    Returns:
        A dictionary with ``status_code``, ``success``, ``workflow``, and
        ``branch`` keys.
    """
    url = (
        f"https://api.github.com/repos/{owner}/{repo}"
        f"/actions/workflows/{workflow}/dispatches"
    )
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN', '')}",
            "Accept": "application/vnd.github+json",
        },
        json={"ref": branch},
        timeout=30,
    )
    return {
        "status_code": response.status_code,
        "success": response.status_code == 204,
        "workflow": workflow,
        "branch": branch,
    }


def build_agent():
    """Construct and return the LangGraph react agent.

    Returns:
        A compiled LangGraph agent with the GitHub Actions dispatch tool.
    """
    llm = ChatOpenAI(
        model="gpt-4.1-mini", 
        temperature=0
        )
    return create_react_agent(
        model=llm,
        tools=[trigger_github_action],
        prompt="You are an agent that triggers GitHub Actions workflows.",
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
            "Execute o workflow CI"
            "do repositório multi_agent_orchestrator "
            "na branch main "
            "do owner renansantosmendes",
        )]
    })
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
