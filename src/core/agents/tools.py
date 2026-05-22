from __future__ import annotations

import os

import requests
from langchain_core.tools import tool


@tool
def list_github_workflows(owner: str, repo: str) -> dict:
    """List all workflows available in a GitHub repository.

    Args:
        owner: GitHub organization or user that owns the repository.
        repo: Repository name.

    Returns:
        A dictionary with ``total_count`` and a ``workflows`` list, each
        entry containing ``id``, ``name``, and ``path``.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows"
    response = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN', '')}",
            "Accept": "application/vnd.github+json",
        },
        timeout=30,
    )
    data = response.json()
    return {
        "total_count": data.get("total_count", 0),
        "workflows": [
            {"id": w["id"], "name": w["name"], "path": w["path"]}
            for w in data.get("workflows", [])
        ],
    }


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
        workflow: Workflow file name (e.g. ``ci.yml``) or numeric workflow id.
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
