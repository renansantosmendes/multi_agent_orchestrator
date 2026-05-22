from langchain.tools import tool
import requests

@tool
def trigger_github_action(
    owner: str,
    repo: str,
    workflow: str,
    branch: str = "main",
):
    """
    Dispara um workflow do GitHub Actions.
    """

    url = (
        f"https://api.github.com/repos/"
        f"{owner}/{repo}/actions/workflows/"
        f"{workflow}/dispatches"
    )

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.github+json",
        },
        json={"ref": branch},
    )

    return {
        "status_code": response.status_code,
        "success": response.status_code == 204,
    }