"""
GitHub API integration for free sample releases.

Handles repository releases, file uploads, and version management.
"""

import os
import base64
import requests
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger


class GitHubIntegration:
    """
    GitHub API integration for publishing free data samples.
    """

    def __init__(self, token: Optional[str] = None, repo: Optional[str] = None):
        """
        Initialize GitHub integration.

        Args:
            token (Optional[str]): GitHub personal access token. Uses env var if None.
            repo (Optional[str]): Repository in format "owner/repo". Uses env var if None.
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.repo = repo or os.getenv("GITHUB_REPO")
        self.base_url = "https://api.github.com"
        self.enabled = self.token is not None and self.repo is not None

        if not self.enabled:
            logger.warning("GitHub token or repo not set. GitHub integration disabled.")

    def create_release(
        self,
        tag: str,
        title: str,
        body: str,
        files: Optional[list[str]] = None,
        draft: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a GitHub release.

        Args:
            tag (str): Release tag (e.g., "v1.0.0").
            title (str): Release title.
            body (str): Release description.
            files (Optional[list[str]]): List of file paths to attach.
            draft (bool): Whether to create as draft.

        Returns:
            Dict[str, Any]: Release creation result.
        """
        if not self.enabled:
            logger.warning("GitHub integration disabled. Simulating release creation.")
            return {
                "success": True,
                "release_id": f"sim_{tag}",
                "url": f"https://github.com/{self.repo}/releases/tag/{tag}",
                "simulated": True,
            }

        try:
            # Create release
            url = f"{self.base_url}/repos/{self.repo}/releases"
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
            }
            data = {
                "tag_name": tag,
                "name": title,
                "body": body,
                "draft": draft,
            }

            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            release = response.json()

            # Upload files if provided
            if files and release.get("upload_url"):
                upload_url = release["upload_url"].split("{")[0]
                for file_path in files:
                    self._upload_release_asset(upload_url, file_path, headers)

            logger.info(f"GitHub release created: {tag} - {title}")
            return {
                "success": True,
                "id": release.get("id"),
                "html_url": release.get("html_url"),
                "tag": tag,
            }

        except Exception as e:
            logger.error(f"Error creating GitHub release: {e}")
            return {"success": False, "error": str(e)}

    def _upload_release_asset(
        self, upload_url: str, file_path: str, headers: Dict[str, str]
    ) -> None:
        """
        Upload a file as a release asset.

        Args:
            upload_url (str): GitHub upload URL.
            file_path (str): Path to file to upload.
            headers (Dict[str, str]): Request headers.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                return

            file_name = path.name
            content_type = "application/octet-stream"

            upload_headers = headers.copy()
            upload_headers["Content-Type"] = content_type

            with open(file_path, "rb") as f:
                response = requests.post(
                    f"{upload_url}?name={file_name}",
                    headers=upload_headers,
                    data=f.read(),
                )
                response.raise_for_status()

            logger.info(f"Uploaded {file_name} to release")

        except Exception as e:
            logger.error(f"Error uploading release asset {file_path}: {e}")

