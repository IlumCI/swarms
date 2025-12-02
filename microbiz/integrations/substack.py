"""
Substack API integration for newsletter/report publishing.

Handles post creation, publishing, and subscriber management.
"""

import os
import requests
from typing import Dict, Any, Optional
from loguru import logger


class SubstackIntegration:
    """
    Substack API integration for publishing data reports.
    """

    def __init__(self, api_key: Optional[str] = None, publication_id: Optional[str] = None):
        """
        Initialize Substack integration.

        Args:
            api_key (Optional[str]): Substack API key. Uses env var if None.
            publication_id (Optional[str]): Publication ID. Uses env var if None.
        """
        self.api_key = api_key or os.getenv("SUBSTACK_API_KEY")
        self.publication_id = publication_id or os.getenv("SUBSTACK_PUBLICATION_ID")
        self.base_url = "https://substack.com/api/v1"
        self.enabled = self.api_key is not None and self.publication_id is not None

        if not self.enabled:
            logger.warning("Substack API key or publication ID not set. Substack integration disabled.")

    def create_post(
        self,
        title: str,
        body: str,
        subtitle: Optional[str] = None,
        publish: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a post on Substack.

        Args:
            title (str): Post title.
            body (str): Post body content (markdown supported).
            subtitle (Optional[str]): Post subtitle.
            publish (bool): Whether to publish immediately.

        Returns:
            Dict[str, Any]: Post creation result.
        """
        if not self.enabled:
            logger.warning("Substack integration disabled. Simulating post creation.")
            return {
                "success": True,
                "post_id": f"sim_{title.lower().replace(' ', '_')}",
                "url": f"https://substack.com/p/{title.lower().replace(' ', '_')}",
                "simulated": True,
            }

        try:
            url = f"{self.base_url}/posts"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "publication_id": self.publication_id,
                "title": title,
                "body": body,
                "subtitle": subtitle,
                "draft": not publish,
            }

            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

            logger.info(f"Substack post created: {title} (published: {publish})")
            return {
                "success": True,
                "post_id": result.get("id"),
                "url": result.get("url"),
            }

        except Exception as e:
            logger.error(f"Error creating Substack post: {e}")
            return {"success": False, "error": str(e)}

