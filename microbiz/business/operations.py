"""
Business operations automation module.

Handles sales automation, marketing, customer service, and inventory management.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger


class OperationsManager:
    """
    Manages business operations: sales, marketing, customer service, inventory.
    """

    def __init__(self, data_dir: str = "output/operations"):
        """
        Initialize operations manager.

        Args:
            data_dir (str): Directory for operations data.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.inventory_file = self.data_dir / "inventory.json"

    def create_gumroad_listing(
        self, product: Dict[str, Any], price: float, description: str, autonomy_mode: str = "advisory"
    ) -> Dict[str, Any]:
        """
        Create a Gumroad product listing.

        Args:
            product (Dict[str, Any]): Product information.
            price (float): Product price.
            description (str): Product description.
            autonomy_mode (str): Autonomy mode (advisory, semi_auto, full_auto).

        Returns:
            Dict[str, Any]: Listing information.
        """
        from integrations.gumroad import GumroadIntegration

        gumroad = GumroadIntegration()
        listing = {
            "platform": "gumroad",
            "product_id": product.get("product_id"),
            "title": f"{product.get('data_type', 'Data')} Dataset - {product.get('item_count', 0)} items",
            "price": price,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "status": "pending",
        }

        # Real API call if enabled and autonomy mode allows
        if autonomy_mode in ["semi_auto", "full_auto"] and gumroad.enabled:
            try:
                result = gumroad.create_product(listing["title"], price, description)
                if result.get("success"):
                    listing["status"] = "active"
                    listing["gumroad_url"] = result.get("url")
                    listing["gumroad_product_id"] = result.get("product_id")
            except Exception as e:
                logger.error(f"Error creating Gumroad product: {e}")

        logger.info(f"Gumroad listing: {listing['title']} at ${price:.2f} (status: {listing['status']})")
        return listing

    def create_substack_post(
        self, title: str, content: str, data_summary: Dict[str, Any], autonomy_mode: str = "advisory", publish: bool = False
    ) -> Dict[str, Any]:
        """
        Create a Substack newsletter post.

        Args:
            title (str): Post title.
            content (str): Post content.
            data_summary (Dict[str, Any]): Data summary to include.
            autonomy_mode (str): Autonomy mode.
            publish (bool): Whether to publish immediately.

        Returns:
            Dict[str, Any]: Post information.
        """
        from integrations.substack import SubstackIntegration

        substack = SubstackIntegration()
        post = {
            "platform": "substack",
            "title": title,
            "content": content,
            "data_summary": data_summary,
            "created_at": datetime.utcnow().isoformat(),
            "status": "draft",
        }

        # Real API call if enabled and autonomy mode allows
        if autonomy_mode in ["semi_auto", "full_auto"] and substack.enabled:
            try:
                result = substack.create_post(title, content, publish=publish)
                if result.get("success"):
                    post["status"] = "published" if publish else "draft"
                    post["substack_url"] = result.get("url")
            except Exception as e:
                logger.error(f"Error creating Substack post: {e}")

        logger.info(f"Substack post: {title} (status: {post['status']})")
        return post

    def create_github_release(
        self, repo: str, tag: str, data_file: str, description: str, autonomy_mode: str = "advisory"
    ) -> Dict[str, Any]:
        """
        Create a GitHub release for free sample.

        Args:
            repo (str): Repository name.
            tag (str): Release tag.
            data_file (str): Data file to include.
            description (str): Release description.
            autonomy_mode (str): Autonomy mode.

        Returns:
            Dict[str, Any]: Release information.
        """
        from integrations.github import GitHubIntegration

        github = GitHubIntegration(repo=repo)
        release = {
            "platform": "github",
            "repo": repo,
            "tag": tag,
            "data_file": data_file,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "status": "pending",
        }

        # Real API call if enabled and autonomy mode allows (GitHub is safest)
        if autonomy_mode in ["semi_auto", "full_auto"] and github.enabled:
            try:
                result = github.create_release(tag, description, description, files=[data_file] if data_file else None)
                if result.get("success") or result.get("id"):
                    release["status"] = "published"
                    release["github_url"] = result.get("html_url")
            except Exception as e:
                logger.error(f"Error creating GitHub release: {e}")

        logger.info(f"GitHub release: {tag} for {repo} (status: {release['status']})")
        return release

    def track_inventory(self, product_id: str, quantity: int, product_type: str) -> None:
        """
        Track inventory for products.

        Args:
            product_id (str): Product identifier.
            quantity (int): Available quantity.
            product_type (str): Type of product.
        """
        inventory = self._load_inventory()
        inventory[product_id] = {
            "product_id": product_id,
            "quantity": quantity,
            "product_type": product_type,
            "last_updated": datetime.utcnow().isoformat(),
        }
        self._save_inventory(inventory)
        logger.info(f"Inventory tracked: {product_id} - {quantity} units")

    def _load_inventory(self) -> Dict[str, Any]:
        """Load inventory data."""
        if self.inventory_file.exists():
            try:
                with open(self.inventory_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading inventory: {e}")
        return {}

    def _save_inventory(self, inventory: Dict[str, Any]) -> None:
        """Save inventory data."""
        try:
            with open(self.inventory_file, "w") as f:
                json.dump(inventory, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving inventory: {e}")

