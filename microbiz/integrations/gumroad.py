"""
Gumroad API integration for dataset sales.

Handles product creation, listing management, and sales tracking.
"""

import os
import requests
from typing import Dict, Any, Optional
from loguru import logger


class GumroadIntegration:
    """
    Gumroad API integration for selling data products.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gumroad integration.

        Args:
            api_key (Optional[str]): Gumroad API key. Uses env var if None.
        """
        self.api_key = api_key or os.getenv("GUMROAD_API_KEY")
        self.base_url = "https://api.gumroad.com/v2"
        self.enabled = self.api_key is not None

        if not self.enabled:
            logger.warning("Gumroad API key not set. Gumroad integration disabled.")

    def create_product(
        self,
        name: str,
        price: float,
        description: str,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a product on Gumroad.

        Args:
            name (str): Product name.
            price (float): Product price in USD.
            description (str): Product description.
            file_path (Optional[str]): Path to product file.

        Returns:
            Dict[str, Any]: Product creation result.
        """
        if not self.enabled:
            logger.warning("Gumroad integration disabled. Simulating product creation.")
            return {
                "success": True,
                "product_id": f"sim_{name.lower().replace(' ', '_')}",
                "url": f"https://gumroad.com/l/{name.lower().replace(' ', '_')}",
                "simulated": True,
            }

        try:
            url = f"{self.base_url}/products"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            data = {
                "name": name,
                "price": int(price * 100),  # Price in cents
                "description": description,
            }

            if file_path:
                files = {"file": open(file_path, "rb")}
                response = requests.post(url, headers=headers, data=data, files=files)
                files["file"].close()
            else:
                response = requests.post(url, headers=headers, data=data)

            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                logger.info(f"Gumroad product created: {name} at ${price:.2f}")
                return result
            else:
                logger.error(f"Gumroad product creation failed: {result}")
                return {"success": False, "error": result}

        except Exception as e:
            logger.error(f"Error creating Gumroad product: {e}")
            return {"success": False, "error": str(e)}

    def get_sales(self, product_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get sales data from Gumroad.

        Args:
            product_id (Optional[str]): Specific product ID, or None for all products.

        Returns:
            Dict[str, Any]: Sales data.
        """
        if not self.enabled:
            return {"sales": [], "simulated": True}

        try:
            url = f"{self.base_url}/sales"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            params = {}
            if product_id:
                params["product_id"] = product_id

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Error fetching Gumroad sales: {e}")
            return {"sales": [], "error": str(e)}

