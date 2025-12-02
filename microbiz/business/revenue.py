"""
Revenue generation strategies module.

Handles multiple revenue models: subscriptions, one-time sales, tiered pricing, API access.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

from business.finance import FinancialManager


class RevenueGenerator:
    """
    Revenue generation strategies for data products.
    """

    def __init__(self, finance_manager: Optional[FinancialManager] = None):
        """
        Initialize revenue generator.

        Args:
            finance_manager (Optional[FinancialManager]): Financial manager for tracking.
        """
        self.finance_manager = finance_manager or FinancialManager()
        self.pricing_models = {
            "subscription": {"weekly": 9.99, "monthly": 29.99, "yearly": 299.99},
            "one_time": {"basic": 19.99, "premium": 49.99, "enterprise": 199.99},
            "tiered": {"free": 0.0, "basic": 9.99, "premium": 29.99, "enterprise": 99.99},
            "api": {"per_request": 0.01, "monthly_1000": 49.99, "monthly_10000": 299.99},
        }

    def generate_subscription_revenue(
        self, product_name: str, tier: str = "monthly", quantity: int = 1
    ) -> float:
        """
        Generate revenue from subscription sales.

        Args:
            product_name (str): Name of the product.
            tier (str): Subscription tier (weekly, monthly, yearly).
            quantity (int): Number of subscriptions.

        Returns:
            float: Total revenue generated.
        """
        price = self.pricing_models["subscription"].get(tier, 29.99)
        total = price * quantity

        self.finance_manager.track_revenue(
            source="subscription",
            amount=total,
            product_type=f"{product_name}_{tier}",
        )

        logger.info(f"Subscription revenue: ${total:.2f} for {quantity} {tier} subscriptions")
        return total

    def generate_one_time_revenue(
        self, product_name: str, tier: str = "premium", quantity: int = 1
    ) -> float:
        """
        Generate revenue from one-time dataset sales.

        Args:
            product_name (str): Name of the product.
            tier (str): Product tier (basic, premium, enterprise).
            quantity (int): Number of sales.

        Returns:
            float: Total revenue generated.
        """
        price = self.pricing_models["one_time"].get(tier, 49.99)
        total = price * quantity

        self.finance_manager.track_revenue(
            source="one_time",
            amount=total,
            product_type=f"{product_name}_{tier}",
        )

        logger.info(f"One-time revenue: ${total:.2f} for {quantity} {tier} products")
        return total

    def generate_api_revenue(self, requests: int, tier: str = "per_request") -> float:
        """
        Generate revenue from API usage.

        Args:
            requests (int): Number of API requests.
            tier (str): Pricing tier.

        Returns:
            float: Total revenue generated.
        """
        if tier == "per_request":
            price_per_request = self.pricing_models["api"]["per_request"]
            total = price_per_request * requests
        else:
            total = self.pricing_models["api"].get(tier, 49.99)

        self.finance_manager.track_revenue(
            source="api",
            amount=total,
            product_type=f"api_{tier}",
        )

        logger.info(f"API revenue: ${total:.2f} for {requests} requests ({tier})")
        return total

    def create_product_package(
        self, data_type: str, data: List[Dict[str, Any]], format: str = "json"
    ) -> Dict[str, Any]:
        """
        Create a product package from collected data.

        Args:
            data_type (str): Type of data (auctions, tenders, etc.).
            data (List[Dict[str, Any]]): Data to package.
            format (str): Output format (json, csv, pdf).

        Returns:
            Dict[str, Any]: Product package information.
        """
        package = {
            "product_id": f"{data_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "data_type": data_type,
            "item_count": len(data),
            "format": format,
            "created_at": datetime.utcnow().isoformat(),
            "estimated_value": self._estimate_value(data_type, len(data)),
        }

        logger.info(f"Product package created: {package['product_id']} with {len(data)} items")
        return package

    def _estimate_value(self, data_type: str, item_count: int) -> float:
        """
        Estimate product value based on data type and count.

        Args:
            data_type (str): Type of data.
            item_count (int): Number of items.

        Returns:
            float: Estimated value.
        """
        base_values = {
            "auctions": 0.50,
            "tenders": 1.00,
            "businesses": 0.75,
            "jobs": 0.25,
            "osint": 2.00,
            "realestate": 1.50,
        }

        base_value = base_values.get(data_type, 0.50)
        return base_value * item_count

    def update_pricing(self, product_type: str, new_price: float) -> None:
        """
        Update pricing for a product type.

        Args:
            product_type (str): Type of product (e.g., "premium", "basic").
            new_price (float): New price to set.
        """
        # Update one_time pricing if it matches
        if product_type in self.pricing_models["one_time"]:
            self.pricing_models["one_time"][product_type] = new_price
            logger.info(f"Updated {product_type} pricing to ${new_price:.2f}")
        else:
            # Create new entry or update tiered
            if product_type in self.pricing_models["tiered"]:
                self.pricing_models["tiered"][product_type] = new_price
            else:
                # Default to one_time
                self.pricing_models["one_time"][product_type] = new_price
                logger.info(f"Created new pricing entry for {product_type}: ${new_price:.2f}")

