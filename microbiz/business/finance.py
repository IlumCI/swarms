"""
Financial management module (Treasurer's domain).

Handles revenue tracking, cost analysis, profit/loss calculations, and budget allocation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from loguru import logger


class FinancialManager:
    """
    Financial manager for tracking revenue, costs, and profitability.
    """

    def __init__(self, data_dir: str = "output/finance"):
        """
        Initialize financial manager.

        Args:
            data_dir (str): Directory for storing financial data.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.revenue_file = self.data_dir / "revenue.json"
        self.costs_file = self.data_dir / "costs.json"
        self.pl_file = self.data_dir / "profit_loss.json"

    def track_revenue(
        self,
        source: str,
        amount: float,
        currency: str = "USD",
        product_type: Optional[str] = None,
    ) -> None:
        """
        Track revenue from a source.

        Args:
            source (str): Revenue source (gumroad, substack, api, etc.).
            amount (float): Revenue amount.
            currency (str): Currency code.
            product_type (Optional[str]): Type of product/service sold.
        """
        revenue_data = self._load_json(self.revenue_file, [])
        revenue_data.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "source": source,
                "amount": amount,
                "currency": currency,
                "product_type": product_type,
            }
        )
        self._save_json(self.revenue_file, revenue_data)
        logger.info(f"Revenue tracked: ${amount:.2f} from {source}")

    def track_cost(
        self,
        category: str,
        amount: float,
        currency: str = "USD",
        description: Optional[str] = None,
    ) -> None:
        """
        Track cost/expense.

        Args:
            category (str): Cost category (api, compute, storage, etc.).
            amount (float): Cost amount.
            currency (str): Currency code.
            description (Optional[str]): Cost description.
        """
        costs_data = self._load_json(self.costs_file, [])
        costs_data.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "category": category,
                "amount": amount,
                "currency": currency,
                "description": description,
            }
        )
        self._save_json(self.costs_file, costs_data)
        logger.info(f"Cost tracked: ${amount:.2f} for {category}")

    def calculate_profit_loss(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate profit/loss for a period.

        Args:
            start_date (Optional[datetime]): Start date for calculation.
            end_date (Optional[datetime]): End date for calculation.

        Returns:
            Dict[str, Any]: P&L calculation results.
        """
        revenue_data = self._load_json(self.revenue_file, [])
        costs_data = self._load_json(self.costs_file, [])

        # Filter by date range
        if start_date:
            revenue_data = [
                r for r in revenue_data if datetime.fromisoformat(r["timestamp"]) >= start_date
            ]
            costs_data = [
                c for c in costs_data if datetime.fromisoformat(c["timestamp"]) >= start_date
            ]

        if end_date:
            revenue_data = [
                r for r in revenue_data if datetime.fromisoformat(r["timestamp"]) <= end_date
            ]
            costs_data = [
                c for c in costs_data if datetime.fromisoformat(c["timestamp"]) <= end_date
            ]

        total_revenue = sum(r["amount"] for r in revenue_data)
        total_costs = sum(c["amount"] for c in costs_data)
        profit = total_revenue - total_costs
        profit_margin = (profit / total_revenue * 100) if total_revenue > 0 else 0

        pl_data = {
            "period_start": start_date.isoformat() if start_date else None,
            "period_end": end_date.isoformat() if end_date else None,
            "total_revenue": total_revenue,
            "total_costs": total_costs,
            "profit": profit,
            "profit_margin_percent": profit_margin,
            "revenue_by_source": self._group_by_field(revenue_data, "source"),
            "costs_by_category": self._group_by_field(costs_data, "category"),
        }

        self._save_json(self.pl_file, pl_data)
        return pl_data

    def get_financial_report(self) -> Dict[str, Any]:
        """
        Get comprehensive financial report for board.

        Returns:
            Dict[str, Any]: Financial report.
        """
        # Last 30 days
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)

        pl_30d = self.calculate_profit_loss(start_date, end_date)

        # All time
        pl_all = self.calculate_profit_loss()

        return {
            "report_date": datetime.utcnow().isoformat(),
            "last_30_days": pl_30d,
            "all_time": pl_all,
            "recommendations": self._generate_recommendations(pl_30d),
        }

    def _generate_recommendations(self, pl_data: Dict[str, Any]) -> List[str]:
        """
        Generate financial recommendations for board.

        Args:
            pl_data (Dict[str, Any]): Profit/loss data.

        Returns:
            List[str]: List of recommendations.
        """
        recommendations = []

        if pl_data["profit"] < 0:
            recommendations.append("Business is operating at a loss. Review costs and revenue streams.")
        elif pl_data["profit_margin_percent"] < 10:
            recommendations.append("Profit margin is low. Consider cost optimization or price increases.")

        if pl_data["total_costs"] > pl_data["total_revenue"] * 0.8:
            recommendations.append("Costs are high relative to revenue. Review cost categories.")

        return recommendations

    def _load_json(self, file_path: Path, default: Any) -> Any:
        """Load JSON file."""
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        return default

    def _save_json(self, file_path: Path, data: Any) -> None:
        """Save JSON file."""
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving {file_path}: {e}")

    def _group_by_field(self, data: List[Dict[str, Any]], field: str) -> Dict[str, float]:
        """Group data by field and sum amounts."""
        grouped = {}
        for item in data:
            key = item.get(field, "unknown")
            grouped[key] = grouped.get(key, 0) + item.get("amount", 0)
        return grouped

