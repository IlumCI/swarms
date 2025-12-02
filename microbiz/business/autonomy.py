"""
Autonomous behavior engine (Board-driven).

Handles self-optimization, automatic scaling, market opportunity detection, and strategic pivoting.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

from business.finance import FinancialManager


class AutonomyEngine:
    """
    Autonomous behavior engine that makes decisions based on business metrics.
    """

    def __init__(self, finance_manager: Optional[FinancialManager] = None):
        """
        Initialize autonomy engine.

        Args:
            finance_manager (Optional[FinancialManager]): Financial manager for metrics.
        """
        self.finance_manager = finance_manager or FinancialManager()
        self.decision_history: List[Dict[str, Any]] = []

    def evaluate_scaling_decision(
        self, worker_type: str, current_performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate whether to scale a worker up or down.

        Args:
            worker_type (str): Type of worker (auctions, tenders, etc.).
            current_performance (Dict[str, Any]): Current performance metrics.

        Returns:
            Dict[str, Any]: Scaling decision.
        """
        # Get financial metrics
        pl_data = self.finance_manager.calculate_profit_loss()

        # Decision factors
        is_profitable = pl_data.get("profit", 0) > 0
        success_rate = current_performance.get("success_rate", 0)
        data_quality = current_performance.get("data_quality", 0)

        decision = {
            "worker_type": worker_type,
            "timestamp": datetime.utcnow().isoformat(),
            "recommendation": "maintain",
            "reasoning": [],
        }

        # Scale up if profitable and high quality
        if is_profitable and success_rate > 0.9 and data_quality > 0.8:
            decision["recommendation"] = "scale_up"
            decision["reasoning"].append("Profitable with high success rate and quality")
        # Scale down if unprofitable or low quality
        elif not is_profitable or success_rate < 0.5 or data_quality < 0.5:
            decision["recommendation"] = "scale_down"
            decision["reasoning"].append("Unprofitable or low performance metrics")
        else:
            decision["reasoning"].append("Performance metrics within acceptable range")

        self.decision_history.append(decision)
        logger.info(f"Scaling decision for {worker_type}: {decision['recommendation']}")
        return decision

    def detect_market_opportunity(
        self, collected_data: Dict[str, Any], data_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Detect market opportunities from collected data.

        Args:
            collected_data (Dict[str, Any]): Recently collected data.
            data_type (str): Type of data collected.

        Returns:
            Optional[Dict[str, Any]]: Market opportunity if detected.
        """
        # Simple opportunity detection based on data volume and trends
        item_count = len(collected_data.get("data", []))
        quality = collected_data.get("average_quality", 0)

        if item_count > 100 and quality > 0.8:
            opportunity = {
                "data_type": data_type,
                "item_count": item_count,
                "quality": quality,
                "estimated_value": item_count * 0.50,  # Rough estimate
                "detected_at": datetime.utcnow().isoformat(),
                "recommendation": "Create product package",
            }

            logger.info(f"Market opportunity detected: {data_type} with {item_count} high-quality items")
            return opportunity

        return None

    def generate_strategic_recommendations(self) -> List[str]:
        """
        Generate strategic recommendations for the board.

        Returns:
            List[str]: List of strategic recommendations.
        """
        recommendations = []
        pl_data = self.finance_manager.calculate_profit_loss()

        # Financial recommendations
        if pl_data.get("profit", 0) < 0:
            recommendations.append("Business is unprofitable. Review costs and consider price increases.")

        profit_margin = pl_data.get("profit_margin_percent", 0)
        if profit_margin < 10:
            recommendations.append("Low profit margin. Optimize costs or increase prices.")

        # Operational recommendations
        if len(self.decision_history) > 0:
            recent_decisions = self.decision_history[-5:]
            scale_down_count = sum(1 for d in recent_decisions if d.get("recommendation") == "scale_down")
            if scale_down_count > 2:
                recommendations.append("Multiple workers underperforming. Review data sources and parsing strategies.")

        return recommendations

