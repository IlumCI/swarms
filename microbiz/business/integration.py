"""
Business integration module.

Ties together Board of Directors, revenue, operations, finance, and autonomy.
"""

import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger

from business.board import create_board_swarm, load_business_config
from business.finance import FinancialManager
from business.revenue import RevenueGenerator
from business.operations import OperationsManager
from business.autonomy import AutonomyEngine
from modules.metrics import build_all_worker_snapshots


class BusinessIntegration:
    """
    Main business integration that coordinates all business modules.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize business integration.

        Args:
            config_path (Optional[str]): Path to business config file.
        """
        self.config = load_business_config(config_path)

        # Check kill switch
        if self.config.get("autonomy", {}).get("kill_switch", False):
            logger.warning("KILL SWITCH ACTIVATED - Forcing advisory mode")
            if "autonomy" not in self.config:
                self.config["autonomy"] = {}
            self.config["autonomy"]["mode"] = "advisory"

        # Check env var kill switch
        if os.getenv("MICROBIZ_KILL_SWITCH", "").lower() in ["true", "1", "yes"]:
            logger.warning("ENV KILL SWITCH ACTIVATED - Forcing advisory mode")
            if "autonomy" not in self.config:
                self.config["autonomy"] = {}
            self.config["autonomy"]["mode"] = "advisory"

        self.finance_manager = FinancialManager()
        self.revenue_generator = RevenueGenerator(self.finance_manager)
        self.operations_manager = OperationsManager()
        self.autonomy_engine = AutonomyEngine(self.finance_manager)
        self.board_swarm = None

        # Initialize Board if enabled
        if self.config.get("board", {}).get("enabled", False):
            try:
                self.board_swarm = create_board_swarm(config=self.config)
                if self.board_swarm:
                    logger.info("Business integration: Board of Directors enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Board: {e}")

    def process_collected_data(
        self, data_type: str, data: List[Dict[str, Any]], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process collected data through business pipeline.

        Args:
            data_type (str): Type of data collected.
            data (List[Dict[str, Any]]): Collected data.
            metadata (Dict[str, Any]): Data metadata.

        Returns:
            Dict[str, Any]: Processing results.
        """
        autonomy_mode = self.config.get("autonomy", {}).get("mode", "advisory")
        results = {
            "data_type": data_type,
            "items_processed": len(data),
            "revenue_generated": 0.0,
            "products_created": [],
        }

        # Create product package
        product = self.revenue_generator.create_product_package(data_type, data)
        results["products_created"].append(product)

        # Generate revenue (tracked, not charged)
        if self.config.get("revenue", {}).get("strategies", {}).get("one_time", {}).get("enabled"):
            revenue = self.revenue_generator.generate_one_time_revenue(
                product["product_id"], "premium", 1
            )
            results["revenue_generated"] = revenue

        # Create operations (gated by autonomy mode)
        ops_config = self.config.get("operations", {})

        if ops_config.get("gumroad", {}).get("enabled"):
            listing = self.operations_manager.create_gumroad_listing(
                product,
                product["estimated_value"],
                f"Dataset containing {len(data)} items",
                autonomy_mode=autonomy_mode,
            )
            results["gumroad_listing"] = listing

        if ops_config.get("substack", {}).get("enabled") and autonomy_mode in ["semi_auto", "full_auto"]:
            post = self.operations_manager.create_substack_post(
                f"Weekly {data_type.title()} Data Digest",
                f"Summary of {len(data)} items collected this week.",
                metadata,
                autonomy_mode=autonomy_mode,
                publish=(autonomy_mode == "full_auto"),
            )
            results["substack_post"] = post

        if ops_config.get("github", {}).get("enabled") and autonomy_mode in ["semi_auto", "full_auto"]:
            repo = ops_config.get("github", {}).get("repo")
            if repo:
                release = self.operations_manager.create_github_release(
                    repo,
                    f"{data_type}-{datetime.utcnow().strftime('%Y%m%d')}",
                    None,  # Would need actual file path
                    f"Free sample: {data_type} dataset with {len(data)} items",
                    autonomy_mode=autonomy_mode,
                )
                results["github_release"] = release

        # Track inventory
        self.operations_manager.track_inventory(product["product_id"], 1, data_type)

        # Detect market opportunities
        opportunity = self.autonomy_engine.detect_market_opportunity(
            {"data": data, "average_quality": metadata.get("average_quality", 0.8)},
            data_type,
        )
        if opportunity:
            results["market_opportunity"] = opportunity

        return results

    def build_board_status_report(self) -> Dict[str, Any]:
        """
        Build comprehensive status report for Board of Directors.

        Returns:
            Dict[str, Any]: Status report with metrics, finance, and opportunities.
        """
        # Worker performance metrics
        worker_snapshots = build_all_worker_snapshots()

        # Financial metrics
        financial_report = self.finance_manager.get_financial_report()

        # Market opportunities
        opportunities = []
        for worker_type, snapshot in worker_snapshots.items():
            if snapshot.get("items_collected", 0) > 100:
                opportunities.append({
                    "worker_type": worker_type,
                    "items": snapshot["items_collected"],
                    "quality": snapshot["data_quality"],
                })

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "worker_performance": worker_snapshots,
            "financial_status": financial_report,
            "market_opportunities": opportunities,
            "autonomy_mode": self.config.get("autonomy", {}).get("mode", "advisory"),
        }

    def get_board_recommendations(self) -> List[str]:
        """
        Get strategic recommendations from Board of Directors.

        Returns:
            List[str]: List of recommendations.
        """
        if not self.board_swarm:
            return self.autonomy_engine.generate_strategic_recommendations()

        try:
            status_report = self.build_board_status_report()

            task = (
                f"Review the following business status report and provide strategic recommendations:\n\n"
                f"Worker Performance:\n{json.dumps(status_report['worker_performance'], indent=2)}\n\n"
                f"Financial Status:\n{json.dumps(status_report['financial_status'], indent=2)}\n\n"
                f"Market Opportunities:\n{json.dumps(status_report['market_opportunities'], indent=2)}\n\n"
                "Provide actionable recommendations for: pricing, scaling, cost optimization, and strategic pivoting."
            )

            result = self.board_swarm.run(task)
            return result if isinstance(result, list) else [str(result)]

        except Exception as e:
            logger.error(f"Error getting board recommendations: {e}")
            return self.autonomy_engine.generate_strategic_recommendations()

    def make_scaling_decision(
        self, worker_type: str, performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make scaling decision through Board or autonomy engine.

        Returns machine-actionable plan.

        Args:
            worker_type (str): Type of worker.
            performance (Dict[str, Any]): Performance metrics.

        Returns:
            Dict[str, Any]: Scaling decision with actionable plan.
        """
        if self.board_swarm:
            task = (
                f"Evaluate scaling decision for {worker_type} worker:\n"
                f"Performance: {json.dumps(performance, indent=2)}\n\n"
                "Respond with JSON: {\"action\": \"scale_up\"|\"scale_down\"|\"maintain\", "
                '"reason": "...", "new_frequency": "HH:MM" (if changing schedule)}'
            )

            try:
                result = self.board_swarm.run(task)
                # Try to parse JSON from result
                import re
                json_match = re.search(r'\{.*\}', str(result), re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group(0))
                else:
                    decision = {"action": "maintain", "reason": str(result)}

                return {
                    "worker_type": worker_type,
                    "action": decision.get("action", "maintain"),
                    "reason": decision.get("reason", ""),
                    "new_frequency": decision.get("new_frequency"),
                    "source": "board",
                }
            except Exception as e:
                logger.error(f"Error getting board scaling decision: {e}")

        # Fallback to autonomy engine
        autonomy_decision = self.autonomy_engine.evaluate_scaling_decision(worker_type, performance)
        return {
            "worker_type": worker_type,
            "action": autonomy_decision.get("recommendation", "maintain"),
            "reason": " ".join(autonomy_decision.get("reasoning", [])),
            "new_frequency": None,  # Autonomy engine doesn't specify frequency
            "source": "autonomy",
        }

    def make_pricing_decision(self, product_type: str, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make pricing decision through Board or autonomy engine.

        Returns machine-actionable pricing changes.

        Args:
            product_type (str): Type of product.
            current_performance (Dict[str, Any]): Current performance metrics.

        Returns:
            Dict[str, Any]: Pricing decision with actionable changes.
        """
        if self.board_swarm:
            task = (
                f"Review pricing for {product_type} products:\n"
                f"Performance: {json.dumps(current_performance, indent=2)}\n\n"
                "Respond with JSON: {\"action\": \"increase\"|\"decrease\"|\"maintain\", "
                '"new_price": <number> (if changing), "reason": "..."}'
            )

            try:
                result = self.board_swarm.run(task)
                import re
                json_match = re.search(r'\{.*\}', str(result), re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group(0))
                else:
                    decision = {"action": "maintain", "reason": str(result)}

                # Apply pricing change if autonomy mode allows
                autonomy_mode = self.config.get("autonomy", {}).get("mode", "advisory")
                if autonomy_mode in ["semi_auto", "full_auto"] and decision.get("action") != "maintain":
                    new_price = decision.get("new_price")
                    if new_price:
                        # Update pricing in revenue generator
                        self.revenue_generator.update_pricing(product_type, new_price)
                        logger.info(f"Pricing updated for {product_type}: ${new_price:.2f}")

                return {
                    "product_type": product_type,
                    "action": decision.get("action", "maintain"),
                    "new_price": decision.get("new_price"),
                    "reason": decision.get("reason", ""),
                    "source": "board",
                    "applied": autonomy_mode in ["semi_auto", "full_auto"],
                }
            except Exception as e:
                logger.error(f"Error getting board pricing decision: {e}")

        autonomy_decision = self.autonomy_engine.evaluate_pricing_strategy(product_type, current_performance)
        return {
            "product_type": product_type,
            "action": autonomy_decision.get("recommendation", "maintain"),
            "new_price": None,
            "reason": " ".join(autonomy_decision.get("reasoning", [])),
            "source": "autonomy",
            "applied": False,  # Autonomy engine doesn't auto-apply
        }

