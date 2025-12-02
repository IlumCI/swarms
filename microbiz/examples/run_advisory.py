"""
Example: Run microbiz in advisory mode.

Board makes recommendations, but no automatic actions are taken.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from business.integration import BusinessIntegration
from loguru import logger


def main():
    """Run in advisory mode."""
    logger.info("Starting microbiz in ADVISORY mode")
    logger.info("Board will make recommendations, but no automatic actions will be taken")

    # Initialize business integration
    biz = BusinessIntegration()

    # Check autonomy mode
    autonomy_mode = biz.config.get("autonomy", {}).get("mode", "advisory")
    logger.info(f"Current autonomy mode: {autonomy_mode}")

    if autonomy_mode != "advisory":
        logger.warning(f"Autonomy mode is {autonomy_mode}, not advisory. Adjusting...")
        biz.config["autonomy"]["mode"] = "advisory"

    # Build status report
    status_report = biz.build_board_status_report()
    logger.info("Status report generated")

    # Get board recommendations
    recommendations = biz.get_board_recommendations()
    logger.info(f"Board provided {len(recommendations)} recommendations:")

    for i, rec in enumerate(recommendations, 1):
        logger.info(f"  {i}. {rec}")

    # Example: Get scaling decision (advisory only)
    worker_type = "auctions"
    performance = status_report["worker_performance"].get(worker_type, {})
    scaling_decision = biz.make_scaling_decision(worker_type, performance)
    logger.info(f"Scaling decision for {worker_type}: {scaling_decision['action']} - {scaling_decision.get('reason', '')}")

    # Example: Get pricing decision (advisory only)
    pricing_decision = biz.make_pricing_decision("premium", {"revenue": 100.0, "sales": 10})
    logger.info(f"Pricing decision: {pricing_decision['action']} - {pricing_decision.get('reason', '')}")

    logger.info("Advisory mode complete. Review recommendations and take manual action if needed.")


if __name__ == "__main__":
    main()

