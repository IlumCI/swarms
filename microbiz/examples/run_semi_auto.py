"""
Example: Run microbiz in semi-auto mode.

Free content (GitHub, Substack) is automatically published.
Paid products (Gumroad) require manual approval.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from business.integration import BusinessIntegration
from business.operations import OperationsManager
from loguru import logger


def main():
    """Run in semi-auto mode."""
    logger.info("Starting microbiz in SEMI-AUTO mode")
    logger.info("Free content will be published automatically")
    logger.info("Paid products require manual approval")

    # Initialize business integration
    biz = BusinessIntegration()

    # Set to semi-auto mode
    biz.config["autonomy"]["mode"] = "semi_auto"
    logger.info(f"Autonomy mode set to: {biz.config['autonomy']['mode']}")

    # Example: Process collected data
    sample_data = [
        {"title": "Sample Auction Lot 1", "price": "$100", "bids": "5"},
        {"title": "Sample Auction Lot 2", "price": "$200", "bids": "10"},
    ]

    result = biz.process_collected_data(
        "auctions",
        sample_data,
        {"total_items": len(sample_data), "average_quality": 0.85},
    )

    logger.info(f"Processed {result['items_processed']} items")
    logger.info(f"Revenue generated (tracked): ${result['revenue_generated']:.2f}")

    # Check what was created
    if "github_release" in result:
        release = result["github_release"]
        logger.info(f"GitHub release: {release['status']} - {release.get('github_url', 'N/A')}")

    if "substack_post" in result:
        post = result["substack_post"]
        logger.info(f"Substack post: {post['status']} - {post.get('substack_url', 'N/A')}")

    if "gumroad_listing" in result:
        listing = result["gumroad_listing"]
        logger.info(f"Gumroad listing: {listing['status']} (requires manual approval for publishing)")

    # Get board recommendations
    recommendations = biz.get_board_recommendations()
    logger.info(f"Board recommendations: {len(recommendations)} items")

    logger.info("Semi-auto mode complete. Review paid products before publishing.")


if __name__ == "__main__":
    main()

