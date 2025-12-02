"""
Auction data collection worker using Swarms Agent.

This worker scrapes auction sites and extracts structured auction data.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarms import Agent
from loguru import logger

# Try to import swarms_tools
try:
    from swarms_tools import scrape_and_format_sync
    SWARMS_TOOLS_AVAILABLE = True
except ImportError:
    SWARMS_TOOLS_AVAILABLE = False
    logger.warning("swarms_tools not available. Using custom tools only.")

from modules.tools import auction_scraper_tool
from modules.parser import parse_agent_output, parse_auctions
from modules.summarizer import summarize
from modules.storage import save_output
from modules.memory import get_memory_system
from modules.rate_limiter import RateLimiter
from modules.validator import validate_data, filter_by_quality
from modules.adaptive_parser import AdaptiveParser
from modules.error_handler import ErrorHandler


def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.json.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def load_sources() -> Dict[str, Any]:
    """
    Load source URLs from sources.json.

    Returns:
        Dict[str, Any]: Sources configuration.
    """
    sources_path = Path(__file__).parent.parent / "config" / "sources.json"
    if sources_path.exists():
        with open(sources_path, "r") as f:
            return json.load(f)
    return {}


def run_worker() -> int:
    """
    Run the auction data collection worker with production features.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    try:
        config = load_config()
        sources_config = load_sources()
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

        # Initialize production modules
        rate_limiter = RateLimiter(default_delay=2.0, respect_robots=True)
        validator = None  # Will use validate_data function
        adaptive_parser = AdaptiveParser(model_name=model_name)
        error_handler = ErrorHandler()

        # Get sources for auctions
        auction_sources = sources_config.get("auctions", {}).get("sources", [])
        if not auction_sources:
            logger.warning("No sources configured for auctions. Using default.")
            auction_sources = [{"url": "https://example-auctions.com/latest", "name": "default"}]

        # Prepare tools
        tools = [auction_scraper_tool]
        if SWARMS_TOOLS_AVAILABLE:
            tools.append(scrape_and_format_sync)

        # Initialize memory system if enabled
        memory_system = get_memory_system(config.get("memory", {}))

        # Create agent
        agent = Agent(
            agent_name="Auction Collector",
            model_name=model_name,
            system_prompt=(
                "You are an auction data collection agent. "
                "Your task is to scrape auction websites and extract structured data including: "
                "title, price, number of bids, and end time. "
                "Return the data as a JSON string with an 'auctions' array containing objects with these fields."
            ),
            tools=tools,
            max_loops=3,
            dynamic_context_window=True,
            verbose=False,
            long_term_memory=memory_system,
        )

        all_data = []
        errors = []

        # Process each source
        for source in auction_sources:
            if not source.get("enabled", True):
                continue

            source_name = source.get("name", "unknown")
            source_url = source.get("url", "")

            try:
                # Rate limiting
                rate_limiter.wait_if_needed(source_name, source_url)

                # Check robots.txt
                user_agent = rate_limiter.get_user_agent(source_name)
                if not rate_limiter.can_fetch(source_url, user_agent):
                    logger.warning(f"robots.txt disallows fetching {source_url}")
                    continue

                # Construct task with real URL
                task = (
                    f"Scrape auction data from {source_url}. "
                    "Extract all auction lots with their title, price, number of bids, and end time. "
                    "Return the data as a JSON string with this structure: "
                    '{"auctions": [{"title": "...", "price": "...", "bids": "...", "end_time": "..."}]}'
                )

                logger.info(f"Running auction collection agent for {source_name}...")
                output = agent.run(task)

                # Parse agent output
                parsed_output = parse_agent_output(output)
                structured_data = parse_auctions(parsed_output)

                # Validate data
                validation_result = validate_data(structured_data, "auctions")
                valid_data = validation_result.get("valid_items", [])

                # Filter by quality
                quality_data = filter_by_quality(valid_data, "auctions", min_quality=0.7)

                all_data.extend(quality_data)

                # Reset rate limit backoff on success
                rate_limiter.reset_backoff(source_name)

                logger.info(
                    f"Collected {len(quality_data)} valid items from {source_name} "
                    f"(quality: {validation_result.get('average_quality', 0):.2f})"
                )

            except Exception as e:
                error_result = error_handler.handle_error(e, {"source": source_name, "url": source_url})
                errors.append(error_result)

                # Handle rate limit errors
                if error_result["error"]["category"] == "rate_limit":
                    rate_limiter.handle_rate_limit_error(source_name)

        if not all_data:
            logger.warning("No valid auction data collected")
            if errors:
                logger.error(f"Errors occurred: {len(errors)}")
            return 1

        # Optional summarization
        summary = None
        if config.get("llm", {}).get("enabled", False):
            try:
                summary_text = json.dumps(all_data[:10], indent=2)  # Summarize sample
                summary = summarize(
                    summary_text,
                    use_llama=not config.get("llm", {}).get("prefer_swarms", True),
                    llama_path=config.get("llm", {}).get("llama_path"),
                )
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")

        # Prepare output data
        output_data = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source": "auctions",
            "data": all_data,
            "statistics": {
                "total_items": len(all_data),
                "sources_processed": len([s for s in auction_sources if s.get("enabled", True)]),
                "errors": len(errors),
            },
        }
        if summary:
            output_data["summary"] = summary

        # Save output
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
        output_path = f"output/auctions/{timestamp}.json"
        gcs_bucket = config.get("storage", {}).get("gcs_bucket")

        success = save_output(output_path, output_data, gcs_bucket)
        if success:
            logger.info(f"Auction data saved to {output_path} ({len(all_data)} items)")
            return 0
        else:
            logger.error("Failed to save auction data")
            return 1

    except Exception as e:
        logger.error(f"Error in auction worker: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = run_worker()
    sys.exit(exit_code)

