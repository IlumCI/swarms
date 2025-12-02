"""
OSINT data collection worker using Swarms Agent.

This worker collects domain/server metadata and OSINT findings.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarms import Agent
from loguru import logger

try:
    from swarms_tools import scrape_and_format_sync
    SWARMS_TOOLS_AVAILABLE = True
except ImportError:
    SWARMS_TOOLS_AVAILABLE = False
    logger.warning("swarms_tools not available. Using custom tools only.")

from modules.tools import osint_tool
from modules.parser import parse_agent_output, parse_osint
from modules.summarizer import summarize
from modules.storage import save_output
from modules.memory import get_memory_system
from modules.rate_limiter import RateLimiter
from modules.validator import validate_data, filter_by_quality
from modules.adaptive_parser import AdaptiveParser
from modules.error_handler import ErrorHandler


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json."""
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def load_sources() -> Dict[str, Any]:
    """Load source URLs from sources.json."""
    sources_path = Path(__file__).parent.parent / "config" / "sources.json"
    if sources_path.exists():
        with open(sources_path, "r") as f:
            return json.load(f)
    return {}


def run_worker(global_config: Dict[str, Any] = None) -> int:
    """Run the OSINT data collection worker."""
    try:
        config = global_config or load_config()
        sources_config = load_sources()
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

        rate_limiter = RateLimiter(default_delay=2.0, respect_robots=True)
        adaptive_parser = AdaptiveParser(model_name=model_name)
        error_handler = ErrorHandler()

        osint_sources = sources_config.get("osint", {}).get("sources", [])
        if not osint_sources:
            osint_sources = [{"url": "example.com", "name": "default"}]

        tools = [osint_tool]
        if SWARMS_TOOLS_AVAILABLE:
            tools.append(scrape_and_format_sync)

        memory_system = get_memory_system(config.get("memory", {}))

        agent = Agent(
            agent_name="OSINT Collector",
            model_name=model_name,
            system_prompt=(
                "You are an OSINT (Open Source Intelligence) data collection agent. "
                "Your task is to collect domain/server metadata including server headers, "
                "tech stack fingerprints, and exposed metadata. "
                "Return the data as a JSON string with an 'osint' object containing these fields."
            ),
            tools=tools,
            max_loops=3,
            dynamic_context_window=True,
            verbose=False,
            long_term_memory=memory_system,
        )

        all_data = []
        errors = []

        for source in osint_sources:
            if not source.get("enabled", True):
                continue

            source_name = source.get("name", "unknown")
            source_url = source.get("url", "")

            try:
                rate_limiter.wait_if_needed(source_name, source_url)
                user_agent = rate_limiter.get_user_agent(source_name)
                if not rate_limiter.can_fetch(source_url, user_agent):
                    continue

                task = (
                    f"Collect OSINT data for {source_url}. "
                    "Gather server headers, tech stack information, and any exposed metadata. "
                    "Return the data as a JSON string with this structure: "
                    '{"osint": {"domain": "...", "server_headers": {...}, "tech_stack": [...], "metadata": {...}}}'
                )

                logger.info(f"Running OSINT collection agent for {source_name}...")
                output = agent.run(task)

                parsed_output = parse_agent_output(output)
                structured_data = parse_osint(parsed_output)

                validation_result = validate_data(structured_data, "osint")
                valid_data = validation_result.get("valid_items", [])
                quality_data = filter_by_quality(valid_data, "osint", min_quality=0.7)

                all_data.extend(quality_data)
                rate_limiter.reset_backoff(source_name)

            except Exception as e:
                error_result = error_handler.handle_error(e, {"source": source_name, "url": source_url})
                errors.append(error_result)
                if error_result["error"]["category"] == "rate_limit":
                    rate_limiter.handle_rate_limit_error(source_name)

        if not all_data:
            logger.warning("No valid OSINT data collected")
            return 1

        structured_data = all_data

        summary = None
        if config.get("llm", {}).get("enabled", False):
            try:
                summary_text = json.dumps(structured_data[:10], indent=2)
                summary = summarize(
                    summary_text,
                    use_llama=not config.get("llm", {}).get("prefer_swarms", True),
                    llama_path=config.get("llm", {}).get("llama_path"),
                )
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")

        output_data = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source": "osint",
            "data": structured_data,
            "statistics": {
                "total_items": len(structured_data),
                "sources_processed": len([s for s in osint_sources if s.get("enabled", True)]),
                "errors": len(errors),
            },
        }
        if summary:
            output_data["summary"] = summary

        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M")
        output_path = f"output/osint/{timestamp}.json"
        gcs_bucket = config.get("storage", {}).get("gcs_bucket")

        success = save_output(output_path, output_data, gcs_bucket)
        if success:
            logger.info(f"OSINT data saved to {output_path}")
            return 0
        else:
            logger.error("Failed to save OSINT data")
            return 1

    except Exception as e:
        logger.error(f"Error in OSINT worker: {e}")
        return 1


if __name__ == "__main__":
    exit_code = run_worker()
    sys.exit(exit_code)
