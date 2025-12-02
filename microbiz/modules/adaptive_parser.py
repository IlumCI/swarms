"""
LLM-based adaptive parsing module.

Provides intelligent parsing that adapts to different site structures using LLM agents.
"""

import os
import json
from typing import Any, Dict, List, Optional
from loguru import logger

try:
    from swarms import Agent
    from swarms_tools import scrape_and_format_sync
    SWARMS_AVAILABLE = True
except ImportError:
    SWARMS_AVAILABLE = False
    logger.warning("Swarms not available. Adaptive parsing will be limited.")


class AdaptiveParser:
    """
    Adaptive parser that uses LLM to extract structured data from various site structures.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize adaptive parser.

        Args:
            model_name (Optional[str]): Model name for LLM agent. Uses env var if None.
        """
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        self.parser_cache: Dict[str, Dict[str, Any]] = {}

    def parse_with_llm(
        self,
        html_content: str,
        target_schema: str,
        url: str,
        extraction_instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse HTML content using LLM agent with adaptive extraction.

        Args:
            html_content (str): HTML content to parse.
            target_schema (str): Target schema type (auctions, tenders, etc.).
            url (str): Source URL for context.
            extraction_instructions (Optional[str]): Additional extraction instructions.

        Returns:
            Dict[str, Any]: Parsed data structure.
        """
        if not SWARMS_AVAILABLE:
            logger.error("Swarms not available for adaptive parsing")
            return {"error": "Swarms not available", "data": []}

        try:
            # Check cache for this URL pattern
            cache_key = f"{url}_{target_schema}"
            if cache_key in self.parser_cache:
                logger.debug(f"Using cached parser pattern for {url}")

            # Create extraction prompt
            schema_descriptions = {
                "auctions": "Extract auction lots with: title, price, bids, end_time",
                "tenders": "Extract tenders with: deadline, authority, budget, category",
                "businesses": "Extract business changes with: company_name, status, date",
                "jobs": "Extract job postings with: title, company, tech_stack, salary, location",
                "osint": "Extract OSINT data with: domain, server_headers, tech_stack, metadata",
                "realestate": "Extract listings with: title, price, area, district, price_per_m2",
            }

            schema_desc = schema_descriptions.get(target_schema, "Extract structured data")

            prompt = f"""
            Analyze the following HTML content and extract structured data according to this schema: {schema_desc}
            
            Source URL: {url}
            
            {extraction_instructions or ""}
            
            Return the data as a JSON object with a key matching the schema type (e.g., "auctions", "tenders") containing an array of objects.
            
            HTML Content:
            {html_content[:5000]}  # Limit content length
            """

            # Create parsing agent
            agent = Agent(
                agent_name="Adaptive Parser",
                model_name=self.model_name,
                system_prompt=(
                    "You are an expert web scraping parser. "
                    "Your task is to extract structured data from HTML content. "
                    "Analyze the HTML structure and extract the requested information. "
                    "Return valid JSON only."
                ),
                tools=[scrape_and_format_sync] if SWARMS_AVAILABLE else [],
                max_loops=2,
                verbose=False,
            )

            # Run parsing
            logger.info(f"Running adaptive parser for {url}")
            result = agent.run(prompt)

            # Try to extract JSON from result
            try:
                # Find JSON in response
                import re

                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    parsed_data = json.loads(json_match.group(0))
                    # Cache successful pattern
                    self.parser_cache[cache_key] = {"pattern": "llm", "success": True}
                    return parsed_data
                else:
                    # Try parsing entire result
                    parsed_data = json.loads(result)
                    return parsed_data
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from LLM response: {result[:200]}")
                return {"error": "JSON parsing failed", "raw_response": result[:500]}

        except Exception as e:
            logger.error(f"Error in adaptive parsing: {e}")
            return {"error": str(e), "data": []}

    def get_parser_for_url(self, url: str, target_schema: str) -> str:
        """
        Determine best parser type for a URL.

        Args:
            url (str): Source URL.
            target_schema (str): Target schema type.

        Returns:
            str: Parser type ("adaptive", "structured", or "manual").
        """
        # Check cache
        cache_key = f"{url}_{target_schema}"
        if cache_key in self.parser_cache:
            cached = self.parser_cache[cache_key]
            if cached.get("success"):
                return cached.get("pattern", "adaptive")

        # Default to adaptive for unknown sites
        return "adaptive"

