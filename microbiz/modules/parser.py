"""
Parser module for extracting and structuring data from agent outputs.

This module handles JSON parsing and task-specific data extraction.
"""

import json
import re
from typing import Any, Dict, List, Optional
from loguru import logger


def parse_agent_output(output: str) -> Dict[str, Any]:
    """
    Extracts JSON from agent response string.

    Args:
        output (str): Agent output string that may contain JSON.

    Returns:
        Dict[str, Any]: Parsed JSON data, or empty dict if parsing fails.
    """
    try:
        # Try to find JSON in the output
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)

        # If no JSON found, try parsing the entire output
        return json.loads(output)

    except json.JSONDecodeError:
        logger.warning(f"Could not parse JSON from output: {output[:100]}...")
        # Try to extract structured data from text
        return {"raw_output": output}


def parse_auctions(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Structures auction data from parsed JSON.

    Args:
        json_data (Dict[str, Any]): Parsed JSON data containing auction information.

    Returns:
        List[Dict[str, Any]]: Structured list of auction entries.
    """
    try:
        if "auctions" in json_data:
            return json_data["auctions"]
        elif isinstance(json_data, list):
            return json_data
        else:
            return [json_data]
    except Exception as e:
        logger.error(f"Error parsing auction data: {e}")
        return []


def parse_tenders(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Structures tender data from parsed JSON.

    Args:
        json_data (Dict[str, Any]): Parsed JSON data containing tender information.

    Returns:
        List[Dict[str, Any]]: Structured list of tender entries.
    """
    try:
        if "tenders" in json_data:
            return json_data["tenders"]
        elif isinstance(json_data, list):
            return json_data
        else:
            return [json_data]
    except Exception as e:
        logger.error(f"Error parsing tender data: {e}")
        return []


def parse_businesses(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Structures business registry data from parsed JSON.

    Args:
        json_data (Dict[str, Any]): Parsed JSON data containing business registry information.

    Returns:
        List[Dict[str, Any]]: Structured list of business registry changes.
    """
    try:
        if "changes" in json_data:
            return json_data["changes"]
        elif isinstance(json_data, list):
            return json_data
        else:
            return [json_data]
    except Exception as e:
        logger.error(f"Error parsing business registry data: {e}")
        return []


def parse_jobs(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Structures job posting data from parsed JSON.

    Args:
        json_data (Dict[str, Any]): Parsed JSON data containing job posting information.

    Returns:
        List[Dict[str, Any]]: Structured list of job postings.
    """
    try:
        if "jobs" in json_data:
            return json_data["jobs"]
        elif isinstance(json_data, list):
            return json_data
        else:
            return [json_data]
    except Exception as e:
        logger.error(f"Error parsing job data: {e}")
        return []


def parse_osint(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Structures OSINT findings from parsed JSON.

    Args:
        json_data (Dict[str, Any]): Parsed JSON data containing OSINT information.

    Returns:
        Dict[str, Any]: Structured OSINT findings.
    """
    try:
        if "osint" in json_data:
            return json_data["osint"]
        else:
            return json_data
    except Exception as e:
        logger.error(f"Error parsing OSINT data: {e}")
        return {}


def parse_realestate(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Structures real estate data from parsed JSON and computes price per m².

    Args:
        json_data (Dict[str, Any]): Parsed JSON data containing real estate information.

    Returns:
        List[Dict[str, Any]]: Structured list of real estate listings with computed metrics.
    """
    try:
        listings = []
        if "listings" in json_data:
            listings = json_data["listings"]
        elif isinstance(json_data, list):
            listings = json_data
        else:
            listings = [json_data]

        # Compute price per m² for each listing
        for listing in listings:
            try:
                price_str = str(listing.get("price", "0")).replace("$", "").replace(",", "")
                area_str = str(listing.get("area", "0")).replace(" m²", "").replace(",", "")

                price = float(re.search(r'\d+\.?\d*', price_str).group(0)) if re.search(r'\d+\.?\d*', price_str) else 0
                area = float(re.search(r'\d+\.?\d*', area_str).group(0)) if re.search(r'\d+\.?\d*', area_str) else 0

                if area > 0:
                    price_per_m2 = price / area
                    listing["price_per_m2"] = f"${price_per_m2:,.2f}/m²"
                else:
                    listing["price_per_m2"] = "N/A"
            except Exception as e:
                logger.warning(f"Could not compute price per m² for listing: {e}")
                listing["price_per_m2"] = "N/A"

        return listings

    except Exception as e:
        logger.error(f"Error parsing real estate data: {e}")
        return []

