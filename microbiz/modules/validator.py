"""
Data validation and quality scoring module.

Provides schema validation, quality scoring, duplicate detection, and data enrichment.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, ValidationError, Field
from loguru import logger


class AuctionItem(BaseModel):
    """Schema for auction items."""

    title: str = Field(..., description="Auction lot title")
    price: Optional[str] = Field(None, description="Price or price range")
    bids: Optional[str] = Field(None, description="Number of bids")
    end_time: Optional[str] = Field(None, description="Auction end time")


class TenderItem(BaseModel):
    """Schema for tender items."""

    deadline: str = Field(..., description="Tender deadline")
    authority: Optional[str] = Field(None, description="Contracting authority")
    budget: Optional[str] = Field(None, description="Budget amount")
    category: Optional[str] = Field(None, description="Tender category")


class BusinessChange(BaseModel):
    """Schema for business registry changes."""

    company_name: str = Field(..., description="Company name")
    status: str = Field(..., description="Registration status")
    date: Optional[str] = Field(None, description="Change date")


class JobPosting(BaseModel):
    """Schema for job postings."""

    title: str = Field(..., description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    tech_stack: Optional[str] = Field(None, description="Required tech stack")
    salary: Optional[str] = Field(None, description="Salary range")
    location: Optional[str] = Field(None, description="Job location")


class RealEstateListing(BaseModel):
    """Schema for real estate listings."""

    title: str = Field(..., description="Property title")
    price: Optional[str] = Field(None, description="Property price")
    area: Optional[str] = Field(None, description="Area in square meters")
    district: Optional[str] = Field(None, description="District/location")
    price_per_m2: Optional[str] = Field(None, description="Price per square meter")


# Schema mapping
SCHEMAS = {
    "auctions": AuctionItem,
    "tenders": TenderItem,
    "businesses": BusinessChange,
    "jobs": JobPosting,
    "realestate": RealEstateListing,
}


def calculate_content_hash(data: Dict[str, Any]) -> str:
    """
    Calculate content hash for duplicate detection.

    Args:
        data (Dict[str, Any]): Data dictionary.

    Returns:
        str: SHA256 hash of the data.
    """
    # Normalize data for hashing
    normalized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode()).hexdigest()


def validate_item(item: Dict[str, Any], schema_type: str) -> tuple[bool, Optional[str]]:
    """
    Validate a single item against its schema.

    Args:
        item (Dict[str, Any]): Item to validate.
        schema_type (str): Type of schema to use.

    Returns:
        tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if schema_type not in SCHEMAS:
        return True, None  # No schema defined, assume valid

    try:
        SCHEMAS[schema_type](**item)
        return True, None
    except ValidationError as e:
        error_msg = "; ".join([f"{err['loc']}: {err['msg']}" for err in e.errors()])
        return False, error_msg


def calculate_quality_score(item: Dict[str, Any], schema_type: str) -> float:
    """
    Calculate quality score for an item (0.0 to 1.0).

    Args:
        item (Dict[str, Any]): Item to score.
        schema_type (str): Type of schema.

    Returns:
        float: Quality score between 0.0 and 1.0.
    """
    if schema_type not in SCHEMAS:
        return 1.0  # No schema, assume perfect

    schema = SCHEMAS[schema_type]
    fields = schema.model_fields

    # Count required vs optional fields
    required_fields = [name for name, field in fields.items() if field.is_required()]
    optional_fields = [name for name, field in fields.items() if not field.is_required()]

    # Score based on completeness
    required_present = sum(1 for field in required_fields if field in item and item[field] is not None)
    optional_present = sum(1 for field in optional_fields if field in item and item[field] is not None)

    required_score = required_present / len(required_fields) if required_fields else 1.0
    optional_score = optional_present / len(optional_fields) if optional_fields else 0.5

    # Weighted score (required fields more important)
    quality = (required_score * 0.7) + (optional_score * 0.3)

    return quality


def validate_data(data: List[Dict[str, Any]], schema_type: str) -> Dict[str, Any]:
    """
    Validate a list of items and return validation results.

    Args:
        data (List[Dict[str, Any]]): List of items to validate.
        schema_type (str): Type of schema to use.

    Returns:
        Dict[str, Any]: Validation results with valid items, errors, and statistics.
    """
    valid_items = []
    invalid_items = []
    quality_scores = []

    for item in data:
        is_valid, error = validate_item(item, schema_type)
        quality = calculate_quality_score(item, schema_type)

        if is_valid:
            valid_items.append(item)
            quality_scores.append(quality)
        else:
            invalid_items.append({"item": item, "error": error})

    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return {
        "valid_items": valid_items,
        "invalid_items": invalid_items,
        "total": len(data),
        "valid_count": len(valid_items),
        "invalid_count": len(invalid_items),
        "average_quality": avg_quality,
        "validation_rate": len(valid_items) / len(data) if data else 0.0,
    }


def detect_duplicates(items: List[Dict[str, Any]], threshold: float = 0.9) -> Dict[str, Any]:
    """
    Detect duplicate items using content hashing.

    Args:
        items (List[Dict[str, Any]]): List of items to check.
        threshold (float): Quality threshold for considering duplicates.

    Returns:
        Dict[str, Any]: Duplicate detection results.
    """
    seen_hashes = {}
    duplicates = []
    unique_items = []

    for item in items:
        content_hash = calculate_content_hash(item)
        if content_hash in seen_hashes:
            duplicates.append({"item": item, "original": seen_hashes[content_hash]})
        else:
            seen_hashes[content_hash] = item
            unique_items.append(item)

    return {
        "unique_items": unique_items,
        "duplicates": duplicates,
        "unique_count": len(unique_items),
        "duplicate_count": len(duplicates),
        "duplicate_rate": len(duplicates) / len(items) if items else 0.0,
    }


def filter_by_quality(
    items: List[Dict[str, Any]], schema_type: str, min_quality: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Filter items by quality score.

    Args:
        items (List[Dict[str, Any]]): List of items to filter.
        schema_type (str): Type of schema.
        min_quality (float): Minimum quality score threshold.

    Returns:
        List[Dict[str, Any]]: Filtered items meeting quality threshold.
    """
    filtered = []
    for item in items:
        quality = calculate_quality_score(item, schema_type)
        if quality >= min_quality:
            filtered.append(item)

    return filtered

