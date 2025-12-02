"""
Robust error handling module.

Handles site structure changes, parser adaptation, graceful degradation, and error categorization.
"""

import hashlib
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime
from loguru import logger


class ErrorCategory(str, Enum):
    """Error categories for classification."""

    NETWORK = "network"
    PARSING = "parsing"
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    STRUCTURE_CHANGE = "structure_change"
    UNKNOWN = "unknown"


class ErrorHandler:
    """
    Handles errors with categorization, retry strategies, and graceful degradation.
    """

    def __init__(self):
        """Initialize error handler."""
        self.error_history: List[Dict[str, Any]] = []
        self.site_structure_hashes: Dict[str, str] = {}
        self.retry_strategies: Dict[ErrorCategory, Dict[str, Any]] = {
            ErrorCategory.NETWORK: {"max_retries": 3, "backoff": "exponential"},
            ErrorCategory.PARSING: {"max_retries": 2, "backoff": "linear"},
            ErrorCategory.VALIDATION: {"max_retries": 1, "backoff": "none"},
            ErrorCategory.RATE_LIMIT: {"max_retries": 5, "backoff": "exponential"},
            ErrorCategory.STRUCTURE_CHANGE: {"max_retries": 1, "backoff": "none"},
        }

    def categorize_error(self, error: Exception, context: Dict[str, Any]) -> ErrorCategory:
        """
        Categorize an error based on type and context.

        Args:
            error (Exception): The error that occurred.
            context (Dict[str, Any]): Context information.

        Returns:
            ErrorCategory: Categorized error type.
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
            return ErrorCategory.RATE_LIMIT
        elif "timeout" in error_str or "connection" in error_str or "network" in error_str:
            return ErrorCategory.NETWORK
        elif "parse" in error_str or "json" in error_str or "html" in error_str:
            return ErrorCategory.PARSING
        elif "validation" in error_str or "schema" in error_str:
            return ErrorCategory.VALIDATION
        elif "structure" in error_str or "changed" in error_str:
            return ErrorCategory.STRUCTURE_CHANGE
        else:
            return ErrorCategory.UNKNOWN

    def detect_structure_change(self, url: str, html_content: str) -> bool:
        """
        Detect if site structure has changed by comparing HTML hash.

        Args:
            url (str): Source URL.
            html_content (str): Current HTML content.

        Returns:
            bool: True if structure changed, False otherwise.
        """
        current_hash = hashlib.md5(html_content.encode()).hexdigest()

        if url in self.site_structure_hashes:
            previous_hash = self.site_structure_hashes[url]
            if current_hash != previous_hash:
                logger.warning(f"Site structure change detected for {url}")
                self.site_structure_hashes[url] = current_hash
                return True

        self.site_structure_hashes[url] = current_hash
        return False

    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        allow_partial: bool = True,
    ) -> Dict[str, Any]:
        """
        Handle an error with appropriate strategy.

        Args:
            error (Exception): The error that occurred.
            context (Dict[str, Any]): Context information.
            allow_partial (bool): Whether to allow partial data extraction.

        Returns:
            Dict[str, Any]: Error handling result with recovery strategy.
        """
        category = self.categorize_error(error, context)
        strategy = self.retry_strategies.get(category, {"max_retries": 1, "backoff": "none"})

        error_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "category": category.value,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "strategy": strategy,
        }

        self.error_history.append(error_record)
        logger.error(f"Error [{category.value}]: {error}")

        # Determine recovery strategy
        recovery = {
            "category": category.value,
            "should_retry": strategy["max_retries"] > 0,
            "max_retries": strategy["max_retries"],
            "backoff_type": strategy["backoff"],
            "allow_partial": allow_partial and category != ErrorCategory.VALIDATION,
            "requires_parser_update": category == ErrorCategory.STRUCTURE_CHANGE,
        }

        return {
            "error": error_record,
            "recovery": recovery,
        }

    def get_retry_delay(self, category: ErrorCategory, attempt: int) -> float:
        """
        Calculate retry delay based on category and attempt number.

        Args:
            category (ErrorCategory): Error category.
            attempt (int): Retry attempt number (0-indexed).

        Returns:
            float: Delay in seconds.
        """
        strategy = self.retry_strategies.get(category, {"backoff": "none"})
        backoff_type = strategy["backoff"]

        if backoff_type == "exponential":
            return min(2.0 ** attempt, 60.0)  # Cap at 60 seconds
        elif backoff_type == "linear":
            return attempt * 2.0
        else:
            return 1.0

    def extract_partial_data(self, data: Any, schema_type: str) -> List[Dict[str, Any]]:
        """
        Attempt to extract partial data from failed extraction.

        Args:
            data (Any): Partial or malformed data.
            schema_type (str): Expected schema type.

        Returns:
            List[Dict[str, Any]]: Extracted partial data.
        """
        partial = []

        if isinstance(data, list):
            # Try to extract valid items from list
            for item in data:
                if isinstance(item, dict):
                    # Keep items with at least some fields
                    if len(item) > 0:
                        partial.append(item)
        elif isinstance(data, dict):
            # Try to find data arrays in dict
            for key, value in data.items():
                if isinstance(value, list):
                    partial.extend(self.extract_partial_data(value, schema_type))

        return partial

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics for monitoring.

        Returns:
            Dict[str, Any]: Error statistics.
        """
        if not self.error_history:
            return {"total_errors": 0, "by_category": {}}

        by_category = {}
        for error in self.error_history:
            category = error["category"]
            by_category[category] = by_category.get(category, 0) + 1

        return {
            "total_errors": len(self.error_history),
            "by_category": by_category,
            "recent_errors": self.error_history[-10:],  # Last 10 errors
        }

