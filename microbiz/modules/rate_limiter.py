"""
Rate limiting and anti-scraping module.

Handles respectful scraping with delays, user agent rotation, robots.txt respect,
and exponential backoff on rate limit errors.
"""

import time
import random
import urllib.robotparser
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger


class RateLimiter:
    """
    Rate limiter with user agent rotation and robots.txt respect.

    Attributes:
        delays (Dict[str, float]): Per-source delay configurations in seconds
        user_agents (List[str]): Pool of user agents for rotation
        robots_cache (Dict[str, Optional[urllib.robotparser.RobotFileParser]]): Cached robots.txt parsers
        last_request (Dict[str, datetime]): Timestamp of last request per source
        rate_limit_backoff (Dict[str, float]): Exponential backoff multipliers per source
    """

    def __init__(
        self,
        default_delay: float = 2.0,
        min_delay: float = 1.0,
        max_delay: float = 5.0,
        respect_robots: bool = True,
    ):
        """
        Initialize rate limiter.

        Args:
            default_delay (float): Default delay between requests in seconds.
            min_delay (float): Minimum delay in seconds.
            max_delay (float): Maximum delay in seconds.
            respect_robots (bool): Whether to respect robots.txt files.
        """
        self.default_delay = default_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.respect_robots = respect_robots

        self.delays: Dict[str, float] = {}
        self.user_agents: List[str] = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        ]
        self.robots_cache: Dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}
        self.last_request: Dict[str, datetime] = {}
        self.rate_limit_backoff: Dict[str, float] = defaultdict(lambda: 1.0)
        self.user_agent_index: Dict[str, int] = defaultdict(lambda: 0)

    def get_user_agent(self, source_name: str) -> str:
        """
        Get a user agent for a source (with rotation).

        Args:
            source_name (str): Name of the source.

        Returns:
            str: User agent string.
        """
        index = self.user_agent_index[source_name] % len(self.user_agents)
        self.user_agent_index[source_name] = (index + 1) % len(self.user_agents)
        return self.user_agents[index]

    def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """
        Check if URL can be fetched according to robots.txt.

        Args:
            url (str): URL to check.
            user_agent (str): User agent string.

        Returns:
            bool: True if URL can be fetched, False otherwise.
        """
        if not self.respect_robots:
            return True

        try:
            from urllib.parse import urlparse, urljoin

            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

            if robots_url not in self.robots_cache:
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(robots_url)
                try:
                    rp.read()
                    self.robots_cache[robots_url] = rp
                except Exception as e:
                    logger.warning(f"Could not read robots.txt from {robots_url}: {e}")
                    self.robots_cache[robots_url] = None

            rp = self.robots_cache[robots_url]
            if rp is None:
                return True  # Allow if robots.txt unavailable

            return rp.can_fetch(user_agent, url)

        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {e}")
            return True  # Allow on error

    def wait_if_needed(self, source_name: str, source_url: str) -> None:
        """
        Wait if needed based on rate limiting rules.

        Args:
            source_name (str): Name of the source.
            source_url (str): URL of the source.
        """
        delay = self.delays.get(source_name, self.default_delay)

        # Apply exponential backoff if source was rate-limited
        backoff_multiplier = self.rate_limit_backoff[source_name]
        delay *= backoff_multiplier
        delay = max(self.min_delay, min(delay, self.max_delay))

        # Check if we need to wait
        if source_name in self.last_request:
            elapsed = (datetime.now() - self.last_request[source_name]).total_seconds()
            if elapsed < delay:
                sleep_time = delay - elapsed
                logger.debug(f"Rate limiting: waiting {sleep_time:.2f}s for {source_name}")
                time.sleep(sleep_time)

        self.last_request[source_name] = datetime.now()

    def handle_rate_limit_error(self, source_name: str) -> None:
        """
        Handle rate limit error by increasing backoff.

        Args:
            source_name (str): Name of the source that was rate-limited.
        """
        self.rate_limit_backoff[source_name] *= 2.0
        logger.warning(
            f"Rate limit hit for {source_name}. Backoff multiplier: {self.rate_limit_backoff[source_name]}"
        )

    def reset_backoff(self, source_name: str) -> None:
        """
        Reset backoff multiplier after successful request.

        Args:
            source_name (str): Name of the source.
        """
        if self.rate_limit_backoff[source_name] > 1.0:
            self.rate_limit_backoff[source_name] = max(1.0, self.rate_limit_backoff[source_name] * 0.9)

    def set_source_delay(self, source_name: str, delay: float) -> None:
        """
        Set custom delay for a specific source.

        Args:
            source_name (str): Name of the source.
            delay (float): Delay in seconds.
        """
        self.delays[source_name] = delay

