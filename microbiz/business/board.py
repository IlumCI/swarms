"""
Board of Directors setup and configuration.

Creates and manages the BoardOfDirectorsSwarm for autonomous micro-business governance.
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from loguru import logger

try:
    from swarms.structs.board_of_directors_swarm import (
        BoardOfDirectorsSwarm,
        BoardMember,
        BoardMemberRole,
    )
    from swarms.structs.agent import Agent
    BOARD_AVAILABLE = True
except ImportError:
    BOARD_AVAILABLE = False
    logger.warning("BoardOfDirectorsSwarm not available. Install swarms package.")


def load_business_config() -> Dict[str, Any]:
    """
    Load business configuration.

    Returns:
        Dict[str, Any]: Business configuration dictionary.
    """
    config_path = Path(__file__).parent.parent / "config" / "business_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def create_board_members(config: Optional[Dict[str, Any]] = None) -> List[BoardMember]:
    """
    Create board members with specialized roles.

    Args:
        config (Optional[Dict[str, Any]]): Business configuration.

    Returns:
        List[BoardMember]: List of board member agents.
    """
    if not BOARD_AVAILABLE:
        logger.error("BoardOfDirectorsSwarm not available")
        return []

    if config is None:
        config = load_business_config()

    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    board_config = config.get("board", {})

    # Chairman - Strategic leadership
    chairman = Agent(
        agent_name="Chairman",
        agent_description="Executive Chairman with strategic vision and final decision authority",
        model_name=model_name,
        max_loops=1,
        system_prompt=(
            "You are the Executive Chairman of a data-driven micro-business. "
            "Your responsibilities include: strategic leadership, facilitating board meetings, "
            "making final decisions, and ensuring the business achieves its goals. "
            "You coordinate with other board members and make decisions based on collective input."
        ),
    )

    # Treasurer - Financial oversight
    treasurer = Agent(
        agent_name="Treasurer",
        agent_description="Chief Financial Officer managing finances, costs, and budgets",
        model_name=model_name,
        max_loops=1,
        system_prompt=(
            "You are the Treasurer of a data-driven micro-business. "
            "Your responsibilities include: financial oversight, cost analysis, budget management, "
            "revenue tracking, profit/loss calculations, and providing financial recommendations to the board. "
            "You monitor all financial metrics and report to the board regularly."
        ),
    )

    # Secretary - Documentation
    secretary = Agent(
        agent_name="Secretary",
        agent_description="Board Secretary handling documentation and record-keeping",
        model_name=model_name,
        max_loops=1,
        system_prompt=(
            "You are the Board Secretary of a data-driven micro-business. "
            "Your responsibilities include: documenting board meetings, maintaining records, "
            "coordinating marketing efforts, and ensuring all business activities are properly documented."
        ),
    )

    # Executive Director - Strategic planning
    executive_director = Agent(
        agent_name="Executive Director",
        agent_description="Executive Director handling strategic planning and operations",
        model_name=model_name,
        max_loops=1,
        system_prompt=(
            "You are the Executive Director of a data-driven micro-business. "
            "Your responsibilities include: strategic planning, operations coordination, "
            "identifying market opportunities, optimizing data collection strategies, "
            "and managing sales automation. You provide strategic recommendations to the board."
        ),
    )

    # Vice Chairman - Operational support
    vice_chairman = Agent(
        agent_name="Vice Chairman",
        agent_description="Vice Chairman providing operational support and day-to-day management",
        model_name=model_name,
        max_loops=1,
        system_prompt=(
            "You are the Vice Chairman of a data-driven micro-business. "
            "Your responsibilities include: operational support, day-to-day management, "
            "overseeing customer service, and supporting the Chairman in decision-making."
        ),
    )

    # Create board members with voting weights
    board_members = [
        BoardMember(
            agent=chairman,
            role=BoardMemberRole.CHAIRMAN,
            voting_weight=board_config.get("chairman_weight", 1.5),
            expertise_areas=["leadership", "strategy", "decision_making"],
        ),
        BoardMember(
            agent=treasurer,
            role=BoardMemberRole.TREASURER,
            voting_weight=board_config.get("treasurer_weight", 1.0),
            expertise_areas=["finance", "budgeting", "cost_analysis"],
        ),
        BoardMember(
            agent=secretary,
            role=BoardMemberRole.SECRETARY,
            voting_weight=board_config.get("secretary_weight", 1.0),
            expertise_areas=["documentation", "marketing", "record_keeping"],
        ),
        BoardMember(
            agent=executive_director,
            role=BoardMemberRole.EXECUTIVE_DIRECTOR,
            voting_weight=board_config.get("executive_weight", 1.2),
            expertise_areas=["strategy", "operations", "sales", "market_analysis"],
        ),
        BoardMember(
            agent=vice_chairman,
            role=BoardMemberRole.VICE_CHAIRMAN,
            voting_weight=board_config.get("vice_chairman_weight", 1.2),
            expertise_areas=["operations", "customer_service", "management"],
        ),
    ]

    return board_members


def create_board_swarm(
    worker_agents: Optional[List[Agent]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[BoardOfDirectorsSwarm]:
    """
    Create BoardOfDirectorsSwarm instance.

    Args:
        worker_agents (Optional[List[Agent]]): List of worker agents for the board to manage.
        config (Optional[Dict[str, Any]]): Business configuration.

    Returns:
        Optional[BoardOfDirectorsSwarm]: Board swarm instance or None if unavailable.
    """
    if not BOARD_AVAILABLE:
        logger.error("BoardOfDirectorsSwarm not available")
        return None

    if config is None:
        config = load_business_config()

    board_config = config.get("board", {})
    board_members = create_board_members(config)

    if not board_members:
        logger.error("Failed to create board members")
        return None

    swarm = BoardOfDirectorsSwarm(
        name="MicroBusiness Board of Directors",
        description="Board of Directors for autonomous data-driven micro-business",
        board_members=board_members,
        agents=worker_agents or [],
        max_loops=board_config.get("max_loops", 3),
        board_model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        decision_threshold=board_config.get("decision_threshold", 0.6),
        enable_voting=board_config.get("enable_voting", True),
        enable_consensus=board_config.get("enable_consensus", True),
        verbose=board_config.get("verbose", False),
    )

    logger.info("Board of Directors swarm created successfully")
    return swarm

