"""
Supervisor for data swarm system.

Time-based scheduler that orchestrates worker agents at scheduled times.
Can optionally use AgentRearrange for multi-agent workflows.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import workers
try:
    from workers import auctions, tenders, businesses, jobs, osint, realestate
except ImportError as e:
    logger.error(f"Failed to import workers: {e}")
    sys.exit(1)

# Try to import AgentRearrange for optional multi-agent workflows
try:
    from swarms.structs.agent_rearrange import AgentRearrange
    AGENT_REARRANGE_AVAILABLE = True
except ImportError:
    AGENT_REARRANGE_AVAILABLE = False
    logger.warning("AgentRearrange not available. Using direct worker execution only.")

# Try to import Board of Directors
try:
    from business.board import create_board_swarm, load_business_config
    BOARD_AVAILABLE = True
except ImportError:
    BOARD_AVAILABLE = False
    logger.warning("Board of Directors not available. Install required dependencies.")


def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.json.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def get_current_time() -> str:
    """
    Get current time in HH:MM format.

    Returns:
        str: Current time as HH:MM string.
    """
    return datetime.now().strftime("%H:%M")


def should_run_task(task_name: str, scheduled_time: str, last_run: Dict[str, float]) -> bool:
    """
    Check if a task should run based on current time and schedule.

    Args:
        task_name (str): Name of the task.
        scheduled_time (str): Scheduled time in HH:MM format.
        current_time (str): Current time in HH:MM format.
        last_run (Dict[str, float]): Dictionary tracking last run times.

    Returns:
        bool: True if task should run, False otherwise.
    """
    current_time = get_current_time()
    
    # Check if current time matches scheduled time
    if current_time == scheduled_time:
        # Check if we've already run this task at this time
        last_run_key = f"{task_name}_{scheduled_time}"
        last_timestamp = last_run.get(last_run_key, 0)
        current_timestamp = time.time()
        
        # Only run if we haven't run in the last 5 minutes (to avoid duplicate runs)
        if current_timestamp - last_timestamp > 300:
            last_run[last_run_key] = current_timestamp
            return True
    
    return False


def run_worker(worker_module, task_name: str, config: Dict[str, Any]) -> bool:
    """
    Run a worker module with retry logic.

    Supports optional AgentRearrange for multi-agent workflows if configured.

    Args:
        worker_module: The worker module to run.
        task_name (str): Name of the task for logging.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        bool: True if worker succeeded, False otherwise.
    """
    supervisor_config = config.get("supervisor", {})
    max_retries = supervisor_config.get("max_retries", 1)
    backoff_min = supervisor_config.get("retry_backoff_min", 30)
    backoff_max = supervisor_config.get("retry_backoff_max", 120)
    use_rearrange = supervisor_config.get("use_agent_rearrange", False)

    # Check if AgentRearrange should be used for this task
    if use_rearrange and AGENT_REARRANGE_AVAILABLE:
        return _run_worker_with_rearrange(worker_module, task_name, config)

    # Standard execution
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Running {task_name} worker (attempt {attempt + 1})...")
            exit_code = worker_module.run_worker()
            
            if exit_code == 0:
                logger.info(f"{task_name} worker completed successfully")
                return True
            else:
                logger.warning(f"{task_name} worker exited with code {exit_code}")
                
        except Exception as e:
            logger.error(f"Error running {task_name} worker: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        # Retry with backoff if not last attempt
        if attempt < max_retries:
            backoff_time = random.randint(backoff_min, backoff_max)
            logger.info(f"Retrying {task_name} in {backoff_time} seconds...")
            time.sleep(backoff_time)

    logger.error(f"{task_name} worker failed after {max_retries + 1} attempts")
    return False


def _run_worker_with_rearrange(worker_module, task_name: str, config: Dict[str, Any]) -> bool:
    """
    Run worker using AgentRearrange for multi-agent workflow (if applicable).

    Args:
        worker_module: The worker module to run.
        task_name (str): Name of the task for logging.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        bool: True if worker succeeded, False otherwise.
    """
    # For now, this is a placeholder for future AgentRearrange integration
    # The current workers are single-agent, but this can be extended
    # to create multi-agent workflows when needed
    logger.info(f"AgentRearrange mode requested for {task_name}, using standard execution")
    return run_worker(worker_module, task_name, config)


def main():
    """
    Main supervisor loop with optional Board of Directors governance.

    Continuously checks schedule and runs workers at their scheduled times.
    Can optionally use Board of Directors for strategic decision-making.
    """
    logger.info("Starting data swarm supervisor...")
    
    config = load_config()
    schedule = config.get("schedule", {})
    check_interval = config.get("supervisor", {}).get("check_interval", 30)
    
    # Track last run times to avoid duplicate executions
    last_run: Dict[str, float] = {}
    
    # Worker mapping
    workers = {
        "auctions": auctions,
        "tenders": tenders,
        "businesses": businesses,
        "jobs": jobs,
        "osint": osint,
        "realestate": realestate,
    }

    # Initialize Board of Directors and Business Integration
    board_swarm = None
    business_integration = None
    if BOARD_AVAILABLE:
        try:
            from business.integration import BusinessIntegration
            business_integration = BusinessIntegration()
            board_swarm = business_integration.board_swarm

            if board_swarm:
                logger.info("Board of Directors initialized and ready")
        except Exception as e:
            logger.warning(f"Failed to initialize Board of Directors: {e}")

    # Scaling plan (in-memory, can be persisted)
    scaling_plan: Dict[str, Dict[str, Any]] = {}

    logger.info(f"Supervisor running with {len(schedule)} scheduled tasks")
    logger.info(f"Check interval: {check_interval} seconds")
    if board_swarm:
        logger.info("Board of Directors: ENABLED")
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            current_time = get_current_time()
            
            # Check each scheduled task
            for task_name, scheduled_time in schedule.items():
                if should_run_task(task_name, scheduled_time, last_run):
                    logger.info(f"Time to run {task_name} (scheduled: {scheduled_time})")
                    
                    if task_name in workers:
                        success = run_worker(workers[task_name], task_name, config)
                        
                        # Process through business integration if available
                        if business_integration and success:
                            try:
                                # Load recent output to process
                                from pathlib import Path
                                import glob
                                output_dir = Path("output") / task_name
                                recent_files = sorted(output_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                                
                                if recent_files:
                                    with open(recent_files[0], "r") as f:
                                        output_data = json.load(f)
                                    
                                    # Process through business pipeline
                                    biz_result = business_integration.process_collected_data(
                                        task_name,
                                        output_data.get("data", []),
                                        output_data.get("statistics", {}),
                                    )
                                    logger.info(f"Business processing complete for {task_name}: {biz_result.get('products_created', [])}")

                                # Get scaling decision
                                performance = {
                                    "success_rate": 1.0 if success else 0.0,
                                    "data_quality": output_data.get("statistics", {}).get("average_quality", 0.8),
                                    "items_collected": len(output_data.get("data", [])),
                                }
                                
                                scaling_decision = business_integration.make_scaling_decision(task_name, performance)
                                scaling_plan[task_name] = scaling_decision
                                
                                # Apply scaling if action is not "maintain"
                                if scaling_decision.get("action") != "maintain":
                                    logger.info(f"Scaling decision for {task_name}: {scaling_decision['action']} - {scaling_decision.get('reason', '')}")
                                    # Could update schedule here based on new_frequency
                                    
                            except Exception as e:
                                logger.warning(f"Error in business processing: {e}")
                                import traceback
                                logger.debug(traceback.format_exc())
                    else:
                        logger.warning(f"Unknown worker: {task_name}")

            # Sleep until next check
            time.sleep(check_interval)

    except KeyboardInterrupt:
        logger.info("Supervisor stopped by user")
    except Exception as e:
        logger.error(f"Supervisor error: {e}")
        raise


if __name__ == "__main__":
    main()

