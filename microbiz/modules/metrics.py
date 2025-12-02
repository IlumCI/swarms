"""
Performance metrics collection module.

Builds performance snapshots for Board consumption.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from loguru import logger


def build_performance_snapshot(worker_type: str, output_dir: str = "output") -> Dict[str, Any]:
    """
    Build performance snapshot for a worker from recent outputs.

    Args:
        worker_type (str): Type of worker (auctions, tenders, etc.).
        output_dir (str): Output directory path.

    Returns:
        Dict[str, Any]: Performance snapshot.
    """
    output_path = Path(output_dir) / worker_type
    if not output_path.exists():
        return {
            "worker_type": worker_type,
            "success_rate": 0.0,
            "data_quality": 0.0,
            "error_count": 0,
            "items_collected": 0,
            "last_run": None,
        }

    # Get recent output files (last 7 days)
    cutoff = datetime.now() - timedelta(days=7)
    recent_files = []
    for file_path in output_path.glob("*.json"):
        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime >= cutoff:
                recent_files.append((file_path, mtime))
        except Exception:
            continue

    if not recent_files:
        return {
            "worker_type": worker_type,
            "success_rate": 0.0,
            "data_quality": 0.0,
            "error_count": 0,
            "items_collected": 0,
            "last_run": None,
        }

    # Sort by time, most recent first
    recent_files.sort(key=lambda x: x[1], reverse=True)

    total_items = 0
    total_quality = 0.0
    total_errors = 0
    successful_runs = 0
    last_run_time = None

    for file_path, mtime in recent_files[:10]:  # Last 10 runs
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            stats = data.get("statistics", {})
            total_items += stats.get("total_items", 0)
            total_errors += stats.get("errors", 0)

            # Estimate quality from validation if available
            if "data" in data and len(data["data"]) > 0:
                successful_runs += 1
                # Rough quality estimate based on data completeness
                sample = data["data"][0] if data["data"] else {}
                quality = len([k for k, v in sample.items() if v is not None]) / max(len(sample), 1)
                total_quality += quality

            if not last_run_time or mtime > last_run_time:
                last_run_time = mtime

        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")

    runs_count = len(recent_files)
    success_rate = successful_runs / runs_count if runs_count > 0 else 0.0
    avg_quality = total_quality / successful_runs if successful_runs > 0 else 0.0

    return {
        "worker_type": worker_type,
        "success_rate": success_rate,
        "data_quality": avg_quality,
        "error_count": total_errors,
        "items_collected": total_items,
        "runs_analyzed": runs_count,
        "last_run": last_run_time.isoformat() if last_run_time else None,
    }


def build_all_worker_snapshots(output_dir: str = "output") -> Dict[str, Dict[str, Any]]:
    """
    Build performance snapshots for all workers.

    Args:
        output_dir (str): Output directory path.

    Returns:
        Dict[str, Dict[str, Any]]: Snapshots keyed by worker type.
    """
    workers = ["auctions", "tenders", "businesses", "jobs", "osint", "realestate"]
    snapshots = {}

    for worker_type in workers:
        snapshots[worker_type] = build_performance_snapshot(worker_type, output_dir)

    return snapshots

