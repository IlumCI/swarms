"""
Storage module for saving output locally and syncing to Google Cloud Storage.

Handles local file writing and GCS bucket synchronization.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
from loguru import logger


def save_output(path: str, data: Dict[str, Any], gcs_bucket: Optional[str] = None) -> bool:
    """
    Saves output data to local file and optionally syncs to GCS.

    Args:
        path (str): Local file path to save the data.
        data (Dict[str, Any]): Data dictionary to save.
        gcs_bucket (Optional[str]): GCS bucket name for syncing. If None, skips GCS sync.

    Returns:
        bool: True if save was successful, False otherwise.
    """
    try:
        # Ensure directory exists
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write local file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved output to {path}")

        # Sync to GCS if bucket is specified
        if gcs_bucket:
            try:
                gcs_path = f"gs://{gcs_bucket}/{path}"
                result = subprocess.run(
                    ["gsutil", "cp", str(file_path), gcs_path],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if result.returncode == 0:
                    logger.info(f"Synced to GCS: {gcs_path}")
                else:
                    logger.warning(f"GCS sync failed: {result.stderr}")
            except FileNotFoundError:
                logger.warning("gsutil not found. Skipping GCS sync.")
            except subprocess.TimeoutExpired:
                logger.warning("GCS sync timed out.")
            except Exception as e:
                logger.error(f"Error syncing to GCS: {e}")

        return True

    except Exception as e:
        logger.error(f"Error saving output to {path}: {e}")
        return False

