"""
Memory system initialization module.

Provides optional memory system support for Swarms Agents.
"""

from typing import Optional, Any
from loguru import logger


def get_memory_system(memory_config: dict) -> Optional[Any]:
    """
    Initialize memory system based on configuration.

    Args:
        memory_config (dict): Memory configuration from config.json.

    Returns:
        Optional[Any]: Memory system instance or None if disabled/unavailable.
    """
    if not memory_config.get("enabled", False):
        return None

    memory_type = memory_config.get("type", "").lower()

    if not memory_type:
        logger.warning("Memory enabled but no type specified. Skipping memory system.")
        return None

    try:
        if memory_type == "chromadb":
            from swarms_memory import ChromaDB

            return ChromaDB(
                metric="cosine",
                output_dir=memory_config.get("output_dir", "data_swarm_memory"),
            )

        elif memory_type == "faiss":
            from swarms_memory.faiss_wrapper import FAISSDB

            return FAISSDB(
                output_dir=memory_config.get("output_dir", "data_swarm_memory"),
            )

        elif memory_type == "qdrant":
            from swarms_memory import QdrantDB
            from qdrant_client import QdrantClient

            client = QdrantClient(
                url=memory_config.get("url", "http://localhost:6333"),
                api_key=memory_config.get("api_key"),
            )

            return QdrantDB(
                client=client,
                embedding_model=memory_config.get("embedding_model", "text-embedding-3-small"),
                collection_name=memory_config.get("collection_name", "data_swarm"),
            )

        else:
            logger.warning(f"Unknown memory type: {memory_type}. Skipping memory system.")
            return None

    except ImportError as e:
        logger.warning(f"Memory system {memory_type} not available: {e}. Install swarms-memory package.")
        return None
    except Exception as e:
        logger.error(f"Error initializing memory system {memory_type}: {e}")
        return None

