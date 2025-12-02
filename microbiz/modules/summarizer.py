"""
Summarizer module with Swarms Agent (primary) and llama.cpp fallback.

Provides text summarization capabilities using LLMs.
"""

import os
import subprocess
from typing import Optional
from loguru import logger

# Try to import Swarms Agent
try:
    from swarms import Agent
    SWARMS_AVAILABLE = True
except ImportError:
    SWARMS_AVAILABLE = False
    logger.warning("Swarms not available. Summarization will use llama.cpp fallback only.")


def summarize(
    text: str,
    use_llama: bool = False,
    model_name: Optional[str] = None,
    llama_path: Optional[str] = None,
) -> str:
    """
    Summarizes text using Swarms Agent (default) or llama.cpp (fallback).

    Args:
        text (str): Text to summarize.
        use_llama (bool): Force use of llama.cpp instead of Swarms Agent.
        model_name (Optional[str]): Model name for Swarms Agent. Uses env vars if None.
        llama_path (Optional[str]): Path to llama.cpp model file.

    Returns:
        str: Summary of the input text.
    """
    # Use llama.cpp if explicitly requested or if Swarms is unavailable
    if use_llama or not SWARMS_AVAILABLE:
        return _summarize_with_llama(text, llama_path)

    # Try Swarms Agent first
    try:
        return _summarize_with_swarms(text, model_name)
    except Exception as e:
        logger.warning(f"Swarms Agent summarization failed: {e}. Falling back to llama.cpp.")
        return _summarize_with_llama(text, llama_path)


def _summarize_with_swarms(text: str, model_name: Optional[str] = None) -> str:
    """
    Summarizes text using Swarms Agent.

    Args:
        text (str): Text to summarize.
        model_name (Optional[str]): Model name. Uses OPENAI_API_KEY from env if None.

    Returns:
        str: Summary of the input text.
    """
    if not SWARMS_AVAILABLE:
        raise ImportError("Swarms not available")

    # Get model name from env or use default
    if not model_name:
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

    # Create summarization agent
    agent = Agent(
        agent_name="Summarizer",
        model_name=model_name,
        system_prompt="You are a summarization expert. Provide concise, informative summaries of the given text.",
        max_loops=1,
        verbose=False,
    )

    # Generate summary
    prompt = f"Summarize the following text in 2-3 sentences:\n\n{text[:2000]}"  # Limit input length
    summary = agent.run(prompt)

    return summary if summary else "Summary generation failed."


def _summarize_with_llama(text: str, llama_path: Optional[str] = None) -> str:
    """
    Summarizes text using llama.cpp.

    Args:
        text (str): Text to summarize.
        llama_path (Optional[str]): Path to llama.cpp model file.

    Returns:
        str: Summary of the input text, or error message if llama.cpp unavailable.
    """
    if not llama_path:
        llama_path = os.getenv("LLAMA_MODEL_PATH")
        if not llama_path:
            logger.warning("llama.cpp path not specified. Skipping summarization.")
            return "Summarization unavailable (llama.cpp not configured)."

    try:
        # Check if llama.cpp binary exists
        llama_binary = os.getenv("LLAMA_CPP_BINARY", "llama-cli")
        prompt = f"Summarize the following text in 2-3 sentences:\n\n{text[:2000]}"

        result = subprocess.run(
            [llama_binary, "-m", llama_path, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            logger.error(f"llama.cpp execution failed: {result.stderr}")
            return "Summarization failed (llama.cpp error)."

    except FileNotFoundError:
        logger.warning("llama.cpp binary not found. Skipping summarization.")
        return "Summarization unavailable (llama.cpp not installed)."
    except subprocess.TimeoutExpired:
        logger.warning("llama.cpp summarization timed out.")
        return "Summarization timed out."
    except Exception as e:
        logger.error(f"Error with llama.cpp summarization: {e}")
        return f"Summarization error: {str(e)}"

