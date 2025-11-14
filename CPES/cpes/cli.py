"""
Command-line interface for CPES.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional
from loguru import logger

from .core.persona import Persona
from .core.cognitive_loop import CognitiveLoop
from .memory.semantic_kg import SemanticKnowledgeGraph
from .memory.episodic_memory import EpisodicMemoryStore
from .memory.procedural_memory import ProceduralMemoryStore
from .utils.embedding import EmbeddingModel
from .utils.llm_wrapper import LLMWrapper


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CPES: Composable Persona Emulation Stack"
    )
    
    parser.add_argument(
        "--persona",
        type=str,
        required=True,
        help="Path to persona YAML file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "litellm"],
        help="LLM provider (default: openai)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (or set environment variable)"
    )
    
    parser.add_argument(
        "--use-litellm",
        action="store_true",
        help="Use LiteLLM for model access"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Single input to process (non-interactive mode)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    try:
        # Load persona
        persona = Persona(args.persona)
        logger.info(f"Loaded persona: {persona.spec.name}")
        
        # Initialize memory systems
        semantic_kg = SemanticKnowledgeGraph()
        episodic_memory = EpisodicMemoryStore()
        procedural_memory = ProceduralMemoryStore()
        
        # Initialize embedding model
        embedding_model = EmbeddingModel("all-MiniLM-L6-v2")
        
        # Initialize LLM wrapper
        api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("No API key provided. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable, or use --api-key")
            sys.exit(1)
        
        llm_wrapper = LLMWrapper(
            model_name=args.model,
            provider=args.provider,
            api_key=api_key,
            use_litellm=args.use_litellm
        )
        
        # Initialize cognitive loop
        cognitive_loop = CognitiveLoop(
            persona=persona,
            semantic_kg=semantic_kg,
            episodic_memory=episodic_memory,
            procedural_memory=procedural_memory,
            embedding_model=embedding_model,
            llm_wrapper=llm_wrapper
        )
        
        if args.interactive:
            run_interactive_mode(cognitive_loop, persona)
        elif args.input:
            response = cognitive_loop.process_input(args.input)
            print(response)
        else:
            print("Use --interactive for interactive mode or --input for single input")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


def run_interactive_mode(cognitive_loop: CognitiveLoop, persona: Persona):
    """Run interactive mode."""
    print(f"\n🤖 CPES Interactive Mode")
    print(f"Persona: {persona.spec.name}")
    print(f"Model: {cognitive_loop.llm_wrapper.model_name}")
    print(f"Provider: {cognitive_loop.llm_wrapper.provider}")
    print(f"Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            response = cognitive_loop.process_input(user_input)
            print(f"\n{persona.spec.name}: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
