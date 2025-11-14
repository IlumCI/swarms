"""
Search tools for ReAct agents.

This module provides search-related tools that ReAct agents can use
to find information and perform lookups.
"""

from typing import Any, Dict, List, Optional
from loguru import logger
import re


class SearchTools:
    """Collection of search tools for ReAct agents."""
    
    def __init__(self, knowledge_base: Optional[Dict[str, str]] = None):
        """
        Initialize search tools.
        
        Args:
            knowledge_base: Optional knowledge base for local search
        """
        self.knowledge_base = knowledge_base or {}
    
    def search(self, query: str) -> str:
        """
        Search for information using the available knowledge base.
        
        Args:
            query: Search query
            
        Returns:
            Search results
        """
        try:
            if not self.knowledge_base:
                return f"No knowledge base available. Query: {query}"
            
            # Simple keyword matching
            query_lower = query.lower()
            results = []
            
            for key, value in self.knowledge_base.items():
                if any(word in value.lower() for word in query_lower.split()):
                    results.append(f"{key}: {value[:200]}...")
            
            if results:
                return f"Found {len(results)} results:\n" + "\n".join(results[:5])
            else:
                return f"No results found for: {query}"
                
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def lookup(self, key: str) -> str:
        """
        Look up a specific key in the knowledge base.
        
        Args:
            key: Key to look up
            
        Returns:
            Value associated with the key
        """
        try:
            if key in self.knowledge_base:
                return self.knowledge_base[key]
            else:
                return f"Key '{key}' not found in knowledge base"
                
        except Exception as e:
            return f"Lookup error: {str(e)}"
    
    def list_keys(self, pattern: str = "") -> str:
        """
        List all keys in the knowledge base, optionally filtered by pattern.
        
        Args:
            pattern: Optional pattern to filter keys
            
        Returns:
            Comma-separated list of keys
        """
        try:
            keys = list(self.knowledge_base.keys())
            
            if pattern:
                pattern_re = re.compile(pattern, re.IGNORECASE)
                keys = [k for k in keys if pattern_re.search(k)]
            
            return ", ".join(keys) if keys else "No keys found"
            
        except Exception as e:
            return f"List keys error: {str(e)}"
    
    def add_knowledge(self, key: str, value: str) -> str:
        """
        Add knowledge to the knowledge base.
        
        Args:
            key: Key for the knowledge
            value: Value to store
            
        Returns:
            Confirmation message
        """
        try:
            self.knowledge_base[key] = value
            return f"Added knowledge: {key}"
            
        except Exception as e:
            return f"Add knowledge error: {str(e)}"
    
    def get_tools(self) -> Dict[str, callable]:
        """Get dictionary of available search tools."""
        return {
            "search": self.search,
            "lookup": self.lookup,
            "list_keys": self.list_keys,
            "add_knowledge": self.add_knowledge
        }
