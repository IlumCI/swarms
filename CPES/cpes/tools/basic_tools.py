"""
Basic tools for ReAct agents.

This module provides fundamental tools like calculation, text processing,
and simple operations that ReAct agents can use.
"""

import re
import math
from typing import Any, Dict, List
from loguru import logger


class BasicTools:
    """Collection of basic tools for ReAct agents."""
    
    @staticmethod
    def calc(expression: str) -> str:
        """
        Evaluate mathematical expressions safely.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Result of the calculation or error message
        """
        try:
            # Clean the expression
            expression = expression.strip()
            
            # Basic safety checks
            if not re.match(r'^[0-9+\-*/().\s]+$', expression):
                return "Error: Invalid characters in expression"
            
            # Evaluate safely
            result = eval(expression, {"__builtins__": {}, "math": math}, {})
            return str(result)
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def compare(a: str, b: str) -> str:
        """
        Compare two values and return the result.
        
        Args:
            a: First value to compare
            b: Second value to compare
            
        Returns:
            Comparison result
        """
        try:
            # Try to convert to numbers first
            try:
                num_a = float(a)
                num_b = float(b)
                if num_a > num_b:
                    return f"{a} > {b}"
                elif num_a < num_b:
                    return f"{a} < {b}"
                else:
                    return f"{a} = {b}"
            except ValueError:
                # String comparison
                if a > b:
                    return f"'{a}' comes after '{b}'"
                elif a < b:
                    return f"'{a}' comes before '{b}'"
                else:
                    return f"'{a}' equals '{b}'"
                    
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def format_text(text: str, format_type: str = "clean") -> str:
        """
        Format text according to specified type.
        
        Args:
            text: Text to format
            format_type: Type of formatting (clean, upper, lower, title)
            
        Returns:
            Formatted text
        """
        try:
            if format_type == "clean":
                # Remove extra whitespace and normalize
                return re.sub(r'\s+', ' ', text.strip())
            elif format_type == "upper":
                return text.upper()
            elif format_type == "lower":
                return text.lower()
            elif format_type == "title":
                return text.title()
            else:
                return text
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def extract_numbers(text: str) -> str:
        """
        Extract all numbers from text.
        
        Args:
            text: Text to extract numbers from
            
        Returns:
            Comma-separated list of numbers found
        """
        try:
            numbers = re.findall(r'-?\d+\.?\d*', text)
            return ", ".join(numbers) if numbers else "No numbers found"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def count_words(text: str) -> str:
        """
        Count words in text.
        
        Args:
            text: Text to count words in
            
        Returns:
            Number of words
        """
        try:
            words = text.split()
            return str(len(words))
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def split_text(text: str, delimiter: str = " ") -> str:
        """
        Split text by delimiter.
        
        Args:
            text: Text to split
            delimiter: Delimiter to split by
            
        Returns:
            Split text as comma-separated values
        """
        try:
            parts = text.split(delimiter)
            return ", ".join(parts)
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def get_tools() -> Dict[str, callable]:
        """Get dictionary of available tools."""
        return {
            "calc": BasicTools.calc,
            "compare": BasicTools.compare,
            "format_text": BasicTools.format_text,
            "extract_numbers": BasicTools.extract_numbers,
            "count_words": BasicTools.count_words,
            "split_text": BasicTools.split_text
        }
