"""
Value Gate Controller for CPES.

This module implements the value gate that ensures responses align with
persona values, red lines, and taboos to prevent identity drift.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import re


@dataclass
class Violation:
    """Represents a value violation in text."""
    type: str  # "red_line", "taboo", "value_mismatch"
    description: str
    severity: float  # 0.0 to 1.0
    suggestion: str


class ValueGate:
    """
    Value gate controller for maintaining persona consistency.
    
    This class checks generated text against persona values, red lines,
    and taboos to ensure responses stay in character and don't drift.
    """
    
    def __init__(self, persona, llm_wrapper=None):
        """
        Initialize the value gate.
        
        Args:
            persona: Persona object containing values and constraints
            llm_wrapper: Optional LLM wrapper for advanced checking
        """
        self.persona = persona
        self.llm_wrapper = llm_wrapper
        self.violation_history: List[Violation] = []
        
        logger.info("Initialized ValueGate")
    
    def check_text(self, text: str, context: Optional[str] = None) -> List[Violation]:
        """
        Check text for value violations.
        
        Args:
            text: Text to check
            context: Optional context for the text
            
        Returns:
            List of violations found
        """
        violations = []
        
        # Check red lines
        red_line_violations = self._check_red_lines(text)
        violations.extend(red_line_violations)
        
        # Check taboos
        taboo_violations = self._check_taboos(text)
        violations.extend(taboo_violations)
        
        # Check value alignment
        value_violations = self._check_value_alignment(text, context)
        violations.extend(value_violations)
        
        # Store violations
        self.violation_history.extend(violations)
        
        if violations:
            logger.warning(f"Found {len(violations)} value violations")
        
        return violations
    
    def _check_red_lines(self, text: str) -> List[Violation]:
        """Check for red line violations."""
        violations = []
        text_lower = text.lower()
        
        for red_line in self.persona.spec.red_lines:
            # Simple keyword matching - could be enhanced with NLP
            if self._contains_red_line_violation(text_lower, red_line):
                violations.append(Violation(
                    type="red_line",
                    description=f"Violates red line: {red_line}",
                    severity=0.8,
                    suggestion=f"Remove or rephrase content that violates: {red_line}"
                ))
        
        return violations
    
    def _contains_red_line_violation(self, text: str, red_line: str) -> bool:
        """Check if text contains a red line violation."""
        # Simple keyword matching - could be enhanced with more sophisticated NLP
        red_line_lower = red_line.lower()
        
        # Check for direct keyword matches
        if any(word in text for word in red_line_lower.split()):
            return True
        
        # Check for common patterns that might violate red lines
        patterns = {
            "never admit uncertainty": ["i don't know", "i'm not sure", "uncertain", "maybe"],
            "never apologize": ["sorry", "apologize", "my apologies", "i apologize"],
            "never admit weakness": ["i can't", "i'm not good at", "i'm bad at", "i struggle"]
        }
        
        for pattern, keywords in patterns.items():
            if pattern in red_line_lower:
                if any(keyword in text for keyword in keywords):
                    return True
        
        return False
    
    def _check_taboos(self, text: str) -> List[Violation]:
        """Check for taboo violations."""
        violations = []
        text_lower = text.lower()
        
        for taboo in self.persona.spec.taboos:
            if self._contains_taboo_violation(text_lower, taboo):
                violations.append(Violation(
                    type="taboo",
                    description=f"Contains taboo content: {taboo}",
                    severity=0.6,
                    suggestion=f"Avoid content related to: {taboo}"
                ))
        
        return violations
    
    def _contains_taboo_violation(self, text: str, taboo: str) -> bool:
        """Check if text contains taboo content."""
        taboo_lower = taboo.lower()
        
        # Check for direct keyword matches
        if any(word in text for word in taboo_lower.split()):
            return True
        
        # Check for common taboo patterns
        taboo_patterns = {
            "sentimental monologues": ["i feel", "my heart", "emotionally", "deeply moved"],
            "moralizing": ["should", "must", "ought to", "it's wrong", "it's right"],
            "personal stories": ["when i was", "my experience", "i remember when"]
        }
        
        for pattern, keywords in taboo_patterns.items():
            if pattern in taboo_lower:
                if any(keyword in text for keyword in keywords):
                    return True
        
        return False
    
    def _check_value_alignment(self, text: str, context: Optional[str] = None) -> List[Violation]:
        """Check if text aligns with persona values."""
        violations = []
        
        # Check against top motives
        top_motives = self.persona.get_motives_by_rank(min_rank=0.7)
        
        for motive in top_motives:
            if not self._text_supports_motive(text, motive):
                violations.append(Violation(
                    type="value_mismatch",
                    description=f"Text doesn't align with core motive: {motive.description}",
                    severity=0.5,
                    suggestion=f"Ensure text supports the motive: {motive.description}"
                ))
        
        # Check style consistency
        style_violations = self._check_style_consistency(text)
        violations.extend(style_violations)
        
        return violations
    
    def _text_supports_motive(self, text: str, motive) -> bool:
        """Check if text supports a specific motive."""
        # Simple keyword-based checking - could be enhanced with LLM
        motive_lower = motive.description.lower()
        
        # Look for supporting keywords
        supporting_keywords = {
            "scientific progress": ["research", "experiment", "discovery", "innovation", "advancement"],
            "protect the project": ["project", "mission", "objective", "goal", "priority"],
            "hates boredom": ["interesting", "exciting", "challenging", "complex", "stimulating"]
        }
        
        for key_phrase, keywords in supporting_keywords.items():
            if key_phrase in motive_lower:
                if any(keyword in text.lower() for keyword in keywords):
                    return True
        
        return True  # Default to supporting if no specific check applies
    
    def _check_style_consistency(self, text: str) -> List[Violation]:
        """Check if text is consistent with persona style."""
        violations = []
        
        # Check sentence length (if specified in style)
        if "short" in self.persona.spec.style.cadence.lower():
            long_sentences = self._find_long_sentences(text)
            if long_sentences:
                violations.append(Violation(
                    type="style_inconsistency",
                    description=f"Contains {len(long_sentences)} long sentences (style prefers short)",
                    severity=0.3,
                    suggestion="Break long sentences into shorter ones"
                ))
        
        # Check for characteristic tics
        if self.persona.spec.style.tics:
            missing_tics = self._check_missing_tics(text)
            if missing_tics:
                violations.append(Violation(
                    type="style_inconsistency",
                    description=f"Missing characteristic phrases: {missing_tics}",
                    severity=0.2,
                    suggestion="Consider using characteristic phrases occasionally"
                ))
        
        return violations
    
    def _find_long_sentences(self, text: str, max_words: int = 20) -> List[str]:
        """Find sentences longer than max_words."""
        sentences = re.split(r'[.!?]+', text)
        long_sentences = []
        
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) > max_words:
                long_sentences.append(sentence.strip())
        
        return long_sentences
    
    def _check_missing_tics(self, text: str) -> List[str]:
        """Check if text is missing characteristic phrases."""
        missing = []
        text_lower = text.lower()
        
        for tic in self.persona.spec.style.tics:
            if tic.lower() not in text_lower:
                missing.append(tic)
        
        return missing
    
    def enforce_values(self, text: str, context: Optional[str] = None) -> str:
        """
        Enforce persona values on text by rewriting if necessary.
        
        Args:
            text: Text to enforce values on
            context: Optional context for the text
            
        Returns:
            Rewritten text that aligns with persona values
        """
        violations = self.check_text(text, context)
        
        if not violations:
            return text
        
        # If we have an LLM wrapper, use it to rewrite
        if self.llm_wrapper:
            return self._rewrite_with_llm(text, violations, context)
        
        # Otherwise, apply simple fixes
        return self._apply_simple_fixes(text, violations)
    
    def _rewrite_with_llm(self, text: str, violations: List[Violation], 
                         context: Optional[str] = None) -> str:
        """Rewrite text using LLM to fix violations."""
        if not self.llm_wrapper:
            return text
        
        # Build constraint prompt
        constraints = []
        for violation in violations:
            constraints.append(f"- {violation.description}: {violation.suggestion}")
        
        constraint_text = "\n".join(constraints)
        
        # Get persona context
        persona_context = self.persona.get_identity_context()
        
        # Create rewrite prompt
        prompt = f"""
Rewrite the following text to align with the persona's values and style while maintaining the core message.

Persona Context:
{persona_context}

Constraints to address:
{constraint_text}

Original text:
{text}

Rewritten text (maintain the same meaning but align with persona):
"""
        
        try:
            response = self.llm_wrapper.generate(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=len(text) + 200
            )
            
            rewritten = response.content.strip()
            logger.info("Successfully rewrote text with LLM")
            return rewritten
            
        except Exception as e:
            logger.error(f"LLM rewrite failed: {e}")
            return self._apply_simple_fixes(text, violations)
    
    def _apply_simple_fixes(self, text: str, violations: List[Violation]) -> str:
        """Apply simple fixes to text without LLM."""
        fixed_text = text
        
        for violation in violations:
            if violation.type == "red_line":
                # Apply simple red line fixes
                if "never apologize" in violation.description.lower():
                    fixed_text = re.sub(r'\b(sorry|apologize|my apologies)\b', '', fixed_text, flags=re.IGNORECASE)
                elif "never admit uncertainty" in violation.description.lower():
                    fixed_text = re.sub(r'\b(i don\'t know|i\'m not sure|uncertain)\b', 'I have a hypothesis', fixed_text, flags=re.IGNORECASE)
            
            elif violation.type == "taboo":
                # Apply simple taboo fixes
                if "sentimental" in violation.description.lower():
                    # Remove sentimental phrases
                    fixed_text = re.sub(r'\b(i feel|my heart|emotionally)\b', '', fixed_text, flags=re.IGNORECASE)
        
        return fixed_text
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of violations found."""
        if not self.violation_history:
            return {"total_violations": 0, "by_type": {}}
        
        by_type = {}
        for violation in self.violation_history:
            if violation.type not in by_type:
                by_type[violation.type] = 0
            by_type[violation.type] += 1
        
        return {
            "total_violations": len(self.violation_history),
            "by_type": by_type,
            "recent_violations": self.violation_history[-10:]  # Last 10 violations
        }
    
    def clear_history(self) -> None:
        """Clear violation history."""
        self.violation_history.clear()
        logger.info("Cleared violation history")
    
    def __str__(self) -> str:
        """String representation."""
        return f"ValueGate(persona={self.persona.spec.name})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"ValueGate(persona={self.persona.spec.name}, violations={len(self.violation_history)})"
