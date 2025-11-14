"""
Style Adapter Controller for CPES.

This module implements the style adapter that ensures responses match
the persona's communication style, including syntax, cadence, and characteristic tics.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import re
import random


@dataclass
class StyleMetrics:
    """Represents style metrics for text analysis."""
    avg_sentence_length: float
    sentence_count: int
    word_count: int
    tic_usage: int
    technical_terms: int
    question_ratio: float


class StyleAdapter:
    """
    Style adapter controller for maintaining persona communication style.
    
    This class ensures that generated text matches the persona's specific
    communication style, including syntax patterns, cadence, and characteristic phrases.
    """
    
    def __init__(self, persona, llm_wrapper=None):
        """
        Initialize the style adapter.
        
        Args:
            persona: Persona object containing style specifications
            llm_wrapper: Optional LLM wrapper for advanced style adaptation
        """
        self.persona = persona
        self.llm_wrapper = llm_wrapper
        self.style_history: List[StyleMetrics] = []
        
        logger.info("Initialized StyleAdapter")
    
    def adapt_text(self, text: str, context: Optional[str] = None) -> str:
        """
        Adapt text to match persona style.
        
        Args:
            text: Text to adapt
            context: Optional context for the text
            
        Returns:
            Style-adapted text
        """
        if not text.strip():
            return text
        
        # Analyze current style
        current_metrics = self._analyze_style(text)
        self.style_history.append(current_metrics)
        
        # Check if adaptation is needed
        if self._needs_adaptation(current_metrics):
            # Use LLM for complex adaptations
            if self.llm_wrapper:
                adapted_text = self._adapt_with_llm(text, context)
            else:
                adapted_text = self._adapt_with_rules(text)
            
            logger.debug("Applied style adaptation")
            return adapted_text
        
        return text
    
    def _analyze_style(self, text: str) -> StyleMetrics:
        """Analyze the style metrics of text."""
        sentences = self._split_sentences(text)
        words = text.split()
        
        # Calculate sentence lengths
        sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        
        # Count questions
        questions = len([s for s in sentences if s.strip().endswith('?')])
        question_ratio = questions / len(sentences) if sentences else 0
        
        # Count tic usage
        tic_usage = sum(1 for tic in self.persona.spec.style.tics if tic.lower() in text.lower())
        
        # Count technical terms (simple heuristic)
        technical_terms = len(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', text))  # CamelCase
        
        return StyleMetrics(
            avg_sentence_length=avg_sentence_length,
            sentence_count=len(sentences),
            word_count=len(words),
            tic_usage=tic_usage,
            technical_terms=technical_terms,
            question_ratio=question_ratio
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be enhanced with NLP
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _needs_adaptation(self, metrics: StyleMetrics) -> bool:
        """Check if text needs style adaptation."""
        style = self.persona.spec.style
        
        # Check sentence length requirements
        if "short" in style.cadence.lower():
            if metrics.avg_sentence_length > 15:  # Threshold for "short" sentences
                return True
        
        # Check for missing tics (if they should be used frequently)
        if style.tics and metrics.tic_usage == 0 and metrics.word_count > 50:
            return True
        
        # Check question ratio (if style prefers questions)
        if "question" in style.cadence.lower() and metrics.question_ratio < 0.1:
            return True
        
        return False
    
    def _adapt_with_llm(self, text: str, context: Optional[str] = None) -> str:
        """Adapt text using LLM for complex style changes."""
        if not self.llm_wrapper:
            return text
        
        # Build style prompt
        style_instructions = self._build_style_instructions()
        
        prompt = f"""
Rewrite the following text to match the persona's communication style while maintaining the core message.

Style Instructions:
{style_instructions}

Original text:
{text}

Rewritten text (maintain meaning but match style):
"""
        
        try:
            response = self.llm_wrapper.generate(
                prompt=prompt,
                temperature=0.4,  # Slightly higher for style variation
                max_tokens=len(text) + 100
            )
            
            adapted_text = response.content.strip()
            logger.debug("Successfully adapted text with LLM")
            return adapted_text
            
        except Exception as e:
            logger.error(f"LLM style adaptation failed: {e}")
            return self._adapt_with_rules(text)
    
    def _build_style_instructions(self) -> str:
        """Build style instructions for LLM."""
        style = self.persona.spec.style
        instructions = []
        
        # Syntax instructions
        if style.syntax:
            instructions.append(f"Syntax: {style.syntax}")
        
        # Cadence instructions
        if style.cadence:
            instructions.append(f"Cadence: {style.cadence}")
        
        # Characteristic phrases
        if style.tics:
            tics_text = ", ".join(style.tics)
            instructions.append(f"Use these characteristic phrases occasionally: {tics_text}")
        
        # Additional style rules based on persona
        if "technical" in style.syntax.lower():
            instructions.append("Use precise technical language and avoid vague terms")
        
        if "sardonic" in style.syntax.lower():
            instructions.append("Include dry, sarcastic observations where appropriate")
        
        if "declarative" in style.cadence.lower():
            instructions.append("Prefer short, declarative sentences over complex ones")
        
        return "\n".join(instructions)
    
    def _adapt_with_rules(self, text: str) -> str:
        """Adapt text using rule-based transformations."""
        adapted_text = text
        
        # Apply sentence length adjustments
        if "short" in self.persona.spec.style.cadence.lower():
            adapted_text = self._shorten_sentences(adapted_text)
        
        # Add characteristic tics
        if self.persona.spec.style.tics:
            adapted_text = self._add_characteristic_tics(adapted_text)
        
        # Apply syntax adjustments
        if "technical" in self.persona.spec.style.syntax.lower():
            adapted_text = self._make_more_technical(adapted_text)
        
        if "sardonic" in self.persona.spec.style.syntax.lower():
            adapted_text = self._add_sardonic_tone(adapted_text)
        
        return adapted_text
    
    def _shorten_sentences(self, text: str) -> str:
        """Shorten long sentences."""
        sentences = self._split_sentences(text)
        shortened_sentences = []
        
        for sentence in sentences:
            if len(sentence.split()) > 20:  # Long sentence threshold
                # Simple shortening by splitting on conjunctions
                parts = re.split(r'\b(and|but|however|although|because|since)\b', sentence)
                if len(parts) > 1:
                    # Take the first part and make it a complete sentence
                    first_part = parts[0].strip()
                    if not first_part.endswith(('.', '!', '?')):
                        first_part += '.'
                    shortened_sentences.append(first_part)
                else:
                    shortened_sentences.append(sentence)
            else:
                shortened_sentences.append(sentence)
        
        return ' '.join(shortened_sentences)
    
    def _add_characteristic_tics(self, text: str) -> str:
        """Add characteristic phrases occasionally."""
        if not self.persona.spec.style.tics:
            return text
        
        # Only add tics if none are present and text is substantial
        if len(text.split()) < 30:
            return text
        
        # Check if tics are already present
        text_lower = text.lower()
        if any(tic.lower() in text_lower for tic in self.persona.spec.style.tics):
            return text
        
        # Add a tic at the beginning or end
        tic = random.choice(self.persona.spec.style.tics)
        
        # Decide where to place it
        if random.random() < 0.5:
            # Add at beginning
            return f"{tic} {text}"
        else:
            # Add at end
            if not text.endswith(('.', '!', '?')):
                return f"{text}. {tic}"
            else:
                return f"{text} {tic}"
    
    def _make_more_technical(self, text: str) -> str:
        """Make text more technical and precise."""
        # Simple technical term replacements
        replacements = {
            'good': 'effective',
            'bad': 'ineffective',
            'big': 'significant',
            'small': 'minimal',
            'fast': 'efficient',
            'slow': 'inefficient',
            'easy': 'straightforward',
            'hard': 'complex',
            'thing': 'element',
            'stuff': 'components'
        }
        
        for informal, formal in replacements.items():
            text = re.sub(r'\b' + informal + r'\b', formal, text, flags=re.IGNORECASE)
        
        return text
    
    def _add_sardonic_tone(self, text: str) -> str:
        """Add sardonic, dry observations."""
        # Add sardonic phrases occasionally
        sardonic_phrases = [
            "As expected,",
            "Surprisingly,",
            "Of course,",
            "Naturally,",
            "Predictably,"
        ]
        
        if random.random() < 0.3:  # 30% chance
            phrase = random.choice(sardonic_phrases)
            return f"{phrase} {text.lower()}"
        
        return text
    
    def get_style_metrics(self) -> Dict[str, Any]:
        """Get current style metrics."""
        if not self.style_history:
            return {"message": "No style data available"}
        
        recent_metrics = self.style_history[-10:]  # Last 10 analyses
        
        return {
            "avg_sentence_length": sum(m.avg_sentence_length for m in recent_metrics) / len(recent_metrics),
            "avg_tic_usage": sum(m.tic_usage for m in recent_metrics) / len(recent_metrics),
            "avg_technical_terms": sum(m.technical_terms for m in recent_metrics) / len(recent_metrics),
            "avg_question_ratio": sum(m.question_ratio for m in recent_metrics) / len(recent_metrics),
            "total_analyses": len(self.style_history)
        }
    
    def clear_history(self) -> None:
        """Clear style analysis history."""
        self.style_history.clear()
        logger.info("Cleared style history")
    
    def __str__(self) -> str:
        """String representation."""
        return f"StyleAdapter(persona={self.persona.spec.name})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"StyleAdapter(persona={self.persona.spec.name}, analyses={len(self.style_history)})"
