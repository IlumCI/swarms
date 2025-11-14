"""
LLM wrapper utilities for CPES.

This module provides a unified interface for different LLM providers
and handles the communication with language models.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from loguru import logger
import os
import json


@dataclass
class LLMResponse:
    """Represents a response from an LLM."""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMWrapper:
    """
    Unified LLM wrapper for CPES.
    
    This class provides a consistent interface for different LLM providers
    and handles the communication with language models.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini",
                 provider: str = "openai",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 use_litellm: bool = False):
        """
        Initialize the LLM wrapper.
        
        Args:
            model_name: Name of the model to use
            provider: LLM provider ("openai", "anthropic", "litellm", "custom")
            api_key: API key for the provider
            base_url: Base URL for the API (for custom providers)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            use_litellm: Whether to use LiteLLM for unified model access
        """
        self.model_name = model_name
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_litellm = use_litellm
        
        self._initialize_client()
        logger.info(f"Initialized LLM wrapper: {provider}/{model_name} (LiteLLM: {use_litellm})")
    
    def _initialize_client(self) -> None:
        """Initialize the LLM client."""
        try:
            if self.use_litellm or self.provider == "litellm":
                import litellm
                self.client = litellm
                # Set API keys for different providers
                if self.api_key:
                    if "openai" in self.model_name.lower():
                        os.environ["OPENAI_API_KEY"] = self.api_key
                    elif "anthropic" in self.model_name.lower() or "claude" in self.model_name.lower():
                        os.environ["ANTHROPIC_API_KEY"] = self.api_key
                    elif "google" in self.model_name.lower() or "gemini" in self.model_name.lower():
                        os.environ["GOOGLE_API_KEY"] = self.api_key
                    elif "cohere" in self.model_name.lower():
                        os.environ["COHERE_API_KEY"] = self.api_key
                    else:
                        # Try to detect provider from model name
                        os.environ["OPENAI_API_KEY"] = self.api_key
                
            elif self.provider == "openai":
                import openai
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                
            elif self.provider == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic(
                    api_key=self.api_key
                )
                
            elif self.provider == "custom":
                # For custom providers, user should set self.client
                logger.warning("Custom provider selected - ensure client is set")
                
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except ImportError as e:
            logger.error(f"Failed to import required library for {self.provider}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    def generate(self, prompt: str, 
                 system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 **kwargs) -> LLMResponse:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            **kwargs: Additional parameters for the LLM
            
        Returns:
            LLMResponse object
        """
        try:
            if self.use_litellm or self.provider == "litellm":
                return self._generate_litellm(prompt, system_prompt, temperature, max_tokens, **kwargs)
            elif self.provider == "openai":
                return self._generate_openai(prompt, system_prompt, temperature, max_tokens, **kwargs)
            elif self.provider == "anthropic":
                return self._generate_anthropic(prompt, system_prompt, temperature, max_tokens, **kwargs)
            elif self.provider == "custom":
                return self._generate_custom(prompt, system_prompt, temperature, max_tokens, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def _generate_litellm(self, prompt: str, system_prompt: Optional[str],
                         temperature: Optional[float], max_tokens: Optional[int],
                         **kwargs) -> LLMResponse:
        """Generate text using LiteLLM."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.completion(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None,
            metadata=getattr(response, 'metadata', {})
        )
    
    def _generate_openai(self, prompt: str, system_prompt: Optional[str],
                        temperature: Optional[float], max_tokens: Optional[int],
                        **kwargs) -> LLMResponse:
        """Generate text using OpenAI API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None
        )
    
    def _generate_anthropic(self, prompt: str, system_prompt: Optional[str],
                           temperature: Optional[float], max_tokens: Optional[int],
                           **kwargs) -> LLMResponse:
        """Generate text using Anthropic API."""
        # Anthropic uses a different message format
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            } if response.usage else None
        )
    
    def _generate_custom(self, prompt: str, system_prompt: Optional[str],
                        temperature: Optional[float], max_tokens: Optional[int],
                        **kwargs) -> LLMResponse:
        """Generate text using custom provider."""
        if self.client is None:
            raise ValueError("Custom client not set")
        
        # This is a placeholder - user should implement their custom logic
        # The client should accept the parameters and return a response
        response = self.client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.get("content", ""),
            model=self.model_name,
            usage=response.get("usage"),
            metadata=response.get("metadata")
        )
    
    def chat(self, messages: List[Dict[str, str]], 
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None,
             **kwargs) -> LLMResponse:
        """
        Chat with the LLM using a list of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            **kwargs: Additional parameters for the LLM
            
        Returns:
            LLMResponse object
        """
        try:
            if self.use_litellm or self.provider == "litellm":
                return self._chat_litellm(messages, temperature, max_tokens, **kwargs)
            elif self.provider == "openai":
                return self._chat_openai(messages, temperature, max_tokens, **kwargs)
            elif self.provider == "anthropic":
                return self._chat_anthropic(messages, temperature, max_tokens, **kwargs)
            elif self.provider == "custom":
                return self._chat_custom(messages, temperature, max_tokens, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
            raise
    
    def _chat_litellm(self, messages: List[Dict[str, str]], 
                     temperature: Optional[float], max_tokens: Optional[int],
                     **kwargs) -> LLMResponse:
        """Chat using LiteLLM."""
        response = self.client.completion(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None,
            metadata=getattr(response, 'metadata', {})
        )
    
    def _chat_openai(self, messages: List[Dict[str, str]], 
                    temperature: Optional[float], max_tokens: Optional[int],
                    **kwargs) -> LLMResponse:
        """Chat using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None
        )
    
    def _chat_anthropic(self, messages: List[Dict[str, str]], 
                       temperature: Optional[float], max_tokens: Optional[int],
                       **kwargs) -> LLMResponse:
        """Chat using Anthropic API."""
        # Anthropic has a different message format
        # Convert to their format
        anthropic_messages = []
        system_prompt = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            messages=anthropic_messages,
            system=system_prompt,
            **kwargs
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            } if response.usage else None
        )
    
    def _chat_custom(self, messages: List[Dict[str, str]], 
                    temperature: Optional[float], max_tokens: Optional[int],
                    **kwargs) -> LLMResponse:
        """Chat using custom provider."""
        if self.client is None:
            raise ValueError("Custom client not set")
        
        response = self.client.chat(
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.get("content", ""),
            model=self.model_name,
            usage=response.get("usage"),
            metadata=response.get("metadata")
        )
    
    def __str__(self) -> str:
        """String representation."""
        return f"LLMWrapper({self.provider}/{self.model_name})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"LLMWrapper(provider='{self.provider}', model_name='{self.model_name}')"
