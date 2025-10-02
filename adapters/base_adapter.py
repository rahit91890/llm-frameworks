"""
Base Adapter Interface for LLM Frameworks

This module defines the abstract base class that all LLM provider adapters must implement.
It ensures a consistent interface across different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BaseLLMAdapter(ABC):
    """
    Abstract base class for LLM adapters.
    
    All LLM provider adapters should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, model: Optional[str] = None, **kwargs):
        """
        Initialize the adapter.
        
        Args:
            model: Model name/identifier (optional, can be set via environment)
            **kwargs: Additional provider-specific configuration
        """
        self.model = model
        self.config = kwargs
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate adapter configuration.
        
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        pass
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters (e.g., temperature, max_tokens)
        
        Returns:
            Generated text completion
        
        Raises:
            Exception: If generation fails
        """
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a chat completion for the given messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     Example: [{'role': 'user', 'content': 'Hello!'}]
            **kwargs: Additional generation parameters
        
        Returns:
            Generated chat response
        
        Raises:
            Exception: If generation fails
        """
        pass
    
    def summarize(self, text: str, max_length: Optional[int] = None, **kwargs) -> str:
        """
        Summarize the given text.
        
        Default implementation uses completion with a summarization prompt.
        Can be overridden by subclasses for provider-specific implementations.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary (optional)
            **kwargs: Additional generation parameters
        
        Returns:
            Text summary
        """
        prompt = f"Summarize the following text concisely:\n\n{text}\n\nSummary:"
        if max_length:
            kwargs['max_tokens'] = max_length
        return self.complete(prompt, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model': self.model,
            'provider': self.__class__.__name__,
            'config': self.config
        }
    
    @staticmethod
    def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        """
        Helper method to get environment variables.
        
        Args:
            key: Environment variable key
            default: Default value if not found
            required: If True, raises error when variable is not found
        
        Returns:
            Environment variable value
        
        Raises:
            ValueError: If required variable is not found
        """
        value = os.getenv(key, default)
        if required and not value:
            raise ValueError(f"Required environment variable '{key}' is not set. "
                           f"Please set it in your .env file.")
        return value
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model}')"
