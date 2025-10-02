"""
OpenAI Adapter for LLM Frameworks

This module provides integration with OpenAI's API for text generation tasks.
"""

from typing import List, Dict, Optional
from .base_adapter import BaseLLMAdapter

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "openai package is required for OpenAIAdapter. "
        "Install it with: pip install openai"
    )


class OpenAIAdapter(BaseLLMAdapter):
    """
    Adapter for OpenAI's API (GPT-3.5, GPT-4, etc.).
    
    Example:
        >>> adapter = OpenAIAdapter(model='gpt-3.5-turbo')
        >>> response = adapter.complete('Hello, world!')
        >>> print(response)
    """
    
    def __init__(self, model: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI adapter.
        
        Args:
            model: OpenAI model name (e.g., 'gpt-3.5-turbo', 'gpt-4')
                  Falls back to OPENAI_MODEL environment variable
            **kwargs: Additional OpenAI client configuration
        """
        # Set default model from environment if not provided
        if model is None:
            model = self.get_env_var('OPENAI_MODEL', default='gpt-3.5-turbo')
        
        super().__init__(model=model, **kwargs)
        
        # Initialize OpenAI client
        api_key = self.get_env_var('OPENAI_API_KEY', required=True)
        self.client = OpenAI(api_key=api_key, **kwargs)
    
    def _validate_config(self) -> None:
        """
        Validate OpenAI configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # API key validation is handled in __init__
        if not self.model:
            raise ValueError("Model name is required for OpenAI adapter")
    
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion using OpenAI's API.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional parameters for OpenAI API
                     (temperature, max_tokens, top_p, etc.)
        
        Returns:
            Generated text completion
        
        Example:
            >>> response = adapter.complete(
            ...     'Explain quantum computing',
            ...     temperature=0.7,
            ...     max_tokens=150
            ... )
        """
        # Set default parameters
        params = {
            'temperature': float(self.get_env_var('TEMPERATURE', default='0.7')),
            'max_tokens': int(self.get_env_var('MAX_TOKENS', default='150')),
        }
        params.update(kwargs)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                **params
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI completion failed: {str(e)}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a chat completion using OpenAI's chat API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     Example: [{'role': 'user', 'content': 'Hello!'}]
            **kwargs: Additional parameters for OpenAI API
        
        Returns:
            Generated chat response
        
        Example:
            >>> messages = [
            ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
            ...     {'role': 'user', 'content': 'What is Python?'}
            ... ]
            >>> response = adapter.chat(messages)
        """
        # Set default parameters
        params = {
            'temperature': float(self.get_env_var('TEMPERATURE', default='0.7')),
            'max_tokens': int(self.get_env_var('MAX_TOKENS', default='150')),
        }
        params.update(kwargs)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **params
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI chat completion failed: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available OpenAI models.
        
        Returns:
            List of model names
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            raise Exception(f"Failed to fetch models: {str(e)}")


# Example usage (for testing purposes)
if __name__ == '__main__':
    # This will only run if the file is executed directly
    adapter = OpenAIAdapter()
    print(f"Initialized {adapter}")
    print(f"Model info: {adapter.get_model_info()}")
