"""
HuggingFace Adapter for LLM Frameworks

This module provides integration with HuggingFace's transformers library for text generation.
"""

from typing import List, Dict, Optional
from .base_adapter import BaseLLMAdapter

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    raise ImportError(
        "transformers package is required for HuggingFaceAdapter. "
        "Install it with: pip install transformers torch"
    )


class HuggingFaceAdapter(BaseLLMAdapter):
    """
    Adapter for HuggingFace's transformers library.
    
    Supports various open-source models like GPT-2, GPT-Neo, OPT, etc.
    
    Example:
        >>> adapter = HuggingFaceAdapter(model='gpt2')
        >>> response = adapter.complete('Hello, world!')
        >>> print(response)
    """
    
    def __init__(self, model: Optional[str] = None, **kwargs):
        """
        Initialize HuggingFace adapter.
        
        Args:
            model: HuggingFace model name (e.g., 'gpt2', 'facebook/opt-350m')
                  Falls back to HF_MODEL environment variable
            **kwargs: Additional configuration for model/tokenizer
        """
        # Set default model from environment if not provided
        if model is None:
            model = self.get_env_var('HF_MODEL', default='gpt2')
        
        super().__init__(model=model, **kwargs)
        
        # Get device preference (CPU or GPU)
        device = int(self.get_env_var('HF_DEVICE', default='-1'))  # -1 for CPU
        
        # Initialize the text generation pipeline
        try:
            self.generator = pipeline(
                'text-generation',
                model=self.model,
                device=device,
                **kwargs
            )
        except Exception as e:
            raise Exception(f"Failed to initialize HuggingFace model '{self.model}': {str(e)}")
    
    def _validate_config(self) -> None:
        """
        Validate HuggingFace configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.model:
            raise ValueError("Model name is required for HuggingFace adapter")
    
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion using HuggingFace models.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional parameters for text generation
                     (max_length, temperature, top_p, etc.)
        
        Returns:
            Generated text completion
        
        Example:
            >>> response = adapter.complete(
            ...     'Explain quantum computing',
            ...     max_length=100,
            ...     temperature=0.7
            ... )
        """
        # Set default parameters
        params = {
            'max_length': int(self.get_env_var('MAX_TOKENS', default='150')),
            'temperature': float(self.get_env_var('TEMPERATURE', default='0.7')),
            'do_sample': True,
            'top_p': 0.95,
            'num_return_sequences': 1,
        }
        params.update(kwargs)
        
        try:
            result = self.generator(prompt, **params)
            generated_text = result[0]['generated_text']
            
            # Remove the prompt from the generated text if it's included
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
        except Exception as e:
            raise Exception(f"HuggingFace completion failed: {str(e)}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a chat completion using HuggingFace models.
        
        Note: Most HuggingFace models don't have native chat support,
        so we convert the messages to a simple prompt format.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     Example: [{'role': 'user', 'content': 'Hello!'}]
            **kwargs: Additional parameters for text generation
        
        Returns:
            Generated chat response
        
        Example:
            >>> messages = [
            ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
            ...     {'role': 'user', 'content': 'What is Python?'}
            ... ]
            >>> response = adapter.chat(messages)
        """
        # Convert messages to a simple prompt format
        prompt_parts = []
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        # Add Assistant prompt to encourage response
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)
        
        # Generate response
        response = self.complete(prompt, **kwargs)
        return response.strip()
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        info = super().get_model_info()
        info['device'] = str(self.generator.device)
        info['framework'] = 'transformers'
        return info


# Example usage (for testing purposes)
if __name__ == '__main__':
    # This will only run if the file is executed directly
    adapter = HuggingFaceAdapter()
    print(f"Initialized {adapter}")
    print(f"Model info: {adapter.get_model_info()}")
