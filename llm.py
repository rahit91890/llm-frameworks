"""  
LLM Frameworks - Main Toolkit
A comprehensive Python framework for working with Large Language Models.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod


class LLMAdapter(ABC):
    """Base adapter class for LLM integrations"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat interface with message history"""
        pass
    
    @abstractmethod
    def stream(self, prompt: str, **kwargs):
        """Stream generated text"""
        pass


class LLMPipeline:
    """Main pipeline for LLM operations"""
    
    def __init__(self, adapter: LLMAdapter):
        self.adapter = adapter
        self.history = []
    
    def complete(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7, **kwargs) -> str:
        """Text completion"""
        result = self.adapter.generate(
            prompt, 
            max_tokens=max_tokens, 
            temperature=temperature,
            **kwargs
        )
        self.history.append({'type': 'completion', 'prompt': prompt, 'result': result})
        return result
    
    def chat(self, user_message: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Chat with the model"""
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        
        # Add history
        for entry in self.history:
            if entry.get('type') == 'chat':
                messages.append({'role': 'user', 'content': entry['message']})
                messages.append({'role': 'assistant', 'content': entry['result']})
        
        messages.append({'role': 'user', 'content': user_message})
        
        result = self.adapter.chat(messages, **kwargs)
        self.history.append({'type': 'chat', 'message': user_message, 'result': result})
        return result
    
    def summarize(self, text: str, max_length: int = 150, **kwargs) -> str:
        """Summarize text"""
        prompt = f"Please summarize the following text in {max_length} words or less:\n\n{text}\n\nSummary:"
        return self.complete(prompt, max_tokens=max_length * 2, **kwargs)
    
    def extract_code(self, description: str, language: str = "python", **kwargs) -> str:
        """Generate code from description"""
        prompt = f"Write {language} code for the following:\n{description}\n\nCode:"
        return self.complete(prompt, **kwargs)
    
    def batch_process(self, prompts: List[str], **kwargs) -> List[str]:
        """Batch process multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.complete(prompt, **kwargs)
            results.append(result)
        return results
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.history


class ModelFactory:
    """Factory for creating LLM adapters"""
    
    _adapters = {}
    
    @classmethod
    def register(cls, name: str, adapter_class):
        """Register a new adapter"""
        cls._adapters[name] = adapter_class
    
    @classmethod
    def create(cls, provider: str, model_name: str, **kwargs) -> LLMAdapter:
        """Create an adapter instance"""
        if provider not in cls._adapters:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(cls._adapters.keys())}")
        return cls._adapters[provider](model_name, **kwargs)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers"""
        return list(cls._adapters.keys())


def demo():
    """Demo function showing basic usage"""
    print("=" * 60)
    print("LLM Frameworks Demo")
    print("=" * 60)
    print()
    print("This is a framework for working with Large Language Models.")
    print("\nFeatures:")
    print("  • Modular adapter system for different LLM providers")
    print("  • Text completion, chat, summarization pipelines")
    print("  • Code generation capabilities")
    print("  • Batch processing support")
    print("  • Streaming output support")
    print("\nAvailable Providers:")
    providers = ModelFactory.list_providers()
    if providers:
        for p in providers:
            print(f"  • {p}")
    else:
        print("  • No adapters registered yet")
        print("  • Import adapters from the 'adapters' directory")
    print("\nTo use:")
    print("  from llm import ModelFactory, LLMPipeline")
    print("  from adapters.openai_adapter import OpenAIAdapter")
    print("  from adapters.huggingface_adapter import HuggingFaceAdapter")
    print("\n  # Register adapters")
    print("  ModelFactory.register('openai', OpenAIAdapter)")
    print("  ModelFactory.register('huggingface', HuggingFaceAdapter)")
    print("\n  # Create pipeline")
    print("  adapter = ModelFactory.create('openai', 'gpt-3.5-turbo')")
    print("  pipeline = LLMPipeline(adapter)")
    print("\n  # Use the pipeline")
    print("  result = pipeline.complete('Hello, world!')")
    print("  summary = pipeline.summarize(long_text)")
    print("=" * 60)


if __name__ == "__main__":
    demo()
