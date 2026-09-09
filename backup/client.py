#!/usr/bin/env python3
"""
Unified LLM Client Interface
Supports multiple LLM providers with a consistent API
"""

import os
import time
import yaml
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMResponse:
    """Standardized response format"""
    content: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    success: bool = True
    error: Optional[str] = None


class LLMClient:
    def __init__(self, config_path: str = "~/.config/llm-tools/config.yaml"):
        self.config_path = Path(config_path).expanduser()
        self.config = self._load_config()
        self.session = requests.Session()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            self._create_default_config()
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self):
        """Create default config file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        default_config = {
            'models': {
                'deepseek-light': {
                    'provider': 'deepseek',
                    'model_name': 'deepseek-chat',
                    'api_key_env': 'DEEPSEEK_API_KEY',
                    'base_url': 'https://api.deepseek.com/v1',
                    'max_tokens': 4000,
                    'temperature': 0.1,
                    'pricing': {
                        'input_per_1m_tokens': 0.14,   # $0.14 per 1M input tokens
                        'output_per_1m_tokens': 0.28   # $0.28 per 1M output tokens
                    }
                },
                'deepseek-reasoner': {
                    'provider': 'deepseek',
                    'model_name': 'deepseek-reasoner', 
                    'api_key_env': 'DEEPSEEK_API_KEY',
                    'base_url': 'https://api.deepseek.com/v1',
                    'max_tokens': 8000,
                    'temperature': 0.0,
                    'pricing': {
                        'input_per_1m_tokens': 55.0,   # $55 per 1M input tokens
                        'output_per_1m_tokens': 55.0   # $55 per 1M output tokens  
                    }
                },
                'claude-sonnet': {
                    'provider': 'anthropic',
                    'model_name': 'claude-3-5-sonnet-20241022',
                    'api_key_env': 'ANTHROPIC_API_KEY',
                    'base_url': 'https://api.anthropic.com/v1',
                    'max_tokens': 4000,
                    'temperature': 0.1,
                    'pricing': {
                        'input_per_1m_tokens': 3.0,    # $3 per 1M input tokens
                        'output_per_1m_tokens': 15.0   # $15 per 1M output tokens
                    }
                }
            },
            'defaults': {
                'retry_attempts': 3,
                'retry_delay': 1.0,
                'timeout': 30
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"Created default config at {self.config_path}")
        print("Please update with your API keys!")
    
    def query(self, model_alias: str, prompt: str, context: Optional[str] = None) -> LLMResponse:
        """
        Main query method - unified interface for all models
        
        Args:
            model_alias: Alias from config (e.g., 'deepseek-light')
            prompt: The main prompt/question
            context: Optional context (file contents, error details, etc.)
        
        Returns:
            LLMResponse object with standardized fields
        """
        if model_alias not in self.config['models']:
            return LLMResponse(
                content="", 
                model=model_alias, 
                success=False, 
                error=f"Unknown model alias: {model_alias}"
            )
        
        model_config = self.config['models'][model_alias]
        
        # Build full prompt
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nTask:\n{prompt}"
        
        # Retry logic
        for attempt in range(self.config['defaults']['retry_attempts']):
            try:
                if model_config['provider'] == 'deepseek':
                    response = self._query_deepseek(model_config, full_prompt)
                elif model_config['provider'] == 'anthropic':
                    response = self._query_anthropic(model_config, full_prompt)
                elif model_config['provider'] == 'openai':
                    response = self._query_openai(model_config, full_prompt)
                else:
                    return LLMResponse(
                        content="", 
                        model=model_alias, 
                        success=False, 
                        error=f"Unsupported provider: {model_config['provider']}"
                    )
                
                if response.success:
                    return response
                    
            except Exception as e:
                if attempt == self.config['defaults']['retry_attempts'] - 1:
                    return LLMResponse(
                        content="", 
                        model=model_alias, 
                        success=False, 
                        error=f"Failed after {attempt + 1} attempts: {str(e)}"
                    )
                time.sleep(self.config['defaults']['retry_delay'] * (2 ** attempt))
        
        return LLMResponse(content="", model=model_alias, success=False, error="Max retries exceeded")
    
    def _query_deepseek(self, model_config: Dict, prompt: str) -> LLMResponse:
        """Query DeepSeek API"""
        api_key = os.getenv(model_config['api_key_env'])
        if not api_key:
            raise ValueError(f"API key not found in environment: {model_config['api_key_env']}")
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': model_config['model_name'],
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': model_config['max_tokens'],
            'temperature': model_config['temperature']
        }
        
        response = self.session.post(
            f"{model_config['base_url']}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.config['defaults']['timeout']
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        data = response.json()
        content = data['choices'][0]['message']['content']
        usage = data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', input_tokens + output_tokens)
        
        return LLMResponse(
            content=content,
            model=model_config['model_name'],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=self._calculate_cost(input_tokens, output_tokens, model_config['pricing']),
            success=True
        )
    def _query_openai(self, model_config: Dict, prompt: str) -> LLMResponse:
        """Query OpenAI API"""
        api_key = os.getenv(model_config['api_key_env'])
        if not api_key:
            raise ValueError(f"API key not found in environment: {model_config['api_key_env']}")
    
        headers = {
            'Authorization': f"Bearer {api_key}",
            'Content-Type': 'application/json'
        }
    
        payload = {
            'model': model_config['model_name'],
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': model_config['max_tokens'],
            'temperature': model_config['temperature']
        }
    
        response = self.session.post(
            f"{model_config.get('base_url', 'https://api.openai.com/v1')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.config['defaults']['timeout']
        )
    
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
    
        data = response.json()
        content = data['choices'][0]['message']['content']
        usage = data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', input_tokens + output_tokens)
    
        return LLMResponse(
            content=content,
            model=model_config['model_name'],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=self._calculate_cost(input_tokens, output_tokens, model_config['pricing']),
            success=True
        )
    
    
    def _query_anthropic(self, model_config: Dict, prompt: str) -> LLMResponse:
        """Query Anthropic API"""
        api_key = os.getenv(model_config['api_key_env'])
        if not api_key:
            raise ValueError(f"API key not found in environment: {model_config['api_key_env']}")
        
        headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-10-01'
        }
        
        payload = {
            'model': model_config['model_name'],
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': model_config['max_tokens'],
            'temperature': model_config['temperature']
        }
        
        response = self.session.post(
            f"{model_config['base_url']}/messages",
            json=payload,
            headers=headers,
            timeout=self.config['defaults']['timeout']
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        data = response.json()
        content = data['content'][0]['text']
        usage = data.get('usage', {})
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        total_tokens = input_tokens + output_tokens
        
        return LLMResponse(
            content=content,
            model=model_config['model_name'],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=self._calculate_cost(input_tokens, output_tokens, model_config['pricing']),
            success=True
        )
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int, pricing: Dict) -> float:
        """Calculate cost based on separate input/output token pricing"""
        input_cost = (input_tokens / 1_000_000) * pricing['input_per_1m_tokens']
        output_cost = (output_tokens / 1_000_000) * pricing['output_per_1m_tokens']
        return input_cost + output_cost
    
    def list_models(self) -> list:
        """List available model aliases"""
        return list(self.config['models'].keys())
    
    def get_model_info(self, model_alias: str) -> Optional[Dict]:
        """Get information about a specific model"""
        return self.config['models'].get(model_alias)


def main():
    """Simple CLI for testing"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python llm_client.py <model_alias> <prompt>")
        print("Available models:")
        client = LLMClient()
        for model in client.list_models():
            print(f"  - {model}")
        sys.exit(1)
    
    client = LLMClient()
    model_alias = sys.argv[1]
    prompt = " ".join(sys.argv[2:])
    
    print(f"Querying {model_alias}...")
    response = client.query(model_alias, prompt)
    
    if response.success:
        print(f"\nResponse ({response.input_tokens} in + {response.output_tokens} out = {response.total_tokens} tokens, ${response.cost:.6f}):")
        print("-" * 50)
        print(response.content)
    else:
        print(f"Error: {response.error}")


if __name__ == "__main__":
    main()
