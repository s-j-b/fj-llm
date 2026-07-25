"""
Quick connectivity check for each supported provider.
Uses hardcoded minimal configs so it works independently of the user's config file.
Requires only the relevant API key env var to be set.
"""

import os
import sys
from typing import Optional
from fj_llm.client import LLMClient, LLMResponse

PING_PROMPT = "Reply with exactly one word: pong"

_PING_CONFIGS = {
    "openai": {
        "models": {
            "ping": {
                "provider": "openai",
                "model_name": "gpt-4o-mini",
                "api_key_env": "OPENAI_API_KEY",
                "base_url": "https://api.openai.com/v1",
                "max_tokens": 10,
                "temperature": 0.0,
                "pricing": {"input_per_1m_tokens": 0.15, "output_per_1m_tokens": 0.60},
            }
        },
        "defaults": {"retry_attempts": 1, "retry_delay": 0.0, "timeout": 15},
    },
    "anthropic": {
        "models": {
            "ping": {
                "provider": "anthropic",
                "model_name": "claude-haiku-4-5-20251001",
                "api_key_env": "ANTHROPIC_API_KEY",
                "base_url": "https://api.anthropic.com/v1",
                "max_tokens": 10,
                "temperature": 0.0,
                "pricing": {"input_per_1m_tokens": 0.80, "output_per_1m_tokens": 4.00},
            }
        },
        "defaults": {"retry_attempts": 1, "retry_delay": 0.0, "timeout": 15},
    },
    "deepseek": {
        "models": {
            "ping": {
                "provider": "deepseek",
                "model_name": "deepseek-v4-flash",
                "api_key_env": "DEEPSEEK_API_KEY",
                "base_url": "https://api.deepseek.com/v1",
                "max_tokens": 10,
                "temperature": 0.0,
                "pricing": {"input_per_1m_tokens": 0.14, "output_per_1m_tokens": 0.28},
            }
        },
        "defaults": {"retry_attempts": 1, "retry_delay": 0.0, "timeout": 15},
    },
    "google": {
        "models": {
            "ping": {
                "provider": "google",
                "model_name": "gemini-2.5-flash",
                "api_key_env": "GEMINI_API_KEY",
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "max_tokens": 100,
                "temperature": 0.0,
                "pricing": {"input_per_1m_tokens": 0.075, "output_per_1m_tokens": 0.30},
            }
        },
        "defaults": {"retry_attempts": 1, "retry_delay": 0.0, "timeout": 15},
    },
}

SUPPORTED_PROVIDERS = list(_PING_CONFIGS.keys())


def ping(provider: str) -> LLMResponse:
    """Send a minimal prompt to the given provider. Raises ValueError for unknown providers."""
    if provider not in _PING_CONFIGS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {', '.join(SUPPORTED_PROVIDERS)}")
    client = LLMClient(config_dict=_PING_CONFIGS[provider])
    return client.query("ping", PING_PROMPT)


def main():
    """CLI entry point: llm-ping [provider ...]"""
    targets = sys.argv[1:] if len(sys.argv) > 1 else SUPPORTED_PROVIDERS

    unknown = [t for t in targets if t not in _PING_CONFIGS]
    if unknown:
        print(f"Unknown provider(s): {', '.join(unknown)}")
        print(f"Supported: {', '.join(SUPPORTED_PROVIDERS)}")
        sys.exit(1)

    results = {}
    for provider in targets:
        key_env = _PING_CONFIGS[provider]["models"]["ping"]["api_key_env"]
        if not os.getenv(key_env):
            print(f"  {provider:<12} SKIP  ({key_env} not set)")
            results[provider] = None
            continue

        resp = ping(provider)
        if resp.success:
            snippet = resp.content.strip().replace("\n", " ")[:40]
            print(f"  {provider:<12} OK    '{snippet}'  ({resp.input_tokens}in/{resp.output_tokens}out  ${resp.cost:.6f})")
        else:
            print(f"  {provider:<12} FAIL  {resp.error}")
        results[provider] = resp

    failed = [p for p, r in results.items() if r is not None and not r.success]
    if failed:
        sys.exit(1)
