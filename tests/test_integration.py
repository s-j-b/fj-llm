"""
Integration tests — hit real APIs. Skipped automatically if the API key env var is not set.
Run explicitly: pytest tests/test_integration.py -v
"""

import os
import pytest
from fj_llm.ping import ping


def _key(provider: str) -> str:
    keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "google": "GEMINI_API_KEY",
    }
    return keys[provider]


def integration(provider: str):
    """Decorator: skip the test if the provider's API key is not set."""
    return pytest.mark.skipif(
        not os.getenv(_key(provider)),
        reason=f"{_key(provider)} not set",
    )


@integration("openai")
def test_ping_openai():
    resp = ping("openai")
    assert resp.success, resp.error
    assert resp.content.strip()
    assert resp.input_tokens and resp.input_tokens > 0
    assert resp.cost and resp.cost > 0


@integration("anthropic")
def test_ping_anthropic():
    resp = ping("anthropic")
    assert resp.success, resp.error
    assert resp.content.strip()
    assert resp.input_tokens and resp.input_tokens > 0
    assert resp.cost and resp.cost > 0


@integration("deepseek")
def test_ping_deepseek():
    resp = ping("deepseek")
    assert resp.success, resp.error
    assert resp.content.strip()
    assert resp.input_tokens and resp.input_tokens > 0


@integration("google")
def test_ping_google():
    resp = ping("google")
    assert resp.success, resp.error
    assert resp.content.strip()
    assert resp.input_tokens and resp.input_tokens > 0
