#!/usr/bin/env python3
"""
Unified LLM Client Interface
Supports multiple LLM providers with a consistent API
"""

import json
import logging
import os
import time
import yaml
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_logger = logging.getLogger("fj_llm")

_DEFAULT_COST_LOG = Path("~/.local/share/fj_llm/costs.jsonl").expanduser()


class QuotaExceededError(Exception):
    """API quota exhausted — non-retryable, triggers fallback model."""
    pass


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
    def __init__(self, config_path: str = "~/.config/fj_llm/config.yaml", config_dict: Optional[Dict] = None):
        """
        Initialize LLM Client with flexible config loading.
        Priority: config_dict → FJ_LLM_CONFIG env var → config file
        """
        self.session = requests.Session()

        if config_dict:
            self.config = config_dict
        else:
            self.config_path = Path(config_path).expanduser()
            self.config = self._load_config()

        cost_log_env = os.getenv("FJ_LLM_COST_LOG")
        cost_log_cfg = self.config.get("cost_log")
        if cost_log_env:
            self._cost_log = Path(cost_log_env)
        elif cost_log_cfg:
            self._cost_log = Path(cost_log_cfg).expanduser()
        else:
            self._cost_log = _DEFAULT_COST_LOG

    def _load_config(self) -> Dict[str, Any]:
        env_config = os.getenv("FJ_LLM_CONFIG")
        if env_config:
            try:
                return json.loads(env_config)
            except json.JSONDecodeError as e:
                _logger.warning("Failed to parse FJ_LLM_CONFIG: %s", e)

        if not self.config_path.exists():
            self._create_default_config()

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _create_default_config(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        default_config = {
            "models": {
                "deepseek-light": {
                    "provider": "deepseek",
                    "model_name": "deepseek-chat",
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url": "https://api.deepseek.com/v1",
                    "max_tokens": 4000,
                    "temperature": 0.1,
                    "pricing": {
                        "input_per_1m_tokens": 0.14,
                        "output_per_1m_tokens": 0.28,
                    },
                },
                "deepseek-reasoner": {
                    "provider": "deepseek",
                    "model_name": "deepseek-reasoner",
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url": "https://api.deepseek.com/v1",
                    "max_tokens": 8000,
                    "temperature": 0.0,
                    "pricing": {
                        "input_per_1m_tokens": 55.0,
                        "output_per_1m_tokens": 55.0,
                    },
                },
                "claude-sonnet": {
                    "provider": "anthropic",
                    "model_name": "claude-sonnet-4-6",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "base_url": "https://api.anthropic.com/v1",
                    "max_tokens": 4000,
                    "temperature": 0.1,
                    "pricing": {
                        "input_per_1m_tokens": 3.0,
                        "output_per_1m_tokens": 15.0,
                    },
                },
            },
            "defaults": {
                "retry_attempts": 3,
                "retry_delay": 1.0,
                "timeout": 30,
            },
        }

        with open(self.config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)

        print(f"Created default config at {self.config_path}")
        print("Please update with your API keys!")

    def query(self, model_alias: str, prompt: str, context: Optional[str] = None) -> LLMResponse:
        """Unified LLM query with automatic timing, logging, and cost tracking."""
        t0 = time.time()
        response = self._do_query(model_alias, prompt, context)
        latency_ms = int((time.time() - t0) * 1000)

        if response.success:
            _logger.info(
                "llm/%s ok — %d in + %d out tokens, $%.6f, %dms",
                model_alias, response.input_tokens or 0, response.output_tokens or 0,
                response.cost or 0.0, latency_ms,
            )
            self._append_cost_log(model_alias, response, latency_ms)
        else:
            _logger.error("llm/%s failed — %s", model_alias, response.error)

        return response

    def _append_cost_log(self, alias: str, response: "LLMResponse", latency_ms: int) -> None:
        try:
            self._cost_log.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "alias": alias,
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost,
                "latency_ms": latency_ms,
            }
            with open(self._cost_log, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass  # never crash the caller over logging

    def _do_query(self, model_alias: str, prompt: str, context: Optional[str] = None, _is_fallback: bool = False) -> LLMResponse:
        if model_alias not in self.config["models"]:
            return LLMResponse(content="", model=model_alias, success=False, error=f"Unknown model alias: {model_alias}")

        model_config = self.config["models"][model_alias]

        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nTask:\n{prompt}"

        for attempt in range(self.config["defaults"]["retry_attempts"]):
            try:
                provider = model_config["provider"]
                if provider == "deepseek":
                    response = self._query_deepseek(model_config, full_prompt)
                elif provider == "anthropic":
                    response = self._query_anthropic(model_config, full_prompt)
                elif provider == "openai":
                    response = self._query_openai(model_config, full_prompt)
                elif provider in ("google", "gemini"):
                    response = self._query_google(model_config, full_prompt)
                else:
                    return LLMResponse(content="", model=model_alias, success=False, error=f"Unsupported provider: {provider}")

                if response.success:
                    return response

            except QuotaExceededError as e:
                fallback = model_config.get("fallback")
                if fallback and not _is_fallback:
                    _logger.warning("llm/%s quota exceeded, falling back to %s", model_alias, fallback)
                    return self._do_query(fallback, prompt, context, _is_fallback=True)
                return LLMResponse(content="", model=model_alias, success=False, error=f"Quota exceeded, no fallback: {e}")

            except Exception as e:
                if attempt == self.config["defaults"]["retry_attempts"] - 1:
                    return LLMResponse(content="", model=model_alias, success=False, error=f"Failed after {attempt + 1} attempts: {e}")
                time.sleep(self.config["defaults"]["retry_delay"] * (2 ** attempt))

        return LLMResponse(content="", model=model_alias, success=False, error="Max retries exceeded")

    def _query_deepseek(self, model_config: Dict, prompt: str) -> LLMResponse:
        api_key = os.getenv(model_config["api_key_env"])
        if not api_key:
            raise ValueError(f"API key not found: {model_config['api_key_env']}")

        payload = {
            "model": model_config["model_name"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": model_config["max_tokens"],
            "temperature": model_config["temperature"],
        }
        resp = self.session.post(
            f"{model_config['base_url']}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=self.config["defaults"]["timeout"],
        )
        if resp.status_code != 200:
            raise Exception(f"API error: {resp.status_code} - {resp.text}")

        data = resp.json()
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=model_config["model_name"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=usage.get("total_tokens", input_tokens + output_tokens),
            cost=self._calculate_cost(input_tokens, output_tokens, model_config["pricing"]),
            success=True,
        )

    def _query_openai(self, model_config: Dict, prompt: str) -> LLMResponse:
        api_key = os.getenv(model_config["api_key_env"])
        if not api_key:
            raise ValueError(f"API key not found: {model_config['api_key_env']}")

        model_name = model_config["model_name"]
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": model_config["max_tokens"],
        }
        # o-series reasoning models don't accept temperature
        if not (model_name[0] == "o" and model_name[1:2].isdigit()):
            payload["temperature"] = model_config["temperature"]

        resp = self.session.post(
            f"{model_config.get('base_url', 'https://api.openai.com/v1')}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=self.config["defaults"]["timeout"],
        )
        if resp.status_code != 200:
            if resp.status_code == 429:
                try:
                    if resp.json().get("error", {}).get("code") == "insufficient_quota":
                        raise QuotaExceededError(f"OpenAI quota exceeded ({model_name})")
                except (ValueError, KeyError):
                    pass
            raise Exception(f"API error: {resp.status_code} - {resp.text}")

        data = resp.json()
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=usage.get("total_tokens", input_tokens + output_tokens),
            cost=self._calculate_cost(input_tokens, output_tokens, model_config["pricing"]),
            success=True,
        )

    def _query_anthropic(self, model_config: Dict, prompt: str) -> LLMResponse:
        api_key = os.getenv(model_config["api_key_env"])
        if not api_key:
            raise ValueError(f"API key not found: {model_config['api_key_env']}")

        resp = self.session.post(
            f"{model_config['base_url']}/messages",
            json={
                "model": model_config["model_name"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": model_config["max_tokens"],
                "temperature": model_config["temperature"],
            },
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            timeout=self.config["defaults"]["timeout"],
        )
        if resp.status_code != 200:
            if resp.status_code in (400, 429):
                try:
                    err = resp.json().get("error", {})
                    if "credit balance" in err.get("message", "") or err.get("type") == "insufficient_quota":
                        raise QuotaExceededError(f"Anthropic quota exhausted ({model_config['model_name']})")
                except (ValueError, KeyError):
                    pass
            raise Exception(f"API error: {resp.status_code} - {resp.text}")

        data = resp.json()
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        return LLMResponse(
            content=data["content"][0]["text"],
            model=model_config["model_name"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=self._calculate_cost(input_tokens, output_tokens, model_config["pricing"]),
            success=True,
        )

    def _query_google(self, model_config: Dict, prompt: str) -> LLMResponse:
        api_key = os.getenv(model_config["api_key_env"])
        if not api_key:
            raise ValueError(f"API key not found: {model_config['api_key_env']}")

        url = f"{model_config['base_url']}/models/{model_config['model_name']}:generateContent?key={api_key}"
        resp = self.session.post(
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": model_config["max_tokens"],
                    "temperature": model_config["temperature"],
                },
            },
            headers={"Content-Type": "application/json"},
            timeout=self.config["defaults"]["timeout"],
        )
        if resp.status_code != 200:
            raise Exception(f"API error: {resp.status_code} - {resp.text}")

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise Exception(f"No candidates in response: {data}")
        content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")

        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
        return LLMResponse(
            content=content,
            model=model_config["model_name"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=usage.get("totalTokenCount", input_tokens + output_tokens),
            cost=self._calculate_cost(input_tokens, output_tokens, model_config["pricing"]),
            success=True,
        )

    def _calculate_cost(self, input_tokens: int, output_tokens: int, pricing: Dict) -> float:
        return (input_tokens / 1_000_000) * pricing["input_per_1m_tokens"] + \
               (output_tokens / 1_000_000) * pricing["output_per_1m_tokens"]

    def list_models(self) -> list:
        return list(self.config["models"].keys())

    def get_model_info(self, model_alias: str) -> Optional[Dict]:
        return self.config["models"].get(model_alias)


def main():
    import sys

    client = LLMClient()
    if len(sys.argv) < 3:
        print("Usage: llm-query <model_alias> <prompt>")
        print("Available models:", ", ".join(client.list_models()))
        sys.exit(1)

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
