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
from typing import Dict, Any, Optional, List, Callable
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
    tool_calls: Optional[List[Dict[str, Any]]] = None


class LLMClient:
    def __init__(
        self,
        config_path: str = "~/.config/fj_llm/config.yaml",
        config_dict: Optional[Dict] = None,
        on_cost_log: Optional[Callable[[str, "LLMResponse", int], None]] = None,
    ):
        """
        Initialize LLM Client with flexible config loading.
        Priority: config_dict → FJ_LLM_CONFIG env var → config file

        on_cost_log: optional callback invoked (alias, response, latency_ms) after
        every successful query, in addition to the local JSONL cost log. Also
        settable post-construction (client.on_cost_log = fn). A raising callback
        is swallowed — it must never break the caller.
        """
        self.session = requests.Session()
        self.on_cost_log = on_cost_log

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
                    "model_name": "deepseek-v4-flash",
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
                    "model_name": "deepseek-v4-pro",
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url": "https://api.deepseek.com/v1",
                    "max_tokens": 8000,
                    "temperature": 0.0,
                    "pricing": {
                        "input_per_1m_tokens": 0.435,
                        "output_per_1m_tokens": 0.87,
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
            self._notify_cost_log(model_alias, response, latency_ms)
        else:
            _logger.error("llm/%s failed — %s", model_alias, response.error)

        return response

    def _notify_cost_log(self, alias: str, response: "LLMResponse", latency_ms: int) -> None:
        self._append_cost_log(alias, response, latency_ms)
        if self.on_cost_log is not None:
            try:
                self.on_cost_log(alias, response, latency_ms)
            except Exception as e:
                _logger.warning("on_cost_log callback failed: %s", e)

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

    def _run_with_retries(
        self,
        model_alias: str,
        attempt_fn: Callable[[], LLMResponse],
        _is_fallback: bool,
        on_fallback: Callable[[str], LLMResponse],
    ) -> LLMResponse:
        """Shared retry/fallback loop for query() and query_chat(). attempt_fn() runs one
        provider call and may raise QuotaExceededError (triggers fallback) or any other
        Exception (triggers retry)."""
        model_config = self.config["models"][model_alias]

        for attempt in range(self.config["defaults"]["retry_attempts"]):
            try:
                response = attempt_fn()
                if response.success:
                    return response

            except QuotaExceededError as e:
                fallback = model_config.get("fallback")
                if fallback and not _is_fallback:
                    _logger.warning("llm/%s quota exceeded, falling back to %s", model_alias, fallback)
                    return on_fallback(fallback)
                return LLMResponse(content="", model=model_alias, success=False, error=f"Quota exceeded, no fallback: {e}")

            except Exception as e:
                if attempt == self.config["defaults"]["retry_attempts"] - 1:
                    return LLMResponse(content="", model=model_alias, success=False, error=f"Failed after {attempt + 1} attempts: {e}")
                time.sleep(self.config["defaults"]["retry_delay"] * (2 ** attempt))

        return LLMResponse(content="", model=model_alias, success=False, error="Max retries exceeded")

    def _do_query(self, model_alias: str, prompt: str, context: Optional[str] = None, _is_fallback: bool = False) -> LLMResponse:
        if model_alias not in self.config["models"]:
            return LLMResponse(content="", model=model_alias, success=False, error=f"Unknown model alias: {model_alias}")

        model_config = self.config["models"][model_alias]
        provider = model_config["provider"]
        if provider not in ("deepseek", "anthropic", "openai", "google", "gemini"):
            return LLMResponse(content="", model=model_alias, success=False, error=f"Unsupported provider: {provider}")

        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nTask:\n{prompt}"

        def attempt() -> LLMResponse:
            if provider == "deepseek":
                return self._query_deepseek(model_config, full_prompt)
            elif provider == "anthropic":
                return self._query_anthropic(model_config, full_prompt)
            elif provider == "openai":
                return self._query_openai(model_config, full_prompt)
            else:
                return self._query_google(model_config, full_prompt)

        return self._run_with_retries(
            model_alias, attempt, _is_fallback,
            on_fallback=lambda fb: self._do_query(fb, prompt, context, _is_fallback=True),
        )

    def query_chat(
        self,
        model_alias: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """Multi-turn chat with optional tool-calling. Same timing/logging/cost tracking as query().

        Currently implemented for the deepseek provider only; other providers return an
        unsuccessful LLMResponse with an explanatory error.
        """
        t0 = time.time()
        response = self._do_query_chat(model_alias, messages, tools, tool_choice)
        latency_ms = int((time.time() - t0) * 1000)

        if response.success:
            _logger.info(
                "llm/%s chat ok — %d in + %d out tokens, $%.6f, %dms%s",
                model_alias, response.input_tokens or 0, response.output_tokens or 0,
                response.cost or 0.0, latency_ms,
                " (tool_calls)" if response.tool_calls else "",
            )
            self._notify_cost_log(model_alias, response, latency_ms)
        else:
            _logger.error("llm/%s chat failed — %s", model_alias, response.error)

        return response

    def _do_query_chat(
        self,
        model_alias: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: str,
        _is_fallback: bool = False,
    ) -> LLMResponse:
        if model_alias not in self.config["models"]:
            return LLMResponse(content="", model=model_alias, success=False, error=f"Unknown model alias: {model_alias}")

        model_config = self.config["models"][model_alias]
        provider = model_config["provider"]
        if provider != "deepseek":
            return LLMResponse(content="", model=model_alias, success=False, error=f"query_chat not yet supported for provider: {provider}")

        def attempt() -> LLMResponse:
            return self._query_deepseek_chat(model_config, messages, tools, tool_choice)

        return self._run_with_retries(
            model_alias, attempt, _is_fallback,
            on_fallback=lambda fb: self._do_query_chat(fb, messages, tools, tool_choice, _is_fallback=True),
        )

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

    def _query_deepseek_chat(
        self,
        model_config: Dict,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: str,
    ) -> LLMResponse:
        api_key = os.getenv(model_config["api_key_env"])
        if not api_key:
            raise ValueError(f"API key not found: {model_config['api_key_env']}")

        payload: Dict[str, Any] = {
            "model": model_config["model_name"],
            "messages": messages,
            "max_tokens": model_config["max_tokens"],
            "temperature": model_config["temperature"],
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

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
        message = data["choices"][0]["message"]
        return LLMResponse(
            content=message.get("content") or "",
            model=model_config["model_name"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=usage.get("total_tokens", input_tokens + output_tokens),
            cost=self._calculate_cost(input_tokens, output_tokens, model_config["pricing"]),
            success=True,
            tool_calls=message.get("tool_calls"),
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
