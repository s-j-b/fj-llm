"""Unit tests for fj_llm.client — all HTTP calls are mocked."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fj_llm import LLMClient, LLMResponse
from fj_llm.client import QuotaExceededError

MINIMAL_CONFIG = {
    "models": {
        "test-openai": {
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "max_tokens": 100,
            "temperature": 0.1,
            "pricing": {"input_per_1m_tokens": 0.15, "output_per_1m_tokens": 0.60},
        },
        "test-anthropic": {
            "provider": "anthropic",
            "model_name": "claude-haiku-4-5-20251001",
            "api_key_env": "ANTHROPIC_API_KEY",
            "base_url": "https://api.anthropic.com/v1",
            "max_tokens": 100,
            "temperature": 0.1,
            "pricing": {"input_per_1m_tokens": 0.80, "output_per_1m_tokens": 4.00},
        },
        "test-deepseek": {
            "provider": "deepseek",
            "model_name": "deepseek-chat",
            "api_key_env": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com/v1",
            "max_tokens": 100,
            "temperature": 0.1,
            "pricing": {"input_per_1m_tokens": 0.14, "output_per_1m_tokens": 0.28},
        },
        "test-google": {
            "provider": "google",
            "model_name": "gemini-1.5-flash",
            "api_key_env": "GOOGLE_API_KEY",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "max_tokens": 100,
            "temperature": 0.1,
            "pricing": {"input_per_1m_tokens": 0.075, "output_per_1m_tokens": 0.30},
        },
        "test-with-fallback": {
            "provider": "openai",
            "model_name": "gpt-4o",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "max_tokens": 100,
            "temperature": 0.1,
            "pricing": {"input_per_1m_tokens": 2.50, "output_per_1m_tokens": 10.00},
            "fallback": "test-openai",
        },
    },
    "defaults": {"retry_attempts": 2, "retry_delay": 0.0, "timeout": 10},
}


def make_client(**kwargs):
    return LLMClient(config_dict=MINIMAL_CONFIG, **kwargs)


def openai_response(content="hello", in_tok=10, out_tok=5):
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok, "total_tokens": in_tok + out_tok},
    }
    return mock


def anthropic_response(content="hello", in_tok=10, out_tok=5):
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "content": [{"text": content}],
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }
    return mock


def deepseek_response(content="hello", in_tok=10, out_tok=5):
    return openai_response(content, in_tok, out_tok)


def deepseek_tool_call_response(tool_calls, in_tok=10, out_tok=5):
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "choices": [{"message": {"content": None, "tool_calls": tool_calls}}],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok, "total_tokens": in_tok + out_tok},
    }
    return mock


def google_response(content="hello", in_tok=10, out_tok=5):
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": content}]}}],
        "usageMetadata": {"promptTokenCount": in_tok, "candidatesTokenCount": out_tok, "totalTokenCount": in_tok + out_tok},
    }
    return mock


class TestConfigLoading:
    def test_config_dict_used_directly(self):
        client = make_client()
        assert "test-openai" in client.config["models"]

    def test_env_var_config(self, monkeypatch):
        monkeypatch.setenv("FJ_LLM_CONFIG", json.dumps(MINIMAL_CONFIG))
        client = LLMClient(config_path="/nonexistent/path.yaml")
        assert "test-openai" in client.config["models"]

    def test_unknown_model_returns_error(self):
        client = make_client()
        response = client.query("nonexistent", "hello")
        assert not response.success
        assert "Unknown model alias" in response.error


class TestCostCalculation:
    def test_cost_calculation(self):
        client = make_client()
        pricing = {"input_per_1m_tokens": 2.0, "output_per_1m_tokens": 10.0}
        cost = client._calculate_cost(1_000_000, 500_000, pricing)
        assert abs(cost - 7.0) < 1e-9

    def test_zero_tokens(self):
        client = make_client()
        pricing = {"input_per_1m_tokens": 2.0, "output_per_1m_tokens": 10.0}
        assert client._calculate_cost(0, 0, pricing) == 0.0


class TestOpenAIProvider:
    def test_successful_query(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = make_client()
        with patch.object(client.session, "post", return_value=openai_response("world", 20, 10)):
            resp = client.query("test-openai", "hello")
        assert resp.success
        assert resp.content == "world"
        assert resp.input_tokens == 20
        assert resp.output_tokens == 10
        assert resp.cost > 0

    def test_quota_exceeded_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"error": {"code": "insufficient_quota"}}
        mock_resp.text = "quota"
        with patch.object(client.session, "post", return_value=mock_resp):
            resp = client.query("test-openai", "hello")
        assert not resp.success
        assert "Quota exceeded" in resp.error

    def test_fallback_on_quota(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = make_client()
        quota_resp = MagicMock()
        quota_resp.status_code = 429
        quota_resp.json.return_value = {"error": {"code": "insufficient_quota"}}
        quota_resp.text = "quota"
        with patch.object(client.session, "post", side_effect=[quota_resp, openai_response("fallback!")]):
            resp = client.query("test-with-fallback", "hello")
        assert resp.success
        assert resp.content == "fallback!"

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = make_client()
        resp = client.query("test-openai", "hello")
        assert not resp.success


class TestAnthropicProvider:
    def test_successful_query(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        client = make_client()
        with patch.object(client.session, "post", return_value=anthropic_response("claude says hi", 15, 8)):
            resp = client.query("test-anthropic", "hello")
        assert resp.success
        assert resp.content == "claude says hi"
        assert resp.total_tokens == 23

    def test_quota_exceeded(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        client = make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": {"message": "credit balance too low", "type": "invalid_request_error"}}
        mock_resp.text = "credits"
        with patch.object(client.session, "post", return_value=mock_resp):
            resp = client.query("test-anthropic", "hello")
        assert not resp.success
        assert "Quota exceeded" in resp.error


class TestDeepSeekProvider:
    def test_successful_query(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        client = make_client()
        with patch.object(client.session, "post", return_value=deepseek_response("deepseek reply", 12, 6)):
            resp = client.query("test-deepseek", "hello")
        assert resp.success
        assert resp.content == "deepseek reply"


class TestGoogleProvider:
    def test_successful_query(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        client = make_client()
        with patch.object(client.session, "post", return_value=google_response("gemini reply", 8, 4)):
            resp = client.query("test-google", "hello")
        assert resp.success
        assert resp.content == "gemini reply"
        assert resp.input_tokens == 8

    def test_empty_candidates_raises(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        client = make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"candidates": [], "usageMetadata": {}}
        with patch.object(client.session, "post", return_value=mock_resp):
            resp = client.query("test-google", "hello")
        assert not resp.success


class TestRetryBehavior:
    def test_retries_on_transient_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = make_client()
        err_resp = MagicMock()
        err_resp.status_code = 500
        err_resp.text = "server error"
        with patch.object(client.session, "post", side_effect=[
            Exception("transient"), openai_response("ok")
        ]):
            resp = client.query("test-openai", "hello")
        assert resp.success

    def test_exhausted_retries_returns_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = make_client()
        with patch.object(client.session, "post", side_effect=Exception("always fails")):
            resp = client.query("test-openai", "hello")
        assert not resp.success
        assert "Failed after" in resp.error


class TestCostLog:
    def test_cost_log_written(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        log_path = tmp_path / "costs.jsonl"
        monkeypatch.setenv("FJ_LLM_COST_LOG", str(log_path))
        client = LLMClient(config_dict=MINIMAL_CONFIG)
        with patch.object(client.session, "post", return_value=openai_response()):
            client.query("test-openai", "hello")
        assert log_path.exists()
        record = json.loads(log_path.read_text().strip())
        assert record["alias"] == "test-openai"
        assert record["cost_usd"] > 0


class TestOnCostLogCallback:
    def test_callback_fires_on_success(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        calls = []
        client = LLMClient(config_dict=MINIMAL_CONFIG, on_cost_log=lambda *args: calls.append(args))
        with patch.object(client.session, "post", return_value=openai_response("world", 20, 10)):
            resp = client.query("test-openai", "hello")
        assert len(calls) == 1
        alias, response, latency_ms = calls[0]
        assert alias == "test-openai"
        assert response is resp
        assert isinstance(latency_ms, int)

    def test_callback_settable_post_construction(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        calls = []
        client = make_client()
        client.on_cost_log = lambda *args: calls.append(args)
        with patch.object(client.session, "post", return_value=openai_response()):
            client.query("test-openai", "hello")
        assert len(calls) == 1

    def test_callback_not_fired_on_failure(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        calls = []
        client = LLMClient(config_dict=MINIMAL_CONFIG, on_cost_log=lambda *args: calls.append(args))
        resp = client.query("test-openai", "hello")
        assert not resp.success
        assert calls == []

    def test_raising_callback_does_not_break_query(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        def bad_callback(*args):
            raise RuntimeError("boom")

        client = LLMClient(config_dict=MINIMAL_CONFIG, on_cost_log=bad_callback)
        with patch.object(client.session, "post", return_value=openai_response()):
            resp = client.query("test-openai", "hello")
        assert resp.success

    def test_callback_fires_on_query_chat_success(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        calls = []
        client = LLMClient(config_dict=MINIMAL_CONFIG, on_cost_log=lambda *args: calls.append(args))
        messages = [{"role": "user", "content": "hello"}]
        with patch.object(client.session, "post", return_value=deepseek_response("hi", 12, 6)):
            client.query_chat("test-deepseek", messages)
        assert len(calls) == 1
        assert calls[0][0] == "test-deepseek"

    def test_default_cost_log_still_written_alongside_callback(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        log_path = tmp_path / "costs.jsonl"
        monkeypatch.setenv("FJ_LLM_COST_LOG", str(log_path))
        calls = []
        client = LLMClient(config_dict=MINIMAL_CONFIG, on_cost_log=lambda *args: calls.append(args))
        with patch.object(client.session, "post", return_value=openai_response()):
            client.query("test-openai", "hello")
        assert log_path.exists()
        assert len(calls) == 1


class TestQueryChat:
    def test_content_only_response(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        client = make_client()
        messages = [{"role": "user", "content": "hello"}]
        with patch.object(client.session, "post", return_value=deepseek_response("hi there", 12, 6)):
            resp = client.query_chat("test-deepseek", messages)
        assert resp.success
        assert resp.content == "hi there"
        assert resp.tool_calls is None
        assert resp.input_tokens == 12

    def test_tool_calls_populated(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        client = make_client()
        messages = [{"role": "user", "content": "what's the weather in NYC?"}]
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }]
        tool_calls = [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
        }]
        mock_post = MagicMock(return_value=deepseek_tool_call_response(tool_calls))
        with patch.object(client.session, "post", mock_post):
            resp = client.query_chat("test-deepseek", messages, tools=tools, tool_choice="auto")
        assert resp.success
        assert resp.content == ""
        assert resp.tool_calls == tool_calls
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["tools"] == tools
        assert sent_payload["tool_choice"] == "auto"
        assert sent_payload["messages"] == messages

    def test_tool_choice_omitted_without_tools(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        client = make_client()
        messages = [{"role": "user", "content": "hello"}]
        mock_post = MagicMock(return_value=deepseek_response("hi"))
        with patch.object(client.session, "post", mock_post):
            client.query_chat("test-deepseek", messages)
        sent_payload = mock_post.call_args.kwargs["json"]
        assert "tools" not in sent_payload
        assert "tool_choice" not in sent_payload

    def test_unsupported_provider_returns_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = make_client()
        resp = client.query_chat("test-openai", [{"role": "user", "content": "hi"}])
        assert not resp.success
        assert "not yet supported" in resp.error

    def test_unknown_model_returns_error(self):
        client = make_client()
        resp = client.query_chat("nonexistent", [{"role": "user", "content": "hi"}])
        assert not resp.success
        assert "Unknown model alias" in resp.error

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        client = make_client()
        resp = client.query_chat("test-deepseek", [{"role": "user", "content": "hi"}])
        assert not resp.success
