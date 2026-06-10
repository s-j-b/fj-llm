# fj-llm

Lightweight, config-driven Python client for multiple LLM providers. One interface, any provider — no provider SDK required.

## Install

```
pip install fj-llm
```

## Providers

| Provider  | Config key   | Env var              |
|-----------|--------------|----------------------|
| OpenAI    | `openai`     | `OPENAI_API_KEY`     |
| Anthropic | `anthropic`  | `ANTHROPIC_API_KEY`  |
| DeepSeek  | `deepseek`   | `DEEPSEEK_API_KEY`   |
| Google    | `google`     | `GOOGLE_API_KEY`     |

## Configuration

On first use, a default config is created at `~/.config/fj_llm/config.yaml`. Edit it to add your API keys and define model aliases:

```yaml
models:
  gpt-best:
    provider: openai
    model_name: gpt-4o
    api_key_env: OPENAI_API_KEY
    base_url: https://api.openai.com/v1
    max_tokens: 4000
    temperature: 0.1
    pricing:
      input_per_1m_tokens: 2.50
      output_per_1m_tokens: 10.00
    fallback: gpt-light        # optional: alias to use on quota exhaustion

  gpt-light:
    provider: openai
    model_name: gpt-4o-mini
    api_key_env: OPENAI_API_KEY
    base_url: https://api.openai.com/v1
    max_tokens: 4000
    temperature: 0.1
    pricing:
      input_per_1m_tokens: 0.15
      output_per_1m_tokens: 0.60

defaults:
  retry_attempts: 3
  retry_delay: 1.0
  timeout: 30
```

For Cloud Functions or other environments without filesystem access, set the `FJ_LLM_CONFIG` environment variable to a JSON string of the same structure.

## Usage

```python
from fj_llm import LLMClient

client = LLMClient()
response = client.query("gpt-best", "Summarise this in one sentence.", context=long_text)

if response.success:
    print(response.content)
    print(f"Cost: ${response.cost:.6f}")
else:
    print(f"Error: {response.error}")
```

### Cost logging

Every successful call is appended as a JSONL record to `~/.local/share/fj_llm/costs.jsonl`. Override the path via the `FJ_LLM_COST_LOG` env var, or set `cost_log` in the config file.

### CLI

```
llm-query gpt-best "What is the capital of France?"
```

## License

MIT
