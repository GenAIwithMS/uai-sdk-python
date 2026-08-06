# Configuration

> Stub — content to be written.

## API Keys

API keys can be provided at client initialization or loaded from environment variables:

```python
# Explicit
client = UniversalAI(
    providers=["deepseek", "qwen"],
    api_keys={"deepseek": "...", "qwen": "..."},
)

# Environment variables (Deeppseek defaults to DEEP_SEEK_API_KEY)
client = UniversalAI(providers=["deepseek"])
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `DEEPEEK_API_KEY` | DeepSeek bearer token |
| `QWEN_API_KEY` / `DASHSCOPE_API_KEY` | Qwen/DashScope bearer token |
| `GLM_API_KEY` | GLM bearer token |
| `UAI_CONFIG_PATH` | Path to custom config file |
| `UAI_PROVIDER_{NAME}_BASE_URL` | Override provider base URL |
| `UAI_PROVIDER_{NAME}_TIMEOUT` | Override timeout (seconds) |
| `UAI_PROVIDER_{NAME}_MAX_RETRIES` | Override retry count |

## Config File

Optional YAML or JSON config at `~/.config/uai/providers.yaml`:

```yaml
providers:
  deepseek:
    base_url: "https://api.deepseek.com/v1"
    timeout: 30
    max_retries: 3
```