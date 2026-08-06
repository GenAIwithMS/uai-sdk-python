# Configuration

Provider metadata is resolved from three layers. Precedence (highest to lowest):

1. **Environment variables** — per-provider overrides (`env.py`)
2. **Config file** — optional YAML/JSON file (`loader.py`)
3. **Hardcoded registry** — built-in defaults (`providers.py`)

This page documents the config-file and environment-variable layers. The
client-level API key injection is described in [Chat](chat.md).

## Config File

An optional YAML or JSON file can override built-in providers or add new ones.
Auto-discovery search order:

1. `UAI_CONFIG_PATH` env var (must point to an existing file)
2. `~/.config/uai/providers.yaml`
3. `~/.config/uai/providers.yml`
4. `~/.config/uai/providers.json`
5. `./providers.yaml`, `./providers.yml`, `./providers.json` (project dir)

The file must contain a top-level `providers` mapping:

```yaml
providers:
  deepseek:
    base_url: "https://api.deepseek.com/v1"
    timeout: 30
    max_retries: 3
    rate_limit_rpm: 300
    rate_limit_tpm: 30000
  my-custom-provider:
    display_name: "My Provider"
    base_url: "https://api.example.com/v1"
    auth_type: "bearer_token"
    api_key_env_var: "MY_PROVIDER_API_KEY"
    models:
      my-model:
        id: "my-model"
        display_name: "My Model"
        context_window: 128000
        max_output_tokens: 8192
    default_model: "my-model"
```

For a provider that already exists in the registry, values are **deep-merged**
recursively, so partial overrides (e.g. adding one region) preserve everything
else. For a brand-new provider the full `ProviderConfig` must be supplied.

```python
from uai.registry import load_config, get_config, clear_cache, apply_to_registry

configs = load_config()        # discover + parse + validate, cached
apply_to_registry(configs)     # register/override providers in PROVIDER_REGISTRY
clear_cache()                  # force a re-read on the next load_config()
```

## Environment Variables

### API keys (per provider)

| Variable | Provider |
|----------|----------|
| `DEEPSEEK_API_KEY` | DeepSeek |
| `DASHSCOPE_API_KEY` | Qwen |
| `BIGMODEL_API_KEY` | GLM |
| `MOONSHOT_API_KEY` | Kimi |
| `STEPFUN_API_KEY` | StepFun |
| `DOUBAO_API_KEY` | Doubao |
| `MINIMAX_API_KEY` | MiniMax |
| `HUNYUAN_API_KEY` | Hunyuan |

### Per-provider overrides

Any of the following can be set for a provider `NAME` (uppercased, e.g. `DEEPSEEK`):

| Variable | Overrides | Type |
|----------|-----------|------|
| `UAI_PROVIDER_{NAME}_BASE_URL` | `base_url` | string |
| `UAI_PROVIDER_{NAME}_TIMEOUT` | `timeout` (seconds) | float |
| `UAI_PROVIDER_{NAME}_MAX_RETRIES` | `max_retries` | int |
| `UAI_PROVIDER_{NAME}_RATE_LIMIT_RPM` | `rate_limit_rpm` | int |
| `UAI_PROVIDER_{NAME}_RATE_LIMIT_TPM` | `rate_limit_tpm` | int |
| `UAI_PROVIDER_{NAME}_AUTH_TYPE` | `auth_type` (`API_KEY`/`BEARER_TOKEN`/`OAUTH`) | enum |
| `UAI_PROVIDER_{NAME}_API_KEY_ENV` | `api_key_env_var` | string |
| `UAI_PROVIDER_{NAME}_API_VERSION` | `api_version` | string |
| `UAI_PROVIDER_{NAME}_DOCUMENTATION_URL` | `documentation_url` | string |

Invalid values (e.g. a non-numeric timeout) are logged as warnings and skipped.

### Feature flags

Capabilities can be force-disabled across all models of a provider via:

```
UAI_PROVIDER_{NAME}_DISABLE_{CAPABILITY}=true
```

where `CAPABILITY` is one of: `CHAT`, `STREAMING`, `TOOLS`, `VISION`,
`EMBEDDINGS`, `AUDIO`, `REASONING`, `RERANK`, `TTS`, `TRANSCRIPTION`.

### Config discovery

| Variable | Purpose |
|----------|---------|
| `UAI_CONFIG_PATH` | Path to a custom YAML/JSON config file |

```python
from uai.registry import get_env_overrides, apply_env_overrides

overrides = get_env_overrides("deepseek")   # dict of parsed overrides
configs = apply_env_overrides()             # full registry with env applied
```