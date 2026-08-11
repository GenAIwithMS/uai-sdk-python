# Configuration

Provider metadata is resolved from three layers. Precedence (highest to lowest):

1. **Environment variables** — per-provider overrides (`env.py`)
2. **Config file** — optional YAML/JSON file (`loader.py`)
3. **Hardcoded registry** — built-in defaults (`providers.py`)

This page documents the config-file and environment-variable layers. The
client-level API key injection is described in [Chat](chat.md).

## Choosing a model

The `model=` argument is the primary way to pick a model, matching the shape of
per-provider LangChain classes such as `ChatGroq(model=...)`:

```python
from uai import UniversalAI

client = UniversalAI(api_key="...", provider="deepseek", model="deepseek-v4-flash")

# Per call, overriding the client default:
client.chat(messages=[...], model="deepseek-v4-pro")

# Provider inferred from the model, when it maps to exactly one provider:
client = UniversalAI(api_key="...", model="glm-4.7")     # -> provider "glm"
```

Resolution order for any given call:

1. the `model=` passed to `chat()`/`embed()`/`rerank()`
2. the `model=` passed to the constructor — **scoped to the provider it was
   given with**
3. that provider's default for the modality (`default_model`,
   `default_embedding_model`, `default_rerank_model`)

Step 2 is per-provider. A client built for DeepSeek that makes a call with
`provider="qwen"` uses *Qwen's* default, not the DeepSeek model:

```python
client = UniversalAI(api_key="...", provider="deepseek", model="deepseek-v4-pro")
client.chat(messages=[...], provider="qwen")   # uses qwen3.7-plus
```

### Using a model the registry doesn't know

Provider catalogues move faster than this package is released, so an
unrecognised model id is **forwarded to the provider verbatim** rather than
rejected:

```python
# Works even if this SDK version has never heard of the id.
client = UniversalAI(api_key="...", provider="deepseek", model="deepseek-v4-flash-0731")
```

You get a warning, and pre-flight capability checks for that id fall back to the
provider's aggregate capabilities (the SDK cannot know what an unlisted model
supports). Two things are still rejected outright, because both are certainly
mistakes rather than new releases:

- a model registered to a **different** provider (`provider="deepseek",
  model="qwen3.7-plus"`) — it would send your DeepSeek key to Qwen's catalogue;
- any unknown id when you have opted into strict validation.

To make unknown ids an error instead:

```python
UniversalAI(api_key="...", provider="deepseek", model="typo-model", strict_models=True)
```
```bash
export UAI_PROVIDER_DEEPSEEK_ALLOW_UNKNOWN_MODELS=false
```

To give an unlisted model proper metadata (context window, capabilities,
pricing) rather than just passing it through, declare it in a config file —
see below.

## `.env` files

**The SDK does not read `.env` files.** It reads the *process environment* only.
Writing `DEEPSEEK_API_KEY=...` or `UAI_PROVIDER_DEEPSEEK_DEFAULT_MODEL=...` into
a `.env` has no effect unless something loads that file first:

```bash
pip install "uai-sdk[dotenv]"
```
```python
from dotenv import load_dotenv
load_dotenv()                 # must run BEFORE constructing the client

from uai import UniversalAI
client = UniversalAI(provider="deepseek")
```

Note also that there is no bare `DEEPSEEK_MODEL` variable — model overrides use
the namespaced `UAI_PROVIDER_{NAME}_DEFAULT_MODEL` form documented below.

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

The client discovers and applies this file automatically at construction — you
do not need to call the loader yourself. Pass `config_path=` to point at a
specific file:

```python
client = UniversalAI(provider="deepseek", config_path="./providers.yaml")
```

### Registering a new model

This is the way to teach the SDK about a model it does not ship metadata for,
so that context windows, pricing and capability checks are accurate rather than
inferred:

```yaml
providers:
  deepseek:
    default_model: "deepseek-v4-flash-0731"
    models:
      deepseek-v4-flash-0731:
        id: "deepseek-v4-flash-0731"
        display_name: "DeepSeek V4 Flash (0731)"
        context_window: 1000000
        max_output_tokens: 384000
        capabilities:
          chat: true
          streaming: true
          tools: true
        pricing:
          input_cost_per_1k: 0.00014
          output_cost_per_1k: 0.00028
```

Because the merge is recursive, the built-in `deepseek-v4-flash` and
`deepseek-v4-pro` entries survive alongside the addition.

The loader API remains available for programmatic use:

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
| `ARK_API_KEY` | Doubao (Volcengine Ark) |
| `MINIMAX_API_KEY` | MiniMax |
| `HUNYUAN_API_KEY` | Hunyuan |

### Credential scoping

Credentials are resolved **per provider**, never shared between them:

- An `api_key` (or `credentials`) passed to `UniversalAI(...)` applies **only
  to that client's provider**. It is not a global default.
- Any other provider — including one reached through a per-call `provider=`
  override — resolves its own key from its `api_key_env_var` above.
- If the target provider has no key available, the call raises `ValueError`
  naming the environment variable to set. It never falls back to another
  provider's credential.

```python
client = UniversalAI(api_key="sk-deepseek", provider="deepseek")

client.chat(messages=[...])                    # uses sk-deepseek
client.chat(messages=[...], provider="qwen")   # uses DASHSCOPE_API_KEY,
                                               # or raises if it is unset
```

The practical model is **one client per provider**, each holding its own
credential:

```python
deepseek = UniversalAI(provider="deepseek")   # DEEPSEEK_API_KEY
qwen     = UniversalAI(provider="qwen")       # DASHSCOPE_API_KEY
```

Because keys are read at call time rather than captured at construction, a
rotated environment variable takes effect without rebuilding the client.

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
| `UAI_PROVIDER_{NAME}_DEFAULT_MODEL` | `default_model` | string |
| `UAI_PROVIDER_{NAME}_DEFAULT_EMBEDDING_MODEL` | `default_embedding_model` | string |
| `UAI_PROVIDER_{NAME}_DEFAULT_RERANK_MODEL` | `default_rerank_model` | string |
| `UAI_PROVIDER_{NAME}_ALLOW_UNKNOWN_MODELS` | `allow_unknown_models` | bool |

Invalid values (e.g. a non-numeric timeout) are logged as warnings and skipped.
The resulting configuration is re-validated as a whole, so an override that
would produce an inconsistent config — pointing `DEFAULT_MODEL` at a model that
advertises only `embeddings`, say — is reported and ignored rather than failing
later inside a request.

```bash
# Pin a model without touching code:
export UAI_PROVIDER_DEEPSEEK_DEFAULT_MODEL=deepseek-v4-pro
```

A constructor `model=` still wins over `UAI_PROVIDER_{NAME}_DEFAULT_MODEL`,
consistent with constructor arguments outranking environment configuration
everywhere else in this page.

### Timeouts and retries

`timeout` and `max_retries` can also be passed to the constructor, where they
take precedence over both the registry defaults and the environment overrides
above:

```python
client = UniversalAI(provider="deepseek", timeout=120.0, max_retries=3)
```

- **`timeout`** is the deadline for a single HTTP request, in seconds. It
  applies to every provider the client calls, including one reached through a
  per-call `provider=` override, because it expresses the caller's own
  deadline rather than a property of the provider.
- **`max_retries`** enables automatic retries for transient failures — rate
  limits, network errors, timeouts, and 5xx responses. It is shorthand for
  registering a [`RetryMiddleware`](middleware.md), composed **inside** every
  middleware added through `use()`, so an open circuit breaker short-circuits
  without consuming attempts and a cache hit skips retrying entirely.

Retrying is **opt-in**. Leaving `max_retries` unset means a failed request
raises immediately; the `max_retries` value carried by each provider's
registry entry is not sufficient to switch retrying on by itself, and
`UAI_PROVIDER_{NAME}_MAX_RETRIES` overrides that value rather than enabling
retries.

For control over backoff, jitter, or which status codes are retryable,
register the middleware directly instead. An explicitly registered
`RetryMiddleware` supersedes the constructor shorthand — composing both would
nest two retry loops and multiply the request count — and a warning is logged
when that happens:

```python
client = UniversalAI(provider="deepseek")
client.use(RetryMiddleware(max_retries=5, base_delay=1.0, jitter=True))
```

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