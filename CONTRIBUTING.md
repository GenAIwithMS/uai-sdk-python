# Contributing to the Universal AI Provider SDK

Thank you for taking the time to contribute. This guide covers everything you need to go from a fresh clone to a merged pull request.

By participating, you agree to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Standards](#code-standards)
- [Adding a New Provider](#adding-a-new-provider)
- [Adding Middleware](#adding-middleware)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Issue Triage and Labeling](#issue-triage-and-labeling)
- [Releasing](#releasing)
- [Getting Help](#getting-help)

---

## Ways to Contribute

Not every contribution is a code change, and all of these are genuinely useful:

| Contribution | Where to start |
|:---|:---|
| **Add a provider adapter** | The highest-leverage change, and it doesn't touch the core — follow the [PDK guide](docs/pdk.md) |
| **Report a bug** | [Open a bug report](https://github.com/GenAIwithMS/uai-sdk-python/issues/new/choose) with a minimal reproduction |
| **Improve docs** | Anything in [`docs/`](docs/) — unclear docs are bugs |
| **Write middleware** | See [Adding Middleware](#adding-middleware) |
| **Fix a `good-first-issue`** | [Browse them here](https://github.com/GenAIwithMS/uai-sdk-python/labels/good-first-issue) |
| **Update model metadata** | Providers change pricing, context windows, and capabilities often; the registry needs to keep up |

If you're planning a large change, open an issue first so we can agree on the approach before you invest the time.

---

## Getting Started

### Prerequisites

- **Python 3.9+** (3.12 recommended for development)
- **[Poetry](https://python-poetry.org/docs/#installation) 1.8+**
- **Git**

### Setup

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/uai-sdk-python.git
cd uai-sdk-python

# 2. Add the upstream remote so you can stay in sync
git remote add upstream https://github.com/GenAIwithMS/uai-sdk-python.git

# 3. Install the project with all dev dependencies
poetry install --with dev

# 4. Verify your environment
poetry run pytest tests/unit -q
poetry run ruff check src tests
poetry run mypy src
```

If all three pass, you're ready to work.

> **No API keys are needed for development.** The entire unit suite runs offline against mocked HTTP and the in-process `MockProviderServer`. Never point tests at a real provider.

---

## Project Structure

The project uses a **src layout** for packaging safety and import isolation.

```
uai-sdk-python/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── workflows/
│   │   ├── ci.yml               # lint → test → security, on push and PR
│   │   └── publish.yml          # PyPI release via OIDC Trusted Publishing
│   └── LABELS.md
├── docs/                        # user-facing documentation
├── src/
│   └── uai/
│       ├── __init__.py          # public API surface — update __all__ when exporting
│       ├── client.py            # UniversalAI orchestrator
│       ├── models.py            # unified request/response Pydantic models
│       ├── exceptions.py        # exception hierarchy rooted at UAIError
│       ├── enforcer.py          # CapabilityMatrixEnforcer
│       ├── structured.py        # structured-output prompting and validation
│       ├── benchmark.py         # benchmark engine behind `uai benchmark`
│       ├── cli.py               # `uai` command-line entry point
│       ├── adapters/
│       │   ├── base_adapter.py  # BaseProviderAdapter — the adapter contract
│       │   ├── deepseek.py      # one module per provider
│       │   └── ...
│       ├── middleware/
│       │   ├── base.py          # BaseMiddleware + MiddlewareContext
│       │   ├── engine.py        # MiddlewareEngine, MiddlewareHalt
│       │   ├── retry.py
│       │   ├── cache.py
│       │   ├── circuit_breaker.py
│       │   ├── logging.py
│       │   ├── metrics.py
│       │   └── tracing.py
│       ├── registry/
│       │   ├── providers.py     # canonical provider + model definitions
│       │   ├── schema.py        # Pydantic schemas validating the registry
│       │   ├── loader.py        # YAML/JSON config loading and merging
│       │   └── env.py           # UAI_PROVIDER_* environment overrides
│       └── testing/
│           └── mock_server.py   # stdlib-only MockProviderServer
├── tests/
│   ├── unit/                    # offline unit tests (the bulk of the suite)
│   │   └── middleware/
│   └── performance/             # KPI regression suite
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── pyproject.toml
```

---

## Development Workflow

1. **Sync with upstream** before starting:

   ```bash
   git checkout main
   git pull upstream main
   ```

2. **Create a branch** named for what it does:

   ```bash
   git checkout -b feat/add-baichuan-adapter
   ```

3. **Make your changes**, following the [Code Standards](#code-standards).

4. **Run the full local check** before committing:

   ```bash
   poetry run ruff check src tests --fix
   poetry run ruff format src tests
   poetry run mypy src
   poetry run pytest
   ```

5. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/):

   ```bash
   git commit -m "feat(adapters): add Baichuan provider adapter"
   ```

6. **Push and open a pull request** against `main`.

### Commit Message Types

| Type | Use for | Version impact |
|:---|:---|:---|
| `feat` | A new user-facing capability | Minor |
| `fix` | A bug fix | Patch |
| `perf` | A performance improvement | Patch |
| `refactor` | Restructuring with no behavior change | None |
| `test` | Adding or fixing tests | None |
| `docs` | Documentation only | None |
| `chore` | Build tooling, CI, dependency bumps | None |

Add a `!` (`feat!:`) or a `BREAKING CHANGE:` footer for anything that alters the public API.

---

## Testing

The project uses **pytest**. Every behavioral change needs a test.

### Running Tests

```bash
# Everything
poetry run pytest

# Unit tests only (fast — this is what you'll run most)
poetry run pytest tests/unit

# A single file or test
poetry run pytest tests/unit/test_client_features.py
poetry run pytest tests/unit/test_middleware.py::TestRetryMiddleware -v

# With coverage
poetry run pytest --cov=uai --cov-report=term-missing

# Performance KPI regression suite
poetry run pytest tests/performance -v

# Include the memory KPIs (subprocess-based, ~10s)
UAI_PERF_MEMORY=1 poetry run pytest tests/performance -v
```

### Test Guidelines

- **Never hit a real provider API.** Use [`respx`](https://lundberg.github.io/respx/) to mock `httpx` traffic, or `uai.testing.MockProviderServer` when you need the full client → middleware → adapter → HTTP → parse path.
- **Cover the error paths**, not just the happy path — a provider returning a 429, a malformed payload, or a truncated stream is exactly where this SDK earns its keep.
- **Test the capability matrix.** A new adapter must prove that unsupported features raise `FeatureNotSupportedError` *before* any network call.
- Use `pytest-mock`'s `mocker` fixture for internal patching; `asyncio_mode = "auto"` is already configured if you add async tests.
- Keep unit tests offline and deterministic — no sleeps longer than a few milliseconds, no network, no clock dependence.

### Using the Mock Server

```python
import os
from uai import UniversalAI
from uai.testing import MockProviderServer

def test_retry_recovers_from_rate_limit():
    with MockProviderServer() as server:
        os.environ["UAI_PROVIDER_DEEPSEEK_BASE_URL"] = server.base_url
        server.fail_with(429, count=2)

        client = UniversalAI(api_key="test", provider="deepseek")
        client.use(RetryMiddleware(max_retries=3, base_delay=0.01))

        assert client.chat(messages=[{"role": "user", "content": "hi"}]).content
        assert server.request_count == 3
```

---

## Code Standards

### Linting and Formatting

**Ruff** handles both:

```bash
poetry run ruff check src tests          # lint
poetry run ruff check src tests --fix    # lint and auto-fix
poetry run ruff format src tests         # format
poetry run ruff format --check src tests # verify formatting (what CI runs)
```

Configuration lives in `pyproject.toml`: 100-character lines, double quotes, `py39` target, and the `E`, `F`, `I`, `UP`, `B`, `SIM`, `LOG`, `RUF` rule sets.

### Type Checking

```bash
poetry run mypy src
```

Every public function, method, and class must be fully annotated. Use `from __future__ import annotations` at the top of new modules so 3.10+ syntax (`X | None`) works on Python 3.9.

### Python Version Compatibility

The SDK supports **Python 3.9 through 3.13**, and CI runs the suite on 3.9–3.12. Before using a newer-Python feature, check it degrades cleanly on 3.9 — `match` statements, `X | Y` at runtime, and `itertools.pairwise` are the usual traps.

### Naming Conventions

| Element | Convention | Example |
|:---|:---|:---|
| Classes | `PascalCase` | `UnifiedResponse` |
| Functions and methods | `snake_case` | `parse_response` |
| Variables | `snake_case` | `api_key` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_TIMEOUT` |
| Private members | `_snake_case` | `_translate_errors` |
| Exceptions | `PascalCase`, `UAI`-prefixed for SDK-wide errors | `UAIRateLimitError` |

### Security Rules

These are non-negotiable and enforced in review:

- **Never log, print, or embed an API key** in a message, exception, span, or metric label.
- **Never include raw credentials** in `response_body` or debug output.
- Validate all external input through Pydantic before acting on it.
- New dependencies need justification — the runtime dependency list is deliberately tiny (`pydantic`, `httpx`, `pyyaml`).

`bandit` and `pip-audit` run on every push; both must pass.

---

## Adding a New Provider

This is the most welcome kind of contribution. Read the [Provider Development Kit](docs/pdk.md) for the full contract, then:

1. **Implement the adapter** — subclass `BaseProviderAdapter` in `src/uai/adapters/<provider>.py`, implementing request formatting, response parsing, and `capabilities()`.
2. **Register the provider** — add a `ProviderConfig` to `src/uai/registry/providers.py` with accurate base URL, auth type, API key env var, and one `ProviderModel` per model (context window, max output tokens, pricing, aliases, capabilities).
3. **Wire lazy loading** — add the `(module, class_name)` entry to `_ADAPTER_SPECS` in `src/uai/client.py`. Do **not** import the adapter eagerly.
4. **Add it to `PROVIDER_ORDER`** so it appears in `uai list-providers`.
5. **Write tests** in `tests/unit/test_adapters_<provider>.py`, covering request formatting, response parsing, streaming chunks, error translation, and capability rejection.
6. **Document it** — add the provider to `docs/providers.md` and to the capability table in `README.md`.
7. **Add a CHANGELOG entry** under `[Unreleased]`.

Declare a capability as `True` only if you have verified it against the live API. An over-claimed capability is worse than a missing one — it turns a clean `FeatureNotSupportedError` into a confusing provider-side failure.

---

## Adding Middleware

Middleware follows the interceptor pattern described in [docs/middleware.md](docs/middleware.md).

1. Subclass `BaseMiddleware` in `src/uai/middleware/<name>.py` and set a unique `name`.
2. Implement only the hooks you need — `before_request`, `execute`, `after_response`, `on_error` all default to pass-through.
3. Export it from `src/uai/middleware/__init__.py` and, if it belongs on the top-level surface, from `src/uai/__init__.py` (`__all__` is sorted).
4. Add tests under `tests/unit/middleware/`.
5. Document its constructor arguments and ordering requirements in `docs/middleware.md`.

Keep middleware **synchronous** — it matches the current client — and keep the no-middleware path free of any cost it introduces.

---

## Documentation

Documentation ships with the code, not after it.

- Public modules, classes, and functions use **[Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)**.
- New features need a page or a section in [`docs/`](docs/), linked from `docs/index.md`.
- Update `README.md` when you change the public surface, the capability matrix, or the supported provider list.
- Code samples in docs must be runnable as written — no pseudo-code, no APIs that don't exist.

---

## Submitting Changes

Open your PR against `main` with a description covering **what changed, why, and how you verified it**. Link any related issue with `Closes #123`.

### PR Checklist

- [ ] Tests pass — `poetry run pytest`
- [ ] Linter passes — `poetry run ruff check src tests`
- [ ] Formatting is clean — `poetry run ruff format --check src tests`
- [ ] Type checker passes — `poetry run mypy src`
- [ ] New behavior has tests, including error paths
- [ ] Public API additions are exported and documented
- [ ] Docs under `docs/` are updated
- [ ] A `CHANGELOG.md` entry was added under `[Unreleased]`
- [ ] No credentials, keys, or secrets appear in code, tests, or logs
- [ ] Breaking changes are called out explicitly in the PR description

### Review

A maintainer will review your PR. Expect questions — they're about the code, not about you. Push follow-up commits to the same branch; we squash on merge, so there's no need to rewrite history.

---

## Issue Triage and Labeling

Every issue is triaged against the roadmap and labeled so its status is never ambiguous:

| Label | Meaning |
|:---|:---|
| `roadmap-aligned` | On the plan — we intend to build it |
| `needs-discussion` | Requires design discussion before a decision |
| `won't-implement` | Out of scope for this project's direction |
| `help-wanted` | We explicitly welcome a PR here |
| `good-first-issue` | Beginner-friendly, well-scoped |
| `bug` | Confirmed defect |
| `enhancement` | New feature request |
| `breaking` | Requires a major version bump |

The full label set and conventions live in [`.github/LABELS.md`](.github/LABELS.md).

---

## Releasing

Releases are maintainer-only and automated:

1. Update the version in `pyproject.toml` and `src/uai/__init__.py` (`__version__`) — these must match, and a test enforces the packaging metadata.
2. Promote the `[Unreleased]` section of `CHANGELOG.md` to a dated version heading and update the comparison links at the bottom.
3. Merge to `main` and confirm CI is green.
4. Tag the release and push it:

   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push upstream v0.2.0
   ```

5. Publish a GitHub Release for the tag. This triggers `.github/workflows/publish.yml`, which builds the package, runs `twine check`, and publishes to [PyPI](https://pypi.org/project/uai-sdk/) via OIDC Trusted Publishing — no API tokens are stored in the repository.

Version numbers follow [Semantic Versioning](https://semver.org/): breaking API changes bump major, new backward-compatible capabilities bump minor, fixes bump patch.

---

## Getting Help

- **Questions about usage** → [open a discussion or issue](https://github.com/GenAIwithMS/uai-sdk-python/issues)
- **Something's broken** → [file a bug report](https://github.com/GenAIwithMS/uai-sdk-python/issues/new/choose)
- **Not sure if an idea fits** → open an issue and label it `needs-discussion`

Thanks for helping make the SDK better.
