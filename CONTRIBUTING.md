# Contributing to the Universal AI Provider SDK

First off, thank you for taking the time to contribute! This document outlines the process for contributing to the **Universal AI Provider SDK (uai-sdk-python)**.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Standards](#code-standards)
- [Adding a New Provider](#adding-a-new-provider)
- [Adding Middleware](#adding-middleware)
- [Submitting Changes](#submitting-changes)
- [Issue Triage and Labeling](#issue-triage-and-labeling)
- [Releasing](#releasing)

---

## Code of Conduct

By participating in this project, you agree to uphold the standards described in the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Please be respectful and constructive in all interactions.

---

## Getting Started

### Prerequisites

- Python **3.9+** (3.12 recommended for best performance)
- [Poetry](https://python-poetry.org/docs/#installation) **1.8+**

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-org/uai-sdk-python.git
   cd uai-sdk-python
   ```

2. **Install Poetry (if not already installed):**

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies:**

   ```bash
   poetry install --no-interaction
   ```

4. **Install the SDK in editable mode with dev dependencies:**

   ```bash
   poetry install --no-interaction --with dev
   ```

5. **Activate the virtual environment:**

   ```bash
   poetry shell
   ```

---

## Project Structure

This project follows the **src layout** for maximum packaging safety and testability.

```
uai-sdk-python/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── feature_request.md
│   │   └── bug_report.md
│   └── workflows/
│       └── test.yml
├── docs/
│   ├── architecture.md
│   ├── chat.md
│   ├── middleware.md
│   ├── pdk.md
│   ├── streaming.md
│   └── ...
├── src/
│   └── uai/
│       ├── __init__.py
│       ├── client.py              # UniversalAI client orchestrator
│       ├── exceptions.py          # SDK exception hierarchy
│       ├── models.py              # Unified request/response data models
│       ├── middleware/
│       │   ├── base.py
│       │   ├── retry.py
│       │   ├── cache.py
│       │   └── logging.py
│       ├── registry/
│       │   ├── providers.py       # Hardcoded provider configs
│       │   ├── schema.py          # Pydantic schemas for registry
│       │   ├── loader.py          # YAML/JSON config loading
│       │   └── env.py             # Environment variable overrides
│       └── adapters/
│           ├── base.py            # Abstract adapter contract
│           ├── deepseek.py
│           ├── qwen.py
│           ├── glm.py
│           └── minimax.py
├── tests/
│   ├── unit/
│   │   ├── conftest.py
│   │   ├── test_registry.py
│   │   ├── test_models.py
│   │   ├── test_client.py
│   │   ├── test_exceptions.py
│   │   └── middleware/
│   └── integration/
│       ├── conftest.py
│       ├── test_chat.py
│       ├── test_streaming.py
│       ├── test_tools.py
│       └── test_providers.py
└── pyproject.toml
```

---

## Development Workflow

1. **Create a branch** for your feature or bugfix:

   ```bash
   git checkout -b feat/my-new-feature
   ```

2. **Make your changes** following the [Code Standards](#code-standards) below.

3. **Run tests** to ensure everything passes (see [Testing](#testing)).

4. **Commit your changes** with a clear message:

   ```bash
   git add .
   git commit -m "feat: add my awesome feature"
   ```

5. **Push to your branch:**

   ```bash
   git push origin feat/my-new-feature
   ```

6. **Open a Pull Request** using the relevant PR template.

We follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:

| Type       | Description                                        |
|------------|----------------------------------------------------|
| `feat`     | New feature                                        |
| `fix`      | Bug fix                                            |
| `refactor` | Code refactor (no functional change)               |
| `perf`     | Performance improvement                            |
| `test`     | Adding or fixing tests                             |
| `docs`     | Documentation changes                              |
| `chore`    | Build tooling, CI config, dependency updates       |

---

## Testing

This project uses **pytest** for testing. Tests are organized into `unit/` and `integration/` directories.

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run only unit tests
poetry run pytest tests/unit

# Run only integration tests
poetry run pytest tests/integration

# Run with coverage
poetry run pytest --cov=uai --cov-report=terminal-missing

# Run performance benchmarks
poetry run pytest tests/performance --benchmark-only

# Run with verbose output
poetry run pytest -v
```

### Test Writing Guidelines

- Use the **`pytest`** framework with **`pytest-asyncio`** for async tests.
- Use **`respx`** or **`responses`** to mock HTTP requests — never hit real provider APIs in unit tests.
- Use **`pytest-mock`** for mocking internal methods.
- Test edge cases, error paths, and invalid inputs.
- Integration tests should use a local mock server (see `conftest.py`).

### Fixtures

Common fixtures are defined in `tests/conftest.py`. These include:

- `mock_deepseek_client` — A fully mocked DeepSeek adapter.
- `mock_qwen_client` — A fully mocked Qwen adapter.
- `mock_universal_ai` — A mock `UniversalAI` client with default providers.
- `sample_chat_request` — A standard `UnifiedRequest` model instance.

---

## Code Standards

### Linting and Formatting

This project uses **Ruff** for both linting and formatting:

```bash
# Check for lint errors
poetry run ruff check src tests

# Auto-fix lint errors
poetry run ruff check src tests --fix

# Format code
poetry run ruff format src tests

# Check formatting
poetry run ruff format --check src tests

# Format check & fix in one step
poetry run ruff check src tests --fix && ruff format src tests
```

### Type Checking

This project uses **MyPy** for static type checking:

```bash
# Type check
poetry run mypy src

# Include tests
poetry run mypy src tests
```

All public-facing functions and classes must be fully typed.

### Naming Conventions

| Element          | Convention              | Example                    |
|------------------|-------------------------|----------------------------|
| Classes          | PascalCase              | `UnifiedResponse`          |
| Functions/Methods| snake_case              | `parse_response`           |
| Variables        | snake_case              | `api_key`                  |
| Constants        | UPPER_SNAKE_CASE        | `DEFAULT_TIMEOUT`          |
| Private methods  | `_snake_case`           | `_translate_errors`        |

### Documentation

- Use **docstrings** on all public modules, classes, and functions following [Google style](https://google.github.io/styleguide/pygdoc_styleguide/references.html).
- Update the [`docs/`](docs/) folder when adding new features.
- Keep `README.md` and this `CONTRIBUTING.md` up to date.

---

## Adding a New Provider

Contributors are encouraged to add support for new LLM providers. Follow the [Provider Development Kit (PDK)](docs/pdk.md) guide.

### Steps:

1. Implement the `BaseProviderAdapter` in `src/uai/adapters/`.
2. Add the provider to the registry in `src/uai/registry/providers.py`.
3. Update the capability matrix.
4. Write unit tests in `tests/unit/test_adapters.py`.
5. Write integration tests if a real API is available (use mocking for CI).
6. Document the provider in `docs/providers.md`.

---

## Adding Middleware

Middleware follows the interceptor pattern described in the [Middleware documentation](docs/middleware.md).

### Steps:

1. Subclass `BaseMiddleware` in `src/uai/middleware/`.
2. Implement `before_request()` and `after_response()` hooks as needed.
3. Register the middleware on the client: `client.use(MyMiddleware())`.
4. Write tests in `tests/unit/middleware/`.

---

## Submitting Changes

1. Ensure all tests pass and code is linted/formatted.
2. Update documentation as needed.
3. Open a PR against the `main` branch.
4. A maintainer will review your PR promptly.

### PR Checklist

- [ ] Tests pass (`poetry run pytest`)
- [ ] Linter passes (`poetry run ruff check`)
- [ ] Formatter is clean (`poetry run ruff format --check`)
- [ ] Type checker passes (`poetry run mypy`)
- [ ] New code is documented
- [ ] New features have tests
- [ ] Docs are updated (if applicable)

---

## Issue Triage and Labeling

This project uses the following label system to triage issues:

| Label              | When to Use                                              |
|--------------------|----------------------------------------------------------|
| `roadmap-aligned`  | Features we want to implement. On the planned roadmap.   |
| `needs-discussion` | Open for debate. Requires community/contributor input.   |
| `won't-implement`  | Clearly not aligned with the project's direction.        |
| `help-wanted`      | We want contributions here. Good for beginners.          |
| `bug`              | Confirmed bugs.                                          |
| `enhancement`      | New feature requests (see `roadmap-aligned` above).      |
| `good-first-issue` | Beginner-friendly tasks.                                 |
| `breaking`         | Requires changes to major version.                       |

When opening an issue, choose the appropriate template and apply labels as described above.

---

## Releasing

Releases are automated via GitHub Actions:

- Pushing a tag `v*` triggers a release to **PyPI**.
- The CI pipeline validates the build, runs all tests, and publishes the package.
- Changelogs are generated automatically from Conventional Commit messages.

Only maintainers can create releases.
