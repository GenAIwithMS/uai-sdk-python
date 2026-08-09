"""
Command-line interface for the Universal AI Provider SDK.

Commands:

- ``uai benchmark`` — benchmark chat models across providers (all by default)
- ``uai list-providers`` — list registered providers
- ``uai list-models [PROVIDER]`` — list models (and capabilities) per provider
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from uai import __version__
from uai.benchmark import (
    DEFAULT_ITERATIONS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_WORKERS,
    DEFAULT_PROMPT,
    BenchmarkResult,
    benchmark_models,
)
from uai.registry import get_provider_config, list_providers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uai",
        description="Universal AI Provider SDK command-line interface.",
    )
    parser.add_argument("--version", action="version", version=f"uai {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("benchmark", help="Benchmark chat models across providers")
    bench.add_argument(
        "--providers",
        default="",
        help="Comma-separated provider names (default: all providers with an API key)",
    )
    bench.add_argument(
        "--models",
        default="",
        help="Comma-separated model ids (default: all chat models of each provider)",
    )
    bench.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text to send")
    bench.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Calls per model")
    bench.add_argument("--parallel", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel workers")
    bench.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max output tokens"
    )
    bench.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    bench.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (no TTFT; uses provider-reported usage)",
    )
    bench.add_argument("--json", action="store_true", help="Output results as JSON")

    sub.add_parser("list-providers", help="List registered providers")

    lm = sub.add_parser("list-models", help="List models per provider")
    lm.add_argument("provider", nargs="?", default=None, help="Provider name (default: all)")

    return parser


def _iter_models(provider: str, models: list[str] | None = None):
    """Yield (model_id, model_info) for *provider*, filtered by *models*."""
    config = get_provider_config(provider)
    for model_id in models or list(config.models.keys()):
        yield model_id, config.get_model(model_id)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_list_providers(args: argparse.Namespace) -> int:
    for name in list_providers():
        config = get_provider_config(name)
        key_env = config.api_key_env_var
        configured = "yes" if os.environ.get(key_env) else "no"
        print(f"{name:<12} {config.display_name:<24} api_key_env={key_env} configured={configured}")
    return 0


def _cmd_list_models(args: argparse.Namespace) -> int:
    providers = [args.provider] if args.provider else list_providers()
    found_any = False
    for provider in providers:
        try:
            config = get_provider_config(provider)
        except ValueError as e:
            print(f"! {e}", file=sys.stderr)
            continue
        found_any = True
        print(f"\n{provider}:")
        for model_id, info in _iter_models(provider):
            caps = [c for c, v in info.capabilities.model_dump().items() if v]
            default = " (default)" if model_id == config.default_model else ""
            print(f"  {model_id:<24} ctx={info.context_window:<7} caps={','.join(caps)}{default}")
    return 0 if found_any else 2


def _result_to_dict(result: BenchmarkResult) -> dict[str, Any]:
    return {
        "provider": result.provider,
        "model": result.model,
        "iterations": result.iterations,
        "success_count": result.success_count,
        "failure_count": result.failure_count,
        "error_rate": round(result.error_rate, 4),
        "ttft_ms_avg": result.ttft_ms_avg,
        "ttft_ms_p95": result.ttft_ms_p95,
        "latency_ms_avg": result.latency_ms_avg,
        "latency_ms_p95": result.latency_ms_p95,
        "tokens_per_sec": result.tokens_per_sec,
        "cost_usd_total": round(result.cost_usd_total, 6),
        "input_tokens_total": result.input_tokens_total,
        "output_tokens_total": result.output_tokens_total,
    }


def _cmd_benchmark(args: argparse.Namespace) -> int:
    providers = (
        [p.strip().lower() for p in args.providers.split(",") if p.strip()]
        if args.providers
        else None
    )
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None

    provider_names = providers or list_providers()
    active: list[str] = []
    for name in provider_names:
        try:
            config = get_provider_config(name)
        except ValueError as e:
            print(f"! {e}", file=sys.stderr)
            continue
        env_var = config.api_key_env_var
        if env_var and not os.environ.get(env_var):
            print(f"! skipping {name}: {env_var} is not set in the environment", file=sys.stderr)
        else:
            active.append(name)

    if not active:
        print(
            "No providers with API keys configured. Set the API key environment "
            "variable(s) and try again.",
            file=sys.stderr,
        )
        return 2

    results = benchmark_models(
        providers=active,
        models=models,
        prompt=args.prompt,
        iterations=args.iterations,
        max_workers=args.parallel,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=not args.no_stream,
    )

    if args.json:
        payload = {"prompt": args.prompt, "results": [_result_to_dict(r) for r in results]}
        print(json.dumps(payload, indent=2))
        return 0

    _print_table(results)
    return 0


def _print_table(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'Provider':<10} {'Model':<22} {'Iter':>4} {'OK':>4} {'TTFT ms':>8} "
        f"{'Lat ms':>8} {'Tok/s':>7} {'Cost $':>9} {'Err %':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:

        def fmt(value: float | None, digits: int = 0) -> str:
            if value is None:
                return "-"
            return f"{value:.{digits}f}"

        print(
            f"{r.provider:<10} {r.model:<22} {r.iterations:>4} {r.success_count:>4} "
            f"{fmt(r.ttft_ms_avg):>8} {fmt(r.latency_ms_avg):>8} "
            f"{fmt(r.tokens_per_sec):>7} {fmt(r.cost_usd_total, 6):>9} "
            f"{r.error_rate * 100:>5.1f}%"
        )
    print(
        "\nTTFT = time-to-first-token (streaming only). Cost = estimated from provider"
        " pricing and token usage."
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    args = _build_parser().parse_args(argv)

    if args.command == "benchmark":
        return _cmd_benchmark(args)
    if args.command == "list-providers":
        return _cmd_list_providers(args)
    if args.command == "list-models":
        return _cmd_list_models(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
