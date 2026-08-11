"""Unit tests for the offline benchmark module and its CLI commands."""

from __future__ import annotations

import pytest

from uai.benchmark import BenchmarkResult, SampleResult, benchmark_models
from uai.models import FinishReason, StreamChunk, UnifiedResponse, UsageMetrics
from uai.registry import list_providers


class FakeChatClient:
    """Minimal client stub that returns canned chat responses."""

    def chat(self, messages=None, model=None, max_tokens=None, temperature=None, stream=False):
        if stream:
            return iter(
                [
                    StreamChunk(content="Hello world", ttft_ms=4.0),
                    StreamChunk(
                        content=None,
                        finish_reason=FinishReason.STOP,
                        is_final=True,
                        usage=UsageMetrics(input_tokens=8, output_tokens=4),
                    ),
                ]
            )
        return UnifiedResponse(
            content="Hello world",
            model=model,
            finish_reason=FinishReason.STOP,
            usage=UsageMetrics(input_tokens=8, output_tokens=4),
        )


class TestBenchmarkResult:
    def test_aggregation(self):
        result = BenchmarkResult(
            provider="deepseek",
            model="deepseek-v4-flash",
            iterations=3,
            samples=[
                SampleResult(
                    success=True,
                    latency_ms=100.0,
                    ttft_ms=20.0,
                    input_tokens=8,
                    output_tokens=4,
                    cost_usd=0.001,
                ),
                SampleResult(
                    success=True,
                    latency_ms=200.0,
                    ttft_ms=40.0,
                    input_tokens=8,
                    output_tokens=4,
                    cost_usd=0.001,
                ),
                SampleResult(success=False, error="boom"),
            ],
        )
        assert result.success_count == 2
        assert result.error_rate == pytest.approx(1 / 3)
        assert result.ttft_ms_avg == pytest.approx(30.0)
        assert result.latency_ms_avg == pytest.approx(150.0)
        assert result.output_tokens_total == 8
        assert result.cost_usd_total == pytest.approx(0.002)

    def test_empty_values(self):
        result = BenchmarkResult(provider="p", model="m", iterations=1, samples=[])
        assert result.error_rate == 1.0
        assert result.ttft_ms_avg is None
        assert result.latency_ms_avg is None
        assert result.tokens_per_sec is None


class TestBenchmarkModels:
    def test_benchmarks_all_chat_models(self):
        results = benchmark_models(
            providers=["deepseek"],
            iterations=2,
            max_workers=2,
            client_factory=lambda p: FakeChatClient(),
        )
        models = {r.model for r in results}
        assert "deepseek-v4-flash" in models
        for r in results:
            assert r.success_count == 2
            assert r.error_rate == 0.0
            assert r.ttft_ms_avg is not None
            assert r.latency_ms_avg is not None
            assert r.cost_usd_total >= 0.0

    def test_skips_non_chat_models(self):
        results = benchmark_models(
            providers=["qwen"],
            models=["text-embedding-v4"],
            iterations=1,
            client_factory=lambda p: FakeChatClient(),
        )
        assert results == []

    def test_models_filter(self):
        results = benchmark_models(
            providers=["deepseek"],
            models=["deepseek-v4-flash"],
            iterations=1,
            client_factory=lambda p: FakeChatClient(),
        )
        assert [r.model for r in results] == ["deepseek-v4-flash"]

    def test_models_filter_skips_models_not_in_provider(self):
        results = benchmark_models(
            providers=["deepseek", "qwen"],
            models=["deepseek-v4-flash"],
            iterations=1,
            client_factory=lambda p: FakeChatClient(),
        )
        assert [r.provider for r in results] == ["deepseek"]

    def test_non_streaming_uses_reported_usage(self):
        results = benchmark_models(
            providers=["deepseek"],
            models=["deepseek-v4-flash"],
            iterations=1,
            stream=False,
            client_factory=lambda p: FakeChatClient(),
        )
        r = results[0]
        assert r.output_tokens_total == 4
        assert r.ttft_ms_avg is None


class TestCLI:
    def test_list_providers(self, capsys):
        from uai.cli import main

        code = main(["list-providers"])
        out = capsys.readouterr().out
        assert code == 0
        for name in list_providers():
            assert name in out

    def test_list_models(self, capsys):
        from uai.cli import main

        code = main(["list-models", "deepseek"])
        out = capsys.readouterr().out
        assert code == 0
        assert "deepseek-v4-flash" in out
        assert "deepseek-v4-pro" in out

    def test_benchmark_command(self, monkeypatch, capsys):
        import uai.cli as cli_module

        fake_results = [
            BenchmarkResult(
                provider="deepseek",
                model="deepseek-v4-flash",
                iterations=1,
                samples=[
                    SampleResult(
                        success=True,
                        latency_ms=50.0,
                        ttft_ms=10.0,
                        input_tokens=8,
                        output_tokens=4,
                        cost_usd=0.0001,
                    )
                ],
            )
        ]
        monkeypatch.setattr(cli_module, "benchmark_models", lambda **kw: fake_results)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        code = cli_module.main(["benchmark", "--providers", "deepseek", "--iterations", "1"])
        out = capsys.readouterr().out
        assert code == 0
        assert "deepseek-v4-flash" in out

    def test_benchmark_json_output(self, monkeypatch, capsys):
        import json

        import uai.cli as cli_module

        fake_results = [
            BenchmarkResult(
                provider="deepseek",
                model="deepseek-v4-flash",
                iterations=1,
                samples=[SampleResult(success=True, latency_ms=50.0, output_tokens=4)],
            )
        ]
        monkeypatch.setattr(cli_module, "benchmark_models", lambda **kw: fake_results)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        code = cli_module.main(
            ["benchmark", "--providers", "deepseek", "--iterations", "1", "--json"]
        )
        out = capsys.readouterr().out
        assert code == 0
        payload = json.loads(out)
        assert payload["results"][0]["provider"] == "deepseek"
        assert payload["results"][0]["model"] == "deepseek-v4-flash"

    def test_benchmark_skips_provider_without_key(self, monkeypatch, capsys):
        import uai.cli as cli_module

        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        code = cli_module.main(["benchmark", "--providers", "deepseek"])
        err = capsys.readouterr().err
        assert code == 2
        assert "skipping deepseek" in err
