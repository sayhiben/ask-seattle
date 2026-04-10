from ask_seattle.cli import main
from ask_seattle.training import resolve_transformer_specs
from ask_seattle.transformer_model import (
    DEFAULT_TRANSFORMER_BENCHMARK_PRESETS,
    resolve_transformer_preset,
)


def test_resolve_transformer_preset() -> None:
    preset = resolve_transformer_preset("deberta-v3-small")

    assert preset.base_model == "microsoft/deberta-v3-small"
    assert preset.tier == "benchmark"


def test_resolve_transformer_specs_uses_benchmark_presets() -> None:
    specs = resolve_transformer_specs(
        transformer_presets=list(DEFAULT_TRANSFORMER_BENCHMARK_PRESETS),
    )

    assert [spec["name"] for spec in specs] == list(DEFAULT_TRANSFORMER_BENCHMARK_PRESETS)
    assert specs[0]["base_model"] == "distilbert/distilbert-base-uncased"


def test_resolve_transformer_specs_allows_custom_base_model() -> None:
    specs = resolve_transformer_specs(transformer_base_model="org/custom-model")

    assert specs == [
        {
            "name": "org_custom_model",
            "preset": None,
            "base_model": "org/custom-model",
            "tier": "custom",
            "max_length": 256,
        }
    ]


def test_transformer_presets_command_lists_presets(capsys) -> None:
    assert main(["transformer-presets"]) == 0

    output = capsys.readouterr().out
    assert "deberta-v3-small" in output
    assert "twitter-roberta-base" in output
