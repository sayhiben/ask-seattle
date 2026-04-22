from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from ask_seattle.model import classify_post, load_model
from ask_seattle.training import (
    DEFAULT_BENCHMARK_SEED_MODELS,
    DEFAULT_BENCHMARK_SEED_SWEEP,
    DEFAULT_SPLIT_SEED,
    DEFAULT_SPLIT_STRATEGY,
    benchmark_model_suite_from_labels,
    benchmark_seed_sweep_from_labels,
    benchmark_model_variants_from_labels,
    retrain_all_from_labels,
    train_model_bundle_from_labels,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_cli_logging(args)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ask-seattle")
    subparsers = parser.add_subparsers(required=True)

    train = subparsers.add_parser("train", help="Train the TF-IDF classifier bundle")
    train.add_argument("--data", required=True, type=Path, help="Path to reviewed .jsonl label data")
    train.add_argument("--output-dir", required=True, type=Path, help="Where model artifacts go")
    train.add_argument(
        "--eval-subreddit",
        help="If set, train on mixed reviewed data but restrict calibration/test evaluation to this subreddit",
    )
    add_split_args(train)
    train.set_defaults(func=train_command)

    retrain_all = subparsers.add_parser(
        "retrain-all",
        help="Retrain the operational TF-IDF model and the five-model comparison suite without benchmarking",
    )
    retrain_all.add_argument("--data", required=True, type=Path, help="Path to reviewed .jsonl label data")
    retrain_all.add_argument(
        "--operational-output-dir",
        required=True,
        type=Path,
        help="Where the operational TF-IDF artifacts go",
    )
    retrain_all.add_argument(
        "--benchmark-output-dir",
        required=True,
        type=Path,
        help="Where the comparison-suite model artifacts go",
    )
    retrain_all.add_argument(
        "--eval-subreddit",
        help="If set, train on mixed reviewed data but restrict calibration/test evaluation to this subreddit",
    )
    add_split_args(retrain_all)
    retrain_all.add_argument(
        "--transformer-model-id",
        default="answerdotai/ModernBERT-base",
        help="Primary transformer checkpoint for the sequence classification comparison path",
    )
    retrain_all.add_argument(
        "--transformer-secondary-model-id",
        default="chandar-lab/NeoBERT",
        help="Secondary transformer checkpoint for the sequence classification comparison path",
    )
    retrain_all.add_argument(
        "--transformer-tertiary-model-id",
        default="answerdotai/ModernBERT-large",
        help="Tertiary transformer checkpoint for the sequence classification comparison path",
    )
    retrain_all.set_defaults(func=retrain_all_command)

    benchmark_variants = subparsers.add_parser(
        "benchmark-variants",
        help="Compare lightweight TF-IDF variants on the same held-out split",
    )
    benchmark_variants.add_argument("--data", required=True, type=Path, help="Path to reviewed .jsonl label data")
    benchmark_variants.add_argument("--output-dir", required=True, type=Path, help="Where benchmark artifacts go")
    benchmark_variants.add_argument(
        "--eval-subreddit",
        help="If set, train on mixed reviewed data but restrict calibration/test evaluation to this subreddit",
    )
    add_split_args(benchmark_variants)
    benchmark_variants.set_defaults(func=benchmark_variants_command)

    benchmark_suite = subparsers.add_parser(
        "benchmark-suite",
        help="Compare the five artifact-backed benchmark models on the same held-out split",
    )
    benchmark_suite.add_argument("--data", required=True, type=Path, help="Path to reviewed .jsonl label data")
    benchmark_suite.add_argument("--output-dir", required=True, type=Path, help="Where benchmark artifacts go")
    benchmark_suite.add_argument(
        "--eval-subreddit",
        help="If set, train on mixed reviewed data but restrict calibration/test evaluation to this subreddit",
    )
    add_split_args(benchmark_suite)
    benchmark_suite.add_argument(
        "--transformer-model-id",
        default="answerdotai/ModernBERT-base",
        help="Primary transformer checkpoint for the sequence classification benchmark path",
    )
    benchmark_suite.add_argument(
        "--transformer-secondary-model-id",
        default="chandar-lab/NeoBERT",
        help="Secondary transformer checkpoint for the sequence classification benchmark path",
    )
    benchmark_suite.add_argument(
        "--transformer-tertiary-model-id",
        default="answerdotai/ModernBERT-large",
        help="Tertiary transformer checkpoint for the sequence classification benchmark path",
    )
    benchmark_suite.add_argument(
        "--notes",
        help="Optional free-form note stored with the benchmark history entry for this run",
    )
    benchmark_suite.set_defaults(func=benchmark_suite_command)

    benchmark_seed_sweep = subparsers.add_parser(
        "benchmark-seed-sweep",
        help="Retrain and benchmark selected suite models across multiple deterministic split seeds",
    )
    benchmark_seed_sweep.add_argument("--data", required=True, type=Path, help="Path to reviewed .jsonl label data")
    benchmark_seed_sweep.add_argument("--output-dir", required=True, type=Path, help="Where benchmark artifacts go")
    benchmark_seed_sweep.add_argument(
        "--eval-subreddit",
        help="If set, train on mixed reviewed data but restrict calibration/test evaluation to this subreddit",
    )
    benchmark_seed_sweep.add_argument(
        "--benchmark-seeds",
        type=_parse_int_csv,
        default=DEFAULT_BENCHMARK_SEED_SWEEP,
        help="Comma-separated split seeds for repeated selected-model evaluation",
    )
    benchmark_seed_sweep.add_argument(
        "--benchmark-seed-models",
        type=_parse_csv,
        default=DEFAULT_BENCHMARK_SEED_MODELS,
        help="Comma-separated suite model names to include in the seed sweep",
    )
    benchmark_seed_sweep.add_argument(
        "--split-strategy",
        choices=["random", "time"],
        default=DEFAULT_SPLIT_STRATEGY,
        help="How to split reviewed labels into train, calibration, and test sets",
    )
    benchmark_seed_sweep.add_argument(
        "--transformer-model-id",
        default="answerdotai/ModernBERT-base",
        help="Primary transformer checkpoint for the sequence classification benchmark path",
    )
    benchmark_seed_sweep.add_argument(
        "--transformer-secondary-model-id",
        default="chandar-lab/NeoBERT",
        help="Secondary transformer checkpoint for the sequence classification benchmark path",
    )
    benchmark_seed_sweep.add_argument(
        "--transformer-tertiary-model-id",
        default="answerdotai/ModernBERT-large",
        help="Tertiary transformer checkpoint for the sequence classification benchmark path",
    )
    benchmark_seed_sweep.set_defaults(func=benchmark_seed_sweep_command)

    check = subparsers.add_parser("check", help="Classify a single post")
    add_inference_args(check)
    check.set_defaults(func=check_command)

    bridge = subparsers.add_parser(
        "serve-bridge",
        help="Run a localhost bridge for the Tampermonkey userscript",
    )
    bridge.add_argument("--host", default="127.0.0.1")
    bridge.add_argument("--port", type=int, default=8765)
    bridge.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to the primary bridge bundle used for fallback and audit decisions",
    )
    bridge.add_argument(
        "--labels",
        type=Path,
        default=Path("data/processed/tampermonkey_labels.jsonl"),
        help="Where Train post appends labeled examples",
    )
    bridge.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Bridge logging verbosity",
    )
    bridge.add_argument(
        "--retrain-every",
        type=int,
        default=0,
        help="Auto-retrain the primary bridge bundle after every N new effective training rows",
    )
    bridge.add_argument(
        "--comparison-suite",
        type=Path,
        default=Path("models/benchmark-suite/benchmark_suite_summary.json"),
        help=(
            "Optional benchmark-suite summary used to load the other benchmark models for side-by-side "
            "UI comparisons when checking posts"
        ),
    )
    bridge.add_argument(
        "--decider-policy",
        choices=["primary_only", "hybrid_consensus", "stacked_transformer_decider"],
        default="stacked_transformer_decider",
        help=(
            "How `/check` should decide the main verdict. `primary_only` keeps the active bridge model, "
            "`hybrid_consensus` routes hard cases through benchmark-weighted comparisons, and "
            "`stacked_transformer_decider` uses the trained stacked transformer policy when available"
        ),
    )
    bridge.add_argument(
        "--eval-subreddit",
        help="If set, auto-retrain will calibrate and test only on this subreddit while training on mixed data",
    )
    add_split_args(bridge)
    bridge.set_defaults(func=serve_bridge_command)

    return parser


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, type=Path, help="Path to a trained model bundle")
    parser.add_argument("--title", required=True, help="Submission title")
    parser.add_argument("--selftext", default="", help="Submission body")


def add_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split-strategy",
        choices=["random", "time"],
        default=DEFAULT_SPLIT_STRATEGY,
        help="How to split reviewed labels into train, calibration, and test sets",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Deterministic seed used for random split sampling",
    )


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _parse_csv(value))


def _configure_cli_logging(args: argparse.Namespace) -> None:
    command = getattr(getattr(args, "func", None), "__name__", "")
    if command == "serve_bridge_command":
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )


def train_command(args: argparse.Namespace) -> int:
    summary = train_model_bundle_from_labels(
        args.data,
        args.output_dir,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
        evaluation_subreddit=args.eval_subreddit,
    )
    print(json.dumps(summary, indent=2))
    return 0


def retrain_all_command(args: argparse.Namespace) -> int:
    summary = retrain_all_from_labels(
        args.data,
        operational_output_dir=args.operational_output_dir,
        benchmark_output_dir=args.benchmark_output_dir,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
        evaluation_subreddit=args.eval_subreddit,
        transformer_model_id=args.transformer_model_id,
        transformer_secondary_model_id=args.transformer_secondary_model_id,
        transformer_tertiary_model_id=args.transformer_tertiary_model_id,
    )
    print(json.dumps(summary, indent=2))
    return 0


def benchmark_variants_command(args: argparse.Namespace) -> int:
    summary = benchmark_model_variants_from_labels(
        args.data,
        args.output_dir,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
        evaluation_subreddit=args.eval_subreddit,
    )
    print(json.dumps(summary, indent=2))
    return 0


def benchmark_suite_command(args: argparse.Namespace) -> int:
    summary = benchmark_model_suite_from_labels(
        args.data,
        args.output_dir,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
        evaluation_subreddit=args.eval_subreddit,
        transformer_model_id=args.transformer_model_id,
        transformer_secondary_model_id=args.transformer_secondary_model_id,
        transformer_tertiary_model_id=args.transformer_tertiary_model_id,
        notes=args.notes,
    )
    print(json.dumps(summary, indent=2))
    return 0


def benchmark_seed_sweep_command(args: argparse.Namespace) -> int:
    summary = benchmark_seed_sweep_from_labels(
        args.data,
        args.output_dir,
        split_strategy=args.split_strategy,
        split_seeds=tuple(args.benchmark_seeds),
        model_names=tuple(args.benchmark_seed_models),
        evaluation_subreddit=args.eval_subreddit,
        transformer_model_id=args.transformer_model_id,
        transformer_secondary_model_id=args.transformer_secondary_model_id,
        transformer_tertiary_model_id=args.transformer_tertiary_model_id,
    )
    print(json.dumps(summary, indent=2))
    return 0


def check_command(args: argparse.Namespace) -> int:
    bundle = load_model(args.model)
    result = classify_post(bundle, title=args.title, selftext=args.selftext)
    print(json.dumps(asdict(result), indent=2))
    return 0


def serve_bridge_command(args: argparse.Namespace) -> int:
    from ask_seattle.local_bridge import run_bridge

    run_bridge(
        host=args.host,
        port=args.port,
        model_path=args.model,
        label_path=args.labels,
        comparison_suite_path=args.comparison_suite,
        log_level=args.log_level,
        retrain_every=args.retrain_every,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
        evaluation_subreddit=args.eval_subreddit,
        decider_policy=args.decider_policy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
