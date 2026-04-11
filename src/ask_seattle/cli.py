from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ask_seattle.model import classify_post, load_model
from ask_seattle.training import benchmark_model_variants_from_labels, train_model_bundle_from_labels


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
    train.set_defaults(func=train_command)

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
    benchmark_variants.set_defaults(func=benchmark_variants_command)

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
        help="Path to a trained TF-IDF bundle",
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
        help="Auto-retrain the TF-IDF model after every N new effective training rows",
    )
    bridge.add_argument(
        "--eval-subreddit",
        help="If set, auto-retrain will calibrate and test only on this subreddit while training on mixed data",
    )
    bridge.set_defaults(func=serve_bridge_command)

    return parser


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, type=Path, help="Path to a trained model bundle")
    parser.add_argument("--title", required=True, help="Submission title")
    parser.add_argument("--selftext", default="", help="Submission body")


def train_command(args: argparse.Namespace) -> int:
    summary = train_model_bundle_from_labels(
        args.data,
        args.output_dir,
        evaluation_subreddit=args.eval_subreddit,
    )
    print(json.dumps(summary, indent=2))
    return 0


def benchmark_variants_command(args: argparse.Namespace) -> int:
    summary = benchmark_model_variants_from_labels(
        args.data,
        args.output_dir,
        evaluation_subreddit=args.eval_subreddit,
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
        log_level=args.log_level,
        retrain_every=args.retrain_every,
        evaluation_subreddit=args.eval_subreddit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
