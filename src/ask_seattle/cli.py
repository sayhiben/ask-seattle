from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from ask_seattle.data import export_labeling_csv, import_labeling_csv, load_labeled_posts
from ask_seattle.decision_log import export_review_csv
from ask_seattle.model import load_model, save_model, score_post, train_and_evaluate
from ask_seattle.moderation import decide
from ask_seattle.reddit_data import collect_submissions, reddit_from_env, refresh_deleted_content
from ask_seattle.transformer_model import DEFAULT_BASE_MODEL
from ask_seattle.training import train_all_models


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ask-seattle")
    subparsers = parser.add_subparsers(required=True)

    train = subparsers.add_parser("train", help="Train and evaluate a classifier")
    train.add_argument("--data", required=True, type=Path, help="Path to labeled .jsonl or .csv data")
    train.add_argument("--model", required=True, type=Path, help="Where to write the model bundle")
    train.add_argument("--threshold", type=float, default=0.85, help="Removal threshold")
    train.add_argument("--test-size", type=float, default=0.25, help="Held-out evaluation fraction")
    train.add_argument("--random-state", type=int, default=42)
    train.set_defaults(func=train_command)

    train_all = subparsers.add_parser("train-all", help="Train and compare TF-IDF and transformer models")
    train_all.add_argument("--data", required=True, type=Path, help="Path to labeled .jsonl or .csv data")
    train_all.add_argument("--output-dir", required=True, type=Path, help="Where model artifacts go")
    train_all.add_argument("--min-precision", type=float, default=0.95)
    train_all.add_argument("--validation-size", type=float, default=0.2)
    train_all.add_argument("--test-size", type=float, default=0.2)
    train_all.add_argument("--random-state", type=int, default=42)
    train_all.add_argument("--transformer-base-model", default=DEFAULT_BASE_MODEL)
    train_all.add_argument("--transformer-epochs", type=int, default=2)
    train_all.add_argument("--transformer-batch-size", type=int, default=8)
    train_all.add_argument("--skip-transformer", action="store_true")
    train_all.set_defaults(func=train_all_command)

    predict = subparsers.add_parser("predict", help="Score a single post")
    add_inference_args(predict)
    predict.set_defaults(func=predict_command)

    decision = subparsers.add_parser("decide", help="Return the moderation decision for one post")
    add_inference_args(decision)
    decision.add_argument("--threshold", type=float, default=None, help="Override the model threshold")
    decision.set_defaults(func=decide_command)

    stream = subparsers.add_parser("stream", help="Watch a subreddit and apply decisions")
    stream.set_defaults(func=stream_command)

    collect = subparsers.add_parser("collect", help="Collect subreddit submissions into raw JSONL")
    collect.add_argument("--output", type=Path, default=Path("data/raw/submissions.jsonl"))
    collect.add_argument("--limit", type=int, default=500, help="Recent submissions to fetch first")
    collect.add_argument("--stream", action="store_true", help="Continue collecting new submissions")
    collect.add_argument("--subreddit", default=None, help="Defaults to REDDIT_SUBREDDIT")
    collect.set_defaults(func=collect_command)

    export_labeling = subparsers.add_parser("export-labeling", help="Export raw posts to labeling CSV")
    export_labeling.add_argument("--raw", required=True, type=Path)
    export_labeling.add_argument("--output", required=True, type=Path)
    export_labeling.set_defaults(func=export_labeling_command)

    import_labels = subparsers.add_parser("import-labels", help="Import reviewed labels to JSONL")
    import_labels.add_argument("--labels", required=True, type=Path)
    import_labels.add_argument("--output", required=True, type=Path)
    import_labels.set_defaults(func=import_labels_command)

    refresh = subparsers.add_parser("refresh-deletions", help="Purge deleted/removed content locally")
    refresh.add_argument("--raw", required=True, type=Path)
    refresh.add_argument("--output", type=Path, default=None)
    refresh.add_argument("--max-items", type=int, default=None)
    refresh.set_defaults(func=refresh_deletions_command)

    export_review = subparsers.add_parser("export-review", help="Export decision JSONL to review CSV")
    export_review.add_argument("--decisions", required=True, type=Path)
    export_review.add_argument("--output", required=True, type=Path)
    export_review.set_defaults(func=export_review_command)

    return parser


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, type=Path, help="Path to a trained model bundle")
    parser.add_argument("--title", required=True, help="Submission title")
    parser.add_argument("--selftext", default="", help="Submission body")


def train_command(args: argparse.Namespace) -> int:
    if not 0 < args.test_size < 1:
        raise SystemExit("--test-size must be between 0 and 1")

    posts = load_labeled_posts(args.data)
    model, evaluation = train_and_evaluate(
        posts,
        threshold=args.threshold,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    save_model(model, args.model, threshold=args.threshold)

    print(f"trained_examples={len(posts)}")
    print(f"model={args.model}")
    print(f"threshold={args.threshold:.3f}")
    if evaluation is None:
        print("evaluation=skipped; not enough data for a stratified held-out split")
        return 0

    print("confusion_matrix_labels=[not_askseattle, askseattle]")
    print(json.dumps(evaluation.confusion_matrix))
    print(
        "askseattle_at_threshold="
        f"precision={evaluation.precision:.3f} "
        f"recall={evaluation.recall:.3f} "
        f"f1={evaluation.f1:.3f} "
        f"support={evaluation.support}"
    )
    print("threshold_sweep=" + json.dumps(evaluation.threshold_sweep))
    print(evaluation.classification_report)
    return 0


def train_all_command(args: argparse.Namespace) -> int:
    posts = load_labeled_posts(args.data)
    summary = train_all_models(
        posts,
        args.output_dir,
        min_precision=args.min_precision,
        validation_size=args.validation_size,
        test_size=args.test_size,
        random_state=args.random_state,
        include_transformer=not args.skip_transformer,
        transformer_base_model=args.transformer_base_model,
        transformer_epochs=args.transformer_epochs,
        transformer_batch_size=args.transformer_batch_size,
        production_ready_blocked_reason=(
            "seed_dataset_smoke_test" if _is_seed_dataset(args.data) else None
        ),
    )
    print(json.dumps(summary, indent=2))
    return 0


def predict_command(args: argparse.Namespace) -> int:
    bundle = load_model(args.model)
    score = score_post(bundle, title=args.title, selftext=args.selftext)
    print(
        json.dumps(
            {
                "score": score,
                "threshold": bundle.get("threshold"),
                "model_name": bundle.get("model_name"),
                "model_version": bundle.get("model_version"),
            },
            indent=2,
        )
    )
    return 0


def decide_command(args: argparse.Namespace) -> int:
    bundle = load_model(args.model)
    decision = decide(bundle, title=args.title, selftext=args.selftext, threshold=args.threshold)
    print(json.dumps(asdict(decision), indent=2))
    return 0


def stream_command(_: argparse.Namespace) -> int:
    from ask_seattle.reddit_stream import main as stream_main

    return stream_main()


def collect_command(args: argparse.Namespace) -> int:
    subreddit_name = args.subreddit or os.getenv("REDDIT_SUBREDDIT")
    if not subreddit_name:
        raise SystemExit("Missing subreddit; pass --subreddit or set REDDIT_SUBREDDIT")

    result = collect_submissions(
        reddit_from_env(),
        subreddit_name=subreddit_name,
        output_path=args.output,
        limit=args.limit,
        stream=args.stream,
    )
    print(json.dumps(result, indent=2))
    return 0


def export_labeling_command(args: argparse.Namespace) -> int:
    result = export_labeling_csv(args.raw, args.output)
    print(json.dumps(result, indent=2))
    return 0


def import_labels_command(args: argparse.Namespace) -> int:
    result = import_labeling_csv(args.labels, args.output)
    print(json.dumps(result, indent=2))
    return 0


def refresh_deletions_command(args: argparse.Namespace) -> int:
    result = refresh_deleted_content(
        reddit_from_env(),
        raw_path=args.raw,
        output_path=args.output,
        max_items=args.max_items,
    )
    print(json.dumps(result, indent=2))
    return 0


def export_review_command(args: argparse.Namespace) -> int:
    result = export_review_csv(args.decisions, args.output)
    print(json.dumps(result, indent=2))
    return 0


def _is_seed_dataset(path: Path) -> bool:
    normalized_parts = {part.lower() for part in path.parts}
    return "seed" in normalized_parts


if __name__ == "__main__":
    raise SystemExit(main())
