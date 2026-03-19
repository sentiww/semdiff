from __future__ import annotations

import argparse


def _handle_semantic(args: argparse.Namespace) -> None:
    from .semantic import build_semantic_metrics

    build_semantic_metrics(args.model, args.dataset)


def register_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    analysis_parser = subparsers.add_parser(
        "analysis",
        help="Analysis commands built from evaluation outputs",
    )
    analysis_subparsers = analysis_parser.add_subparsers(dest="analysis_command")
    analysis_subparsers.required = True

    semantic_parser = analysis_subparsers.add_parser(
        "semantic",
        help="Compute WordNet semantic metrics from output predictions",
    )
    semantic_parser.add_argument(
        "model",
        help="Model output directory name under output/",
    )
    semantic_parser.add_argument(
        "dataset",
        help="Dataset output directory name under output/<model>/",
    )
    semantic_parser.set_defaults(handler=_handle_semantic)
