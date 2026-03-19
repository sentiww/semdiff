from __future__ import annotations

import argparse


def _handle_path_distance(args: argparse.Namespace) -> None:
    from .path_distance import build_path_distance

    build_path_distance(args.model, args.dataset)


def register_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    analysis_parser = subparsers.add_parser(
        "analysis",
        help="Analysis commands built from evaluation outputs",
    )
    analysis_subparsers = analysis_parser.add_subparsers(dest="analysis_command")
    analysis_subparsers.required = True

    path_distance_parser = analysis_subparsers.add_parser(
        "path-distance",
        help="Compute WordNet path distances from output predictions",
    )
    path_distance_parser.add_argument(
        "model",
        help="Model output directory name under output/",
    )
    path_distance_parser.add_argument(
        "dataset",
        help="Dataset output directory name under output/<model>/",
    )
    path_distance_parser.set_defaults(handler=_handle_path_distance)
