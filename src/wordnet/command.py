from __future__ import annotations

import argparse


def _handle_init(_: argparse.Namespace) -> None:
    from .init import init_wordnet

    init_wordnet()


def register_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    wordnet_parser = subparsers.add_parser("wordnet", help="WordNet commands")
    wordnet_subparsers = wordnet_parser.add_subparsers(dest="wordnet_command")
    wordnet_subparsers.required = True

    init_parser = wordnet_subparsers.add_parser(
        "init",
        help="Download and initialize the WordNet corpus",
    )
    init_parser.set_defaults(handler=_handle_init)
