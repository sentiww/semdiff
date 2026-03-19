from __future__ import annotations

import argparse


def _handle_synset_id(args: argparse.Namespace) -> None:
    from .id import synset_id

    for item in synset_id(args.query):
        print(item)


def _handle_synset_readable(args: argparse.Namespace) -> None:
    from .readable import synset_readable

    for item in synset_readable(args.synset_id):
        print(item)


def register_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    synset_parser = subparsers.add_parser(
        "synset",
        help="Synset commands",
    )
    synset_subparsers = synset_parser.add_subparsers(dest="synset_command")
    synset_subparsers.required = True

    synset_id_parser = synset_subparsers.add_parser(
        "id",
        help="Get synset id for a label",
    )
    synset_id_parser.add_argument(
        "query",
        help="ImageNet label or synonym, for example 'goldfish'",
    )
    synset_id_parser.set_defaults(handler=_handle_synset_id)

    synset_readable_parser = synset_subparsers.add_parser(
        "readable",
        help="Get human-readable labels for a synset id",
    )
    synset_readable_parser.add_argument(
        "synset_id",
        help="Synset id, for example 'n01443537'",
    )
    synset_readable_parser.set_defaults(handler=_handle_synset_readable)
