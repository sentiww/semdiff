from cli.app import build_app
from bootstrap.logging import configure_logging


def main() -> None:
    configure_logging()
    app = build_app()
    app()


if __name__ == "__main__":
    main()
