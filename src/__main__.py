from cli.app import build_app


def main() -> None:
    app = build_app()
    app()


if __name__ == "__main__":
    main()
