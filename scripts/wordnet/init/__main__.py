from scripts.common import logging

from . import main

if __name__ == "__main__":
    logging.configure()

    main.run()
