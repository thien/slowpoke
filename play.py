"""Play checkers from project root."""

import multiprocessing

from slowpoke.play import main as _main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    _main()
