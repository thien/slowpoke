"""Run slowpoke training from project root."""

import multiprocessing

from slowpoke.train import train

if __name__ == "__main__":
    multiprocessing.freeze_support()
    train()
