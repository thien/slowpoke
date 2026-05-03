# Slowpoke

![slowpoke.png](slowpoke.png)

Slowpoke is a checkboard playing program for my 3rd year dissertation. It's inspired by Blondie24 but also includes a series of modifications that allow it to be a system that can be taken seriously in 2017. It revolves around GANNs (Genetic Algorithms / Neural Networks) and moves are evaluated using a modified Monte-Carlo Tree Search. 

## Setup with uv

This project uses `uv` for dependency management. Install `uv` first, then:

```bash
# Install dependencies
uv sync

# Or install with dev dependencies
uv sync --group dev
```

## Training

Training is called by running the python file from the library directory:

```bash
cd library && uv run python train.py light    # quick simulation
cd library && uv run python train.py medium   # medium load
cd library && uv run python train.py heavy    # full load
cd library && uv run python train.py debug    # debug mode
```

The program will attempt to utilise as many cores that the computer running the program has. I'm currently running this on a 128-core machine, which takes around 20 hours to finish (200 generations, 15 players per generation, 6ply).

## Playing a Champion

```bash
cd library && uv run python play.py
```

The program above also allows arguments; so you can quickly test the system:

```bash
cd library && uv run python play.py b=slowpoke w=slowpoke ply=8
```

## Evaluations

```bash
cd library && uv run python evaluate.py
```

There is currently a variety of configured games that slowpoke will try to win. They're currently set to play for 256 games (128 on both sides - black and white). Statistics are scored in `root/results/evaluations` in the form of a `.json` and a `.csv`.

## Dependencies

- numpy - numerical operations
- matplotlib - charting and statistics
- pymongo - MongoDB integration (optional, for storing results)
- termcolor - colored terminal output

## Development

```bash
# Format with ruff (black-compatible), then lint
ruff format .
ruff check --fix --unsafe-fixes .
# Run this before every commit

# Run tests
make test        # fast tests only
make test-all    # all tests including slow
```