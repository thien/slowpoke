"""
Hybrid TMCTS - Combines multiprocessing for games with parallel MCTS for tree search.

Strategy:
- Multiprocessing: Distributes games across CPU cores
- Parallel MCTS: Each bot runs multiple MCTS instances in parallel for better GPU utilization

Optimal configuration depends on your hardware:
- M2 Ultra (24 cores): 6 game processes × 4 parallel threads each = optimal
- 16-core CPU: 4 game processes × 4 parallel threads each = optimal
"""

import multiprocessing
from typing import Optional


def get_optimal_parallel_config(num_cores: Optional[int] = None) -> dict:
    """
    Calculate optimal configuration for hybrid parallelism.

    Args:
        num_cores: Number of CPU cores. If None, uses multiprocessing.cpu_count()

    Returns:
        dict with 'num_game_processes', 'num_parallel_threads', 'total_threads'
    """
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()

    # Each parallel MCTS uses threads (not processes), so we can oversubscribe slightly
    # Aim for: num_game_processes * num_parallel_threads ≈ 1.5 × num_cores

    # For M-series: More game processes (they're efficient)
    # For Intel/AMD: Fewer game processes, more parallel threads

    if num_cores >= 20:  # M2 Ultra, high-end CPUs
        num_game_processes = max(4, num_cores // 4)
        num_parallel_threads = 4
    elif num_cores >= 10:  # Mid-range CPUs
        num_game_processes = max(2, num_cores // 3)
        num_parallel_threads = 3
    else:  # Low-core systems
        num_game_processes = 2
        num_parallel_threads = 2

    # Don't exceed available cores for game processes
    num_game_processes = min(num_game_processes, num_cores)

    return {
        "num_game_processes": num_game_processes,
        "num_parallel_threads": num_parallel_threads,
        "total_threads": num_game_processes * num_parallel_threads,
    }


def create_bot_with_parallel_tmcts(
    ply: int,
    nn,
    num_parallel: int = 4,
    use_mlx: bool = True,
    debug: bool = False,
    seed: Optional[int] = None,
):
    """
    Create a Slowpoke bot configured with parallel TMCTS.

    Args:
        ply: Search depth
        nn: Neural network instance
        num_parallel: Number of parallel MCTS instances
        use_mlx: Whether to use MLX for batch evaluation
        debug: Debug mode
        seed: Random seed for reproducibility

    Returns:
        Configured Slowpoke bot instance
    """
    from agents.slowpoke import Slowpoke
    from search.parallel_tmcts import ParallelTMCTS

    # Create bot with parallel TMCTS
    bot = Slowpoke(
        ply_depth=ply,
        layers=nn.layers if hasattr(nn, "layers") else [91, 40, 10, 1],
        weights=[],
        isminimax=False,
        debug=debug,
        use_mlx=use_mlx,
    )

    # Replace the decision function with parallel TMCTS
    parallel_mcts = ParallelTMCTS(
        ply=ply, evaluator=bot, num_parallel=num_parallel, debug=debug, seed=seed
    )
    bot.decision_function = parallel_mcts

    return bot


class HybridTournamentConfig:
    """Configuration for hybrid parallelism in tournaments."""

    def __init__(self, num_cores: Optional[int] = None):
        self.num_cores = num_cores or multiprocessing.cpu_count()
        self.config = get_optimal_parallel_config(self.num_cores)

        # Tournament settings
        self.num_game_processes = self.config["num_game_processes"]
        self.num_parallel_threads = self.config["num_parallel_threads"]

        # MCTS settings
        self.ply = 4
        self.rounds_per_instance = 100
        self.seed = 42

    def get_tournament_settings(self) -> dict:
        """Get settings for tournament.Tournament."""
        return {"processors": self.num_game_processes, "ply_depth": self.ply}

    def get_bot_settings(self) -> dict:
        """Get settings for creating bots with parallel MCTS."""
        return {
            "ply": self.ply,
            "num_parallel": self.num_parallel_threads,
            "seed": self.seed,
        }

    def print_config(self):
        """Print the configuration."""
        print("Hybrid Tournament Configuration:")
        print(f"  CPU Cores: {self.num_cores}")
        print(f"  Game Processes: {self.num_game_processes}")
        print(f"  Parallel Threads per Bot: {self.num_parallel_threads}")
        print(f"  Total Threads: {self.config['total_threads']}")
        print(f"  MCTS Ply: {self.ply}")
        print(f"  Rounds per Instance: {self.rounds_per_instance}")


# Example usage in tournament
def run_hybrid_tournament(
    population_size: int = 15,
    generations: int = 10,
    ply: int = 4,
    num_cores: Optional[int] = None,
):
    """
    Run a tournament with hybrid parallelism.

    This function demonstrates how to integrate parallel MCTS with the
    existing tournament system.
    """
    import core.tournament as tournament
    from search.parallel_tmcts import ParallelTMCTS

    config = HybridTournamentConfig(num_cores)
    config.ply = ply

    print("Setting up hybrid tournament...")
    config.print_config()

    # Create tournament options
    options = {
        "Population": population_size,
        "NumberOfGenerations": generations,
        "ply_depth": ply,
        "NumberOfGamesPerPlayer": 5,
        "printStatus": True,
    }

    # The tournament will use multiprocessing for games
    # Each bot will use parallel MCTS internally
    t = tournament.Generator(options)

    # Modify bots to use parallel MCTS
    for player_id in t.population.current_population:
        bot = t.population.players[player_id].bot
        parallel_mcts = ParallelTMCTS(
            ply=ply,
            evaluator=bot,
            num_parallel=config.num_parallel_threads,
            seed=config.seed,
        )
        bot.decision_function = parallel_mcts

    # Run tournament
    t.Tournament()

    return t


if __name__ == "__main__":
    # Example: Show optimal configuration
    config = HybridTournamentConfig()
    config.print_config()
