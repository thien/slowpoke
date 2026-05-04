"""
Agent

This represents a computer player;
It'll contain information about its ranking,
and its move function.

It also has a default ELO.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Optional

from core import checkers


class Agent:
    def __init__(
        self,
        bot: Any,
        agent_id: Optional[str] = None,
        initial_elo: Optional[float] = None,
    ) -> None:
        self.bot = bot
        self.elo = (
            initial_elo if initial_elo is not None else 100
        )  # Updated by Population.allocate_points()
        self.points = 0
        self.champ_range = 0
        self.champ_score = 0
        self.move_function = bot.move_function
        self.colour = None
        self.games_played = 0  # Track games played for K-factor calibration

        # check for ID
        if agent_id is not None:
            self.set_id(agent_id)
        else:
            self.gen_id()

        # genomic properties.
        self.origin = []
        # store parent's ID
        self.parents = []
        self.generate_origin()

    def __getstate__(self) -> dict:
        """Serialize agent to a plain dict (for checkpoint)."""
        try:
            nn_mode = getattr(self.bot.nn, "_mode", "standard")
            raw = self.bot.nn.get_all_coefficients()
            if nn_mode == "neat":
                coefficients = bytes(raw).decode("utf-8")  # JSON string
            else:
                coefficients = raw.tolist()  # float list
        except Exception:
            coefficients = None
            nn_mode = "standard"

        bot_type = type(self.bot).__name__

        data: dict = {
            "id": self.id,
            "elo": self.elo,
            "points": self.points,
            "games_played": self.games_played,
            "champ_range": self.champ_range,
            "champ_score": self.champ_score,
            "origin": self.origin,
            "parents": self.parents,
            "bot_type": bot_type,
            "nn_mode": nn_mode,
            "coefficients": coefficients,
        }

        if hasattr(self, "is_baseline"):
            data["is_baseline"] = self.is_baseline
        if hasattr(self, "entity_name"):
            data["entity_name"] = self.entity_name

        return data

    def __setstate__(self, state: dict) -> None:
        """Restore agent from a checkpoint dict."""
        import numpy as np
        from agents.slowbro import Slowbro
        from agents.onix import Onix
        from agents.evaluator.neural import NeuralNetwork

        bot_type = state.get("bot_type", "Slowbro")
        ply = state.get("ply_depth", 4)
        nn_mode = state.get("nn_mode", "standard")

        if bot_type == "Onix":
            bot = Onix(ply_depth=ply)
        else:
            use_mlx = state.get("use_mlx", True)
            use_parallel = state.get("use_parallel_mcts", False)
            num_parallel = state.get("num_parallel", 4)
            debug = state.get("debug", False)

            if nn_mode == "neat":
                from agents.evaluator.genome import Genome

                nn = NeuralNetwork(layer_list=[32, 1], use_mlx=False, mode="neat")
                genome_data = state.get("coefficients")
                if genome_data is not None:
                    nn._genome = Genome.from_dict(json.loads(genome_data))
                bot = Slowbro(
                    ply_depth=ply,
                    use_mlx=False,
                    use_parallel=use_parallel,
                    num_parallel=num_parallel,
                    debug=debug,
                )
                bot.nn = nn
            else:
                bot = Slowbro(
                    ply_depth=ply,
                    use_mlx=use_mlx,
                    use_parallel=use_parallel,
                    num_parallel=num_parallel,
                    debug=debug,
                )
                coeffs = state.get("coefficients")
                if coeffs is not None:
                    bot.nn.load_coefficients(np.array(coeffs, dtype=np.float32))

        self.bot = bot
        self.elo = state.get("elo", 100)
        self.points = state.get("points", 0)
        self.games_played = state.get("games_played", 0)
        self.champ_range = state.get("champ_range", 0)
        self.champ_score = state.get("champ_score", 0)
        self.move_function = bot.move_function
        self.colour = None
        self.origin = state.get("origin", [])
        self.parents = state.get("parents", [])
        self.id = state.get("id")

        if state.get("is_baseline"):
            self.is_baseline = True
        if state.get("entity_name"):
            self.entity_name = state["entity_name"]

    def generate_origin(self) -> None:
        """Generate genesis origin block for evolution tracking."""
        self.origin.append([0, 0, 0])

    def gen_id(self) -> None:
        """Generate a unique ID from NN coefficients or timestamp."""
        try:
            self.id = hashlib.md5(self.bot.nn.get_all_coefficients()).hexdigest()
        except Exception:
            k = str(time.time()).encode("utf-8")
            self.id = hashlib.md5(k).hexdigest()

    def set_id(self, value: str) -> None:
        """Set the agent ID explicitly."""
        self.id = value

    def get_dict(self) -> Dict[str, Any]:
        """Serialize agent state to a dictionary."""
        try:
            weights = self.bot.nn.weights.tolist()
        except AttributeError:
            weights = None
        return {
            "_id": self.id,
            "weights": weights,
            "elo": self.elo,
            "points": self.points,
        }

    def assign_colour(self, colID: int) -> None:
        """Assign colour to agent (0=black, 1=white)."""
        self.colour = colID
        try:
            self.bot.current_colour = colID
        except AttributeError:
            pass

    def make_move(self, board: checkers.CheckerBoard, colour: int) -> int:
        """Make a move by delegating to the bot's move_function."""
        return self.bot.move_function(board, colour)
