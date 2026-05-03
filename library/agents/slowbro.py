"""Slowbro — evolved Slowpoke with fused 32-input NN. Tournament agent."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from agents import minimax_draw, minimax_empty, minimax_lose, minimax_win

from .evaluator.neural import NeuralNetwork


class Slowbro:
    """Tournament agent with fused 32-input NN. Supports serial and parallel TMCTS."""

    def __init__(
        self,
        ply_depth: int = 4,
        layers: Optional[List[int]] = None,
        weights: Optional[np.ndarray] = None,
        use_mlx: bool = False,
        use_parallel: bool = False,
        num_parallel: int = 4,
        debug: bool = False,
    ) -> None:
        """
        Initialise Slowbro agent.

        Args:
            ply_depth: MCTS search depth.
            layers: NN architecture (default [32,40,10,1]).
            weights: Optional flat coefficient vector (auto-fuses legacy [91,...] if needed).
            use_mlx: Enable MLX GPU evaluation.
            use_parallel: Use ParallelTMCTS instead of serial TMCTS.
            num_parallel: Number of parallel threads (for parallel mode).
            debug: Enable debug output.
        """
        self.debug = debug
        self.ply = ply_depth
        self.layers = list(layers if layers is not None else [32, 40, 10, 1])

        # Piece weights for board evaluation — tuned for [32] input
        self.pieceWeights = {
            "Black": 1,
            "White": -1,
            "empty": 0,
            "blackKing": 1.5,
            "whiteKing": -1.5,
        }

        # Create neural network with direct 32-input architecture
        self.nn = NeuralNetwork(self.layers, use_mlx=use_mlx)
        self.use_mlx = use_mlx and self.nn._use_mlx

        # Optional evaluation cache (used by serial TMCTS path)
        self.cache = {}
        self.enable_cache = True

        # Load weights if provided (handles legacy fusing)
        if weights is not None and len(weights) > 0:
            self.load_weights(weights)

        # Decision function — parallel or serial TMCTS
        if use_parallel:
            from decision.parallel_tmcts import ParallelTMCTS

            self.decision_function = ParallelTMCTS(
                ply_depth, self, num_parallel=num_parallel, debug=debug
            )
        else:
            import decision.tmcts as tmcts

            self.decision_function = tmcts.TMCTS(ply_depth, self, debug=debug)

    def load_weights(self, weights: np.ndarray) -> None:
        """Load weights, auto-fusing legacy [91,40,10,1] coefficients if needed."""
        if len(weights) != self.nn.len_coefficients:
            from .evaluator.subsquares import make_fused_nn

            legacy_nn = NeuralNetwork([91, 40, 10, 1])
            legacy_nn.load_coefficients(weights)
            fused_nn = make_fused_nn(legacy_nn)
            self.nn = fused_nn
            if self.use_mlx:
                self.nn._init_mlx_weights()
        else:
            self.nn.load_coefficients(weights)

    def move_function(self, board: Any, colour: int) -> int:
        """Entry point called by Agent.make_move()."""
        return self.decision_function.decide(board, colour)

    def evaluate_board(self, board: Any, colour: int) -> float:
        """Evaluate board position using direct 32-input neural network."""
        if board.is_over():
            if board.winner != minimax_empty:
                return minimax_win if board.winner == colour else minimax_lose
            return minimax_draw

        boardStatus = board.get_board_pos_weighted(colour, self.pieceWeights)

        if self.enable_cache:
            hashd = tuple(boardStatus)
            if hashd in self.cache:
                return self.cache[hashd]

        val = self.nn.compute(boardStatus)

        if self.enable_cache:
            self.cache[hashd] = val
        return val
