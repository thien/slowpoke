"""Slowbro — evolved Slowpoke with fused 32-input NN. Tournament agent."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

_lib_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _lib_dir not in sys.path:
    sys.path.insert(0, _lib_dir)

from agents import minimax_draw, minimax_empty, minimax_lose, minimax_win

from .evaluator.neural import NeuralNetwork


class Slowbro:
    """Tournament agent with fused 32-input NN. Supports serial and parallel TMCTS."""

    def __init__(
        self,
        plyDepth: int = 4,
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
            plyDepth: MCTS search depth.
            layers: NN architecture (default [32,40,10,1]).
            weights: Optional flat coefficient vector (auto-fuses legacy [91,...] if needed).
            use_mlx: Enable MLX GPU evaluation.
            use_parallel: Use ParallelTMCTS instead of serial TMCTS.
            num_parallel: Number of parallel threads (for parallel mode).
            debug: Enable debug output.
        """
        self.debug = debug
        self.ply = plyDepth
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
        self.enableCache = True

        # Load weights if provided (handles legacy fusing)
        if weights is not None and len(weights) > 0:
            self.loadWeights(weights)

        # Decision function — parallel or serial TMCTS
        if use_parallel:
            from decision.parallel_tmcts import ParallelTMCTS

            self.decisionFunction = ParallelTMCTS(
                plyDepth, self, num_parallel=num_parallel, debug=debug
            )
        else:
            import decision.tmcts as tmcts

            self.decisionFunction = tmcts.TMCTS(plyDepth, self, debug=debug)

    def loadWeights(self, weights: np.ndarray) -> None:
        """Load weights, auto-fusing legacy [91,40,10,1] coefficients if needed."""
        if len(weights) != self.nn.lenCoefficents:
            from .evaluator.subsquares import make_fused_nn

            legacy_nn = NeuralNetwork([91, 40, 10, 1])
            legacy_nn.loadCoefficents(weights)
            fused_nn = make_fused_nn(legacy_nn)
            self.nn = fused_nn
            if self.use_mlx:
                self.nn._init_mlx_weights()
        else:
            self.nn.loadCoefficents(weights)

    def move_function(self, board: Any, colour: int) -> int:
        """Entry point called by Agent.make_move()."""
        return self.decisionFunction.Decide(board, colour)

    def evaluate_board(self, board: Any, colour: int) -> float:
        """Evaluate board position using direct 32-input neural network."""
        if board.is_over():
            if board.winner != minimax_empty:
                return minimax_win if board.winner == colour else minimax_lose
            return minimax_draw

        boardStatus = board.getBoardPosWeighted(colour, self.pieceWeights)

        if self.enableCache:
            hashd = tuple(boardStatus)
            if hashd in self.cache:
                return self.cache[hashd]

        val = self.nn.compute(boardStatus)

        if self.enableCache:
            self.cache[hashd] = val
        return val
