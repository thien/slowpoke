"""Slowpoke — legacy agent with 91-input NN using subsquares."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

import decision.minimax as minimax
import decision.tmcts as tmcts
from agents import minimax_draw, minimax_empty, minimax_lose, minimax_win

from .evaluator.neural import NeuralNetwork


class Slowpoke:
    """Legacy agent with 91-input NN using subsquare feature extraction."""

    def __init__(
        self,
        ply_depth: int = 4,
        kingWeight: float = 1.5,
        weights: Optional[List[float]] = None,
        layers: Optional[List[int]] = None,
        isminimax: bool = False,
        debug: bool = False,
        use_mlx: bool = False,
    ) -> None:
        """Initialise Slowpoke agent.

        Args:
            ply_depth: MCTS search depth.
            kingWeight: Relative weight of kings vs men.
            weights: Optional flat coefficient vector.
            layers: NN architecture (default [91, 40, 10, 1]).
            isminimax: Use minimax instead of TMCTS.
            debug: Enable debug output.
            use_mlx: Enable MLX GPU evaluation.
        """
        self.debug = debug
        self.chooseMinimax = isminimax
        self.nn = False
        self.ply = ply_depth
        self.layers = layers if layers is not None else [91, 40, 10, 1]
        self.pieceWeights = {
            "Black": 1,
            "White": -1,
            "empty": 0,
            "blackKing": kingWeight,
            "whiteKing": -kingWeight,
        }

        self.init_neural_network(
            self.layers, weights if weights else [], use_mlx=use_mlx
        )
        self.movesConsidered = []

        self.decision_function = None
        if isminimax:
            self.decision_function = minimax.MiniMax(self.ply, self.evaluate_board)
        else:
            self.decision_function = tmcts.TMCTS(self.ply, self, debug=self.debug)

        self.cache: Dict[Any, float] = {}
        self.enable_cache = True
        self.use_mlx = use_mlx and self.nn._use_mlx
        self._batch_inputs = []
        self._batch_refs = []

    def init_neural_network(
        self,
        layers: List[int],
        weights: Optional[List[float]] = None,
        use_mlx: bool = False,
    ) -> None:
        """Create and optionally load weights into the neural network."""
        self.nn = NeuralNetwork(layers, use_mlx=use_mlx)
        if weights:
            self.load_weights(weights)

    def load_weights(self, weights: np.ndarray) -> None:
        """Load weights into the neural network."""
        self.nn.load_coefficients(weights)

    def move_function(self, board: Any, colour: int) -> int:
        """Make a move by delegating to the decision function."""
        return self.decision_function.decide(board, colour)

    def evaluate_board(self, board: Any, colour: int) -> float:
        """Evaluate board position using neural network.

        If the first layer has 91 inputs, applies subsquares feature extraction
        before passing to the NN. Otherwise uses 32-element input directly.
        """
        if board.is_over():
            if board.winner != minimax_empty:
                if board.winner == colour:
                    return minimax_win
                else:
                    return minimax_lose
            else:
                return minimax_draw

        boardStatus = board.get_board_pos_weighted(colour, self.pieceWeights)

        if self.layers[0] == 91:
            boardStatus = self.nn.subsquares(boardStatus)

        hashd = None
        if self.enable_cache:
            hashd = tuple(boardStatus)
            if hashd in self.cache:
                return self.cache[hashd]

        val = self.nn.compute(boardStatus)

        if self.enable_cache:
            self.cache[hashd] = val
        return val

    def evaluate_board_mlx(self, board: Any, colour: int) -> Any:
        """MLX-native evaluation returning mx.array."""
        if board.is_over():
            if board.winner != minimax_empty:
                if board.winner == colour:
                    return float(minimax_win)
                else:
                    return float(minimax_lose)
            else:
                return float(minimax_draw)

        boardStatus = board.get_board_pos_weighted(colour, self.pieceWeights)

        if self.layers[0] == 91:
            boardStatus = self.nn.subsquares(boardStatus)

        import mlx.core as mx

        return mx.array([float(self.nn.compute(boardStatus))])
