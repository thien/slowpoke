"""
Slowbro — evolved Slowpoke that uses fused 32-input neural network directly.

The 91-element subsquares vector has been fused into the first-layer weights,
so Slowbro receives 32-element board positions straight from getBoardPosWeighted()
with no subsquares call. Supports both serial TMCTS and parallel TMCTS.

Slowbro is the tournament agent — cleaner, faster, no external NN attachment needed.
"""

import os
import sys

# Ensure library/ is in sys.path for decision.* imports
_lib_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _lib_dir not in sys.path:
    sys.path.insert(0, _lib_dir)

from agents import minimax_draw, minimax_empty, minimax_lose, minimax_win

from .evaluator.neural import NeuralNetwork


class Slowbro:
    def __init__(
        self,
        plyDepth=4,
        layers=None,
        weights=None,
        use_mlx=False,
        use_parallel=False,
        num_parallel=4,
        debug=False,
    ):
        """
        Initialise Slowbro agent.

        Args:
            plyDepth: MCTS search depth
            layers: NN architecture (default [32,40,10,1])
            weights: Optional flat coefficient vector (auto-fuses legacy [91,40,10,1] if needed)
            use_mlx: Enable MLX GPU evaluation
            use_parallel: Use ParallelTMCTS instead of serial TMCTS
            num_parallel: Number of parallel threads (for parallel mode)
            debug: Enable debug output
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

    def loadWeights(self, weights):
        """Load weights, auto-fusing legacy [91,40,10,1] coefficients if needed."""
        if len(weights) != self.nn.lenCoefficents:
            # Legacy [91,40,10,1] weights — fuse into [32,40,10,1] via subsquares matrix
            from .evaluator.subsquares import make_fused_nn

            legacy_nn = NeuralNetwork([91, 40, 10, 1])
            legacy_nn.loadCoefficents(weights)
            fused_nn = make_fused_nn(legacy_nn)
            self.nn = fused_nn
            # Sync MLX if enabled
            if self.use_mlx:
                self.nn._init_mlx_weights()
        else:
            self.nn.loadCoefficents(weights)

    def move_function(self, board, colour):
        """Entry point called by Agent.make_move()."""
        return self.decisionFunction.Decide(board, colour)

    def evaluate_board(self, board, colour):
        """
        Evaluate board position using direct 32-input neural network.
        No subsquares call — the 91->32 fusion is in the weights.
        """
        if board.is_over():
            if board.winner != minimax_empty:
                return minimax_win if board.winner == colour else minimax_lose
            return minimax_draw

        # Get 32-element board vector
        boardStatus = board.getBoardPosWeighted(colour, self.pieceWeights)

        # Check cache (tuple is fast for small arrays)
        if self.enableCache:
            hashd = tuple(boardStatus)
            if hashd in self.cache:
                return self.cache[hashd]

        # Direct NN evaluation — no subsquares
        val = self.nn.compute(boardStatus)

        if self.enableCache:
            self.cache[hashd] = val
        return val
