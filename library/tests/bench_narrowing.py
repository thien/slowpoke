"""
Benchmark: Gumbel-Top-K Progressive Narrowing

Measures:
1. Wall clock speedup from reducing root branching factor (7 -> 5/4/3)
2. Visit concentration & diversity (does Gumbel noise preserve exploration?)
3. Temperature sweep — effect of T on exploration vs exploitation
4. K sensitivity — how many moves to keep for optimal speed-quality tradeoff
5. Scaling across simulation budgets

Opening position has 7 legal moves — enough to trigger K=5 narrowing.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time as time_module
import numpy as np
import math
from decision.tmcts import TMCTS
from core.checkers import CheckerBoard

Black = 0


class MockNeuralNetwork:
    """Mock NN with batch evaluation — returns plausible checkers scores."""
    def __init__(self):
        self._use_mlx = True
        self.layer_size = [91, 40, 10, 1]

    def subsquares(self, x):
        """Simplified subsquares — returns 91-element feature vector."""
        x = np.array(x, dtype=float)
        # Pad to 32 elements if needed
        if len(x) < 32:
            x = np.pad(x, (0, 32 - len(x)))
        # Return a basic feature vector: raw board + aggregated stats
        feats = list(x)
        # Add piece counts
        feats.append(np.sum(x > 0) / 12.0)
        feats.append(np.sum(x < 0) / 12.0)
        feats.append(np.abs(np.sum(x)) / 12.0)
        # Add positional features (center control)
        center_squares = [9, 10, 13, 14, 17, 18, 21, 22]
        center_sum = sum(x[i] for i in center_squares if i < len(x)) / 8.0
        feats.append(center_sum)
        # Pad/truncate to 91
        while len(feats) < 91:
            feats.append(0.0)
        return np.array(feats[:91], dtype=np.float32)

    def compute_batch_mlx(self, positions):
        """Simulate NN batch eval: return heuristic position scores."""
        scores = []
        for pos in positions:
            if len(pos) >= 32:
                piece_sum = sum(pos[:32])
            else:
                piece_sum = sum(pos)
            score = np.clip(piece_sum / 12.0, -1.0, 1.0)
            scores.append(score * 0.5)  # Scale down for mock
        return np.array(scores, dtype=np.float32)

    def compute_batch(self, positions):
        return self.compute_batch_mlx(positions)


class MockEval:
    """Evaluator with a mock neural network for MLX-path testing."""
    def __init__(self):
        self.nn = MockNeuralNetwork()

    def evaluate_board(self, board, colour):
        return float(np.random.random() * 2 - 1)


def compute_hhi(stats):
    total = sum(s['plays'] for s in stats.values())
    if total == 0:
        return 0.0
    return sum((s['plays'] / total) ** 2 for s in stats.values())


def compute_entropy(stats):
    """Shannon entropy of visit distribution — higher = more exploration."""
    total = sum(s['plays'] for s in stats.values())
    if total == 0:
        return 0.0
    probs = [s['plays'] / total for s in stats.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def run_once(k_value=None, temperature=None, base_round=300, ply=4, narrowing=True):
    """Run one TMCTS decision with specified narrowing config."""
    board = CheckerBoard()
    ev = MockEval()
    tc = TMCTS(ply=ply, evaluator=ev, debug=False)
    tc.baseRound = base_round
    tc.progressive_narrowing = narrowing
    if k_value is not None:
        tc.progressive_narrowing_k = k_value
    if temperature is not None:
        tc.gumbel_temperature = temperature

    start = time_module.time()
    move = tc.Decide(board, Black)
    elapsed = time_module.time() - start
    return move, elapsed, dict(tc.movesets), tc


# ============================================================
# MAIN BENCHMARK
# ============================================================
print("=" * 70)
print("TMCTS: Gumbel-Top-K Progressive Narrowing — Benchmark")
print("=" * 70)
print()

# Verify board move count
board = CheckerBoard()
full_moves = board.get_moves()
print(f"Opening position: {len(full_moves)} legal moves (threshold K=5)")

# --- A. Cost of NN batch evaluation ---
print("\nA. COST OF NN BATCH EVALUATION AT ROOT")
print("-" * 40)
board2 = CheckerBoard()
ev = MockEval()
tc = TMCTS(ply=1, evaluator=ev, debug=False)
moves = board2.get_moves()
start = time_module.time()
scored = tc._evaluate_moves_batch(board2, moves, Black)
batch_time = time_module.time() - start
print(f"  Root moves: {len(moves)}, NN batch eval: {batch_time*1000:.2f}ms")
if scored:
    scores = [s for _, s in scored]
    print(f"  Score range: [{min(scores):+.3f}, {max(scores):+.3f}]")
    order = "score: " + ", ".join(f"{s:+.2f}" for s in sorted(scores, reverse=True))
    print(f"  {order}")


# --- B. Speed comparison at baseRound=300 ---
print("\nB. WALL CLOCK SPEED — Narrowing vs Baseline (5 trials each)")
print("-" * 50)
print(f"  {'Config':>23}  {'Time':>8s}  {'HHI':>8s}  {'Entropy':>8s}  {'Moves in tree':>14s}")

base_t = 0
results_b = []
for k_val, nflag, label in [
    (None, False, "Baseline (no narrowing)"),
    (7, True, "Top-7 (K=7, T=0.5)"),
    (5, True, "Top-5 (K=5, T=0.5)"),
    (4, True, "Top-4 (K=4, T=0.5)"),
    (3, True, "Top-3 (K=3, T=0.5)"),
]:
    times = []
    hhis = []
    ents = []
    n_moves = 0
    for _ in range(5):
        _, t, stats, _ = run_once(k_value=k_val, narrowing=nflag,
                                 temperature=0.5, base_round=300)
        times.append(t)
        hhis.append(compute_hhi(stats))
        ents.append(compute_entropy(stats))
        n_moves = len(stats)
    avg_t = np.mean(times)
    if "Baseline" in label:
        base_t = avg_t
    speedup = base_t / avg_t
    results_b.append((label, avg_t, np.mean(hhis), np.mean(ents), n_moves, speedup))
    print(f"  {label:>23}  {avg_t:>7.3f}s  {np.mean(hhis):.4f}  {np.mean(ents):>7.3f}  {n_moves:>10d}")

print()
for label, avg_t, hhi, ent, n_moves, sp in results_b:
    print(f"    {label:>23}: {sp:.2f}x speedup vs baseline")


# --- C. Temperature sweep ---
print("\n\nC. TEMPERATURE SWEEP — Effect on Exploration (K=5, 5 trials each)")
print("-" * 50)
print(f"  {'Temp':>6s}  {'HHI':>8s}  {'Entropy':>8s}  {'Time':>8s}  {'Best visit %':>12s}")
for temp in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
    hhis = []
    ents = []
    times = []
    best_pcts = []
    for _ in range(5):
        _, t, stats, tc = run_once(k_value=5, temperature=temp,
                                  narrowing=True, base_round=300)
        times.append(t)
        hhis.append(compute_hhi(stats))
        ents.append(compute_entropy(stats))
        total = sum(s['plays'] for s in stats.values())
        best = max(s['plays'] for s in stats.values()) / total * 100
        best_pcts.append(best)

    print(f"  {temp:>5.1f}  {np.mean(hhis):>7.4f}  {np.mean(ents):>7.3f}"
          f"  {np.mean(times):>7.3f}s  {np.mean(best_pcts):>10.1f}%")


# --- D. Move diversity across runs ---
print("\n\nD. MOVE DIVERSITY — Same K=5, 10 runs each")
print("-" * 50)
for temp in [0.0, 0.5, 2.0]:
    chosen = []
    for _ in range(10):
        move, _, _, _ = run_once(k_value=5, temperature=temp,
                                narrowing=True, base_round=300)
        chosen.append(str(move))
    unique = len(set(chosen))
    print(f"  T={temp:>4.1f}: {unique}/10 distinct best-move selections")


# --- E. Scaling across simulation budgets ---
print("\n\nE. SCALING — Simulation budget (5 trials per config)")
print("-" * 70)
print(f"  {'Rounds':>8s}  {'Baseline':>10s}  {'Top-5':>10s}  {'Top-3':>10s}"
      f"  {'Speedup(5)':>10s}  {'Speedup(3)':>10s}")
budgets = [50, 100, 300, 600, 1000]
for br in budgets:
    _, t_b, _, _ = run_once(k_value=None, narrowing=False, base_round=br)

    t5s = [run_once(k_value=5, narrowing=True, temperature=0.5, base_round=br)[1]
           for _ in range(3)]
    t5 = np.mean(t5s)

    t3s = [run_once(k_value=3, narrowing=True, temperature=0.5, base_round=br)[1]
           for _ in range(3)]
    t3 = np.mean(t3s)

    print(f"  {br:>6d}  {t_b:>8.3f}s  {t5:>8.3f}s  {t3:>8.3f}s"
          f"  {t_b/t5:>8.2f}x  {t_b/t3:>8.2f}x")


print("\n\n" + "=" * 70)
print("Benchmark complete.")
print("=" * 70)
