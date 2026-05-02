"""Human — interactive checkers player with console interface."""

from __future__ import annotations

from typing import Any, Optional


class Human:
    """Interactive human player that reads moves from the console."""

    def __init__(self) -> None:
        """Initialise Human player."""
        self.null = 0

    def printStatus(self, B: Any) -> None:
        """Print the current board state and move information."""
        print('\033c', end=None)
        print("--------")
        print(B)
        print(B.pdn)
        print(B.AIBoardPos)
        print("--------")

    def move_function(self, B: Any, colour: Optional[int] = None) -> int:
        """Read a move from the console and return it."""
        legal_moves = B.get_moves()
        if B.jump:
            print("Make jump.")
            print("")
        else:
            print("Turn %i" % B.turnCount)
            print("")
        for (i, move) in enumerate(B.get_move_strings()):
            print("Move " + str(i) + ": " + move)
        while True:
            move_idx = input("Enter your move number: ")
            try:
                move_idx = int(move_idx)
            except ValueError:
                is_move = move_idx
                move_idx = -1
                for i in range(len(B.get_move_strings())):
                    if B.get_move_strings()[i] == is_move:
                        move_idx = i
            if move_idx in range(len(legal_moves)):
                break
            else:
                print("Please input a valid move number.")
                continue
        return legal_moves[move_idx]
