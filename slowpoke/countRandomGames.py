from __future__ import annotations

from slowpoke import play
import multiprocessing

# initiate players
p1 = play.load_player_class("magikarp")
p2 = play.load_player_class("magikarp")

# initiate options
gameOptions = {
    "show_dialog": False,
    "show_board": False,
    "human_white": False,
    "human_black": False,
    "clear_screen_on_end": True,
    "preload_moves": [],
}

numberOfGames = 100000
games = [[p1, p2] for i in range(numberOfGames)]


def f(i: int) -> None:
    return len(play.run_game(i[0], i[1], gameOptions).pdn["Moves"])


p = multiprocessing.Pool(multiprocessing.cpu_count())
moves = p.map(f, games)

print("Average Number of Moves:", sum(moves) / len(moves))
