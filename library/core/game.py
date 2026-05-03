"""Game loop and tournament match functions."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from core import checkers
from agents.agent import Agent
from core.constants import BLACK, WHITE, EMPTY

Black, White, empty = BLACK, WHITE, EMPTY


def printStatus(B: checkers.CheckerBoard) -> None:
    """Print current board status and PGN info."""
    print("--------")
    print(B)
    print(B.pdn)
    print(B.ai_board_pos)
    print("--------")


baseOptions: Dict[str, Any] = {
    "show_dialog": True,
    "show_board": True,
    "human_white": False,
    "human_black": False,
    "clear_screen_on_end": True,
    "preload_moves": [],
}


def print_perspective_board(B: checkers.CheckerBoard, options: Dict[str, Any]) -> None:
    """Print the board from the current player's perspective."""
    if options["clear_screen_on_end"]:
        print("\033c", end=None)

    blackPOV = None
    if options["human_black"] or options["human_white"]:
        if options["human_black"] and options["human_white"]:
            blackPOV = (
                True
                if (B.turn_count % 2 != 0 and options["human_black"])
                else (False if options["human_white"] else None)
            )
        else:
            blackPOV = bool(options["human_black"])
    else:
        blackPOV = True
    print(B.print_board(blackPOV))


def play_game(
    black_player: Agent,
    white_player: Agent,
    options: Optional[Dict[str, Any]] = None,
) -> checkers.CheckerBoard:
    """Play a game between two agents with optional display options.

    Args:
        black_player: Agent playing Black.
        white_player: Agent playing White.
        options: Display and configuration options.

    Returns:
        The final CheckerBoard state after the game ends.
    """
    if options is None:
        options = baseOptions
    B = checkers.CheckerBoard()
    current_player = B.active

    if len(options["preload_moves"]) > 0:
        for i in options["preload_moves"]:
            moveIndex = B.get_move_strings().index(i[1])
            move = B.get_moves()[moveIndex]
            B.make_move(move)

    while not B.is_over():
        if options["show_board"]:
            print_perspective_board(B, options)
        if B.turn_count % 2 != 0:
            if options["show_dialog"]:
                print("blacks turn")
            B.make_move(black_player.make_move(B, Black))
        else:
            if options["show_dialog"]:
                print("whites turn")
            B.make_move(white_player.make_move(B, White))
        if B.active == current_player:
            if options["show_dialog"]:
                print("Jumps must be taken.")
            continue
        else:
            current_player = B.active

    if options["show_board"]:
        print(B)
        B.get_winner_message()
    return B


def debug_print(check: bool, msg: str) -> None:
    """Print debug message if check is True."""
    if check:
        print(msg)


def generate_debug_msg(
    debug: Dict[str, Any], moveCount: int, B: checkers.CheckerBoard
) -> str:
    """Generate a formatted debug status message."""
    gameCountMsg = (
        "Game: "
        + str(debug["gameCount"]).zfill(2)
        + "/"
        + str(debug["totalGames"]).zfill(2)
    )
    moveMsg = "Move: " + str(moveCount).zfill(3)
    GenerationMsg = "Gen: " + str(debug["genCount"]).zfill(3)
    PlayersMsg = "B: " + str(B.pdn["Black"]) + " | W: " + str(B.pdn["White"])
    msg = " | ".join([GenerationMsg, gameCountMsg, moveMsg, PlayersMsg])
    debug_print(debug["printDebug"], msg)
    return msg


def tournament_match(
    blackCPU: Agent,
    whiteCPU: Agent,
    gameID: Union[str, int] = "NULL",
    dbURI: Union[bool, str] = False,
    debug: Union[bool, Dict[str, Any]] = False,
    multiProcessing: bool = False,
) -> Dict[str, Any]:
    """Run a tournament match between two agents.

    Args:
        blackCPU: Agent playing Black.
        whiteCPU: Agent playing White.
        gameID: Identifier for the game.
        dbURI: MongoDB URI string or False.
        debug: Debug configuration or False.
        multiProcessing: Whether running in multiprocessing mode.

    Returns:
        PGN dictionary with game results.
    """
    db = None
    if dbURI:
        import core.mongo as mongo

        db = mongo.Mongo()
        db.initiate(dbURI)

    blackCPU.assign_colour(Black)
    whiteCPU.assign_colour(White)

    B = checkers.CheckerBoard()
    B.set_id(gameID)
    B.set_colours(blackCPU.id, whiteCPU.id)

    if dbURI:
        db.write("games", B.pdn)

    current_player = B.active
    while not B.is_over():
        if debug:
            generate_debug_msg(debug, B.turn_count, B)

        if B.turn_count % 2 != 0:
            B.make_move(blackCPU.make_move(B, Black))
        else:
            B.make_move(whiteCPU.make_move(B, White))
        if B.active == current_player:
            continue
        else:
            current_player = B.active

        if debug and debug.get("print_board"):
            debug_print(True, B)

    return B.pdn
