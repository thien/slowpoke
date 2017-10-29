import checkers
import agent
import sys


"""
Displays the ASCII Board and other various information.
"""
def printStatus(B):
    print("--------")
    print (B)
    print(B.pgn)
    print(B.AIBoardPos)
    # print(B.moves())
    print("--------")

"""
Function to handle player move logic.
"""
def makeHumanMove(B):
    legal_moves = B.get_moves()
    if B.jump:
        print ("Make jump.")
        print ("")
    else:
        print ("Turn %i" % B.turnCount)
        print ("")
    for (i, move) in enumerate(B.get_move_strings()):
        print ("Move " + str(i) + ": " + move)
    while True:
        move_idx = input("Enter your move number: ")
        try:
            move_idx = int(move_idx)
        except ValueError:
            print ("Please input a valid move number.")
            continue
        if move_idx in range(len(legal_moves)):
            break
        else:
            print ("Please input a valid move number.")
            continue

    B.make_move(legal_moves[move_idx])


def playSlowpoke():
    # import slowpoke :)
    import slowpoke as sp

    # Initiate a slowpoke
    bot = sp.Slowpoke()

    # initiate agent for Slowpoke (we'll need this so we can)
    # make competitions.
    cpu = agent.Agent(bot)

    choice = 0
    while True:
        choice = input("Enter 0 to go first and 1 to go second: ")
        try:
            choice = int(choice)
            break
        except ValueError:
            print ("Please input 0 or 1.")
            continue

    # Assign colour to CPU
    # black is 0, white is 1.
    Black, White = 0, 1
    if choice == 0:
        cpu.assignColour(Black)
    else:
        cpu.assignColour(White)

    B = checkers.CheckerBoard()
    current_player = B.active
    print ("Black moves first.")
    # Start the game loop.
    while not B.is_over():
        print (B)
        if  B.turnCount % 2 != choice:
            makeHumanMove(B)
            # If jumps remain, then the board will not update current player
            if B.active == current_player:
                print ("Jumps must be taken.")
                continue
            else:
                current_player = B.active
        else:
            botMove = cpu.make_move(B)
            B.make_move(botMove)
            if B.active == current_player:
                print ("Jumps must be taken.")
                continue
            else:
                current_player = B.active
    print (B)
    B.getWinnerMessage()
    return 0



def main():
    playSlowpoke()
    # handlePlayerOption()

if __name__ == '__main__':
    try:
        status = main()
        sys.exit(status)
    except KeyboardInterrupt:
        print ("Game terminated.")
        sys.exit(1)
