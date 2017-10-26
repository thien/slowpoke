import checkers
import agent
import sys

"""
Allows the user to decide what gameplay they want to play on.
"""
def handlePlayerOption():
    n = -1
    while not n in [0, 1, 2]:
        n = input("How many human players? (0, 1, 2): ")
        try:
            n = int(n)
        except ValueError:
            print ("Please input 0, 1, or 2.")
    if n == 2:
        play2Player()
    elif n == 1:
        playCPU()
    else:
        BotGame()

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

def play2Player():
    B = checkers.CheckerBoard()
    print ("Black moves first.")
    current_player = B.active
    while not B.is_over():
        printStatus(B)
        makeHumanMove(B)
        # If jumps remain, then the board will not update current player
        if B.active == current_player:
            print ("Jumps must be taken.")
            continue
        else:
            current_player = B.active
    # print for the last time before showing the winner.
    print(B)
    B.getWinnerMessage()

    return 0

def playCPU():
    agent_module = input("Enter name of agent module: ");
    __import__(agent_module)
    agent_module = sys.modules[agent_module]
    cpu = agent.CheckersAgent(agent_module.move_function)
    choice = 0
    while True:
        choice = input("Enter 0 to go first and 1 to go second: ")
        try:
            choice = int(choice)
            break
        except ValueError:
            print ("Please input 0 or 1.")
            continue
    B = checkers.CheckerBoard()
    current_player = B.active
    print ("Black moves first.")
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
            B.make_move(cpu.make_move(B))
            if B.active == current_player:
                print ("Jumps must be taken.")
                continue
            else:
                current_player = B.active
    print (B)
    B.getWinnerMessage()
    return 0

def slowpokeGame():
    agent_module = "alakazam";
    __import__(agent_module)
    agent_module = sys.modules[agent_module]
    cpu_1 = agent.CheckersAgent(agent_module.move_function)
    # agent_module = input("Enter name of second agent module: ");
    # __import__(agent_module)
    agent_module = sys.modules[agent_module]
    cpu_2 = agent.CheckersAgent(agent_module.move_function)
    # debug = input("Would you like to step through game play? [Y/N]: ")
    # debug = 1 if debug.lower()[0] == 'y' else 0
        
    # start game.
    B = checkers.CheckerBoard()
    current_player = B.active
    if debug:
        print ("sorry not ready")
        return 0
    else:
        while not B.is_over():
            B.make_move(cpu_1.make_move(B))
            if B.active == current_player:
                continue
            current_player = B.active
            while B.active == current_player and not B.is_over():
                B.make_move(cpu_2.make_move(B))
            current_player = B.active
        B.getWinnerMessage()
        print(B.pgn)
        return 0

def BotGame():
    agent_module = input("Enter name of first agent module: ");
    __import__(agent_module)
    agent_module = sys.modules[agent_module]
    cpu_1 = agent.CheckersAgent(agent_module.move_function)
    agent_module = input("Enter name of second agent module: ");
    __import__(agent_module)
    agent_module = sys.modules[agent_module]
    cpu_2 = agent.CheckersAgent(agent_module.move_function)
    debug = input("Would you like to step through game play? [Y/N]: ")
    debug = 1 if debug.lower()[0] == 'y' else 0
        
    # start game.
    B = checkers.CheckerBoard()
    current_player = B.active
    if debug:
        print ("sorry not ready")
        return 0
    else:
        while not B.is_over():
            B.make_move(cpu_1.make_move(B))
            if B.active == current_player:
                continue
            current_player = B.active
            while B.active == current_player and not B.is_over():
                B.make_move(cpu_2.make_move(B))
            current_player = B.active
        B.getWinnerMessage()
        print(B.pgn)
        return 0

def main():
    # handlePlayerOption()
    play2Player()

if __name__ == '__main__':
    try:
        status = main()
        sys.exit(status)
    except KeyboardInterrupt:
        print ("Game terminated.")
        sys.exit(1)
