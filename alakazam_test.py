import checkers
import agent
import sys

"""
Allows the user to decide what gameplay they want to play on.
"""
def handlePlayerOption():
    slowpokeGame()

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
    B.checkWinner()
    return 0

def slowpokeGame():
    agent_module = 'alakazam';
    __import__(agent_module)
    agent_module = sys.modules[agent_module]
    cpu_1 = agent.CheckersAgent(agent_module.move_function)
    agent_module = sys.modules[agent_module]
    cpu_2 = agent.CheckersAgent(agent_module.move_function)

        
    # start game.
    B = checkers.CheckerBoard()
    current_player = B.active

    while not B.is_over():
        B.make_move(cpu_1.make_move(B))
        if B.active == current_player:
            continue
        current_player = B.active
        while B.active == current_player and not B.is_over():
            B.make_move(cpu_2.make_move(B))
        current_player = B.active
    B.checkWinner()
    print(B.pgn)
    return 0


def main():
    handlePlayerOption()
    # play2Player()

if __name__ == '__main__':
    try:
        status = main()
        sys.exit(status)
    except KeyboardInterrupt:
        print ("Game terminated.")
        sys.exit(1)
