import checkers
import agent
import sys
import slowpoke as sp
import mongo

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

# -----------

def sampleGame(self):
    """
    Generates a sample game for testing purposes.
    """
    # initiate agent for Slowpoke (we'll need this so we can make competitions.)
    bot1 = sp.Slowpoke()
    cpu1 = agent.Agent(bot1)

    bot2 = sp.Slowpoke()
    cpu2 = agent.Agent(bot2)

    # make them play a game.
    results = self.playGame(cpu1, cpu2)
    # print results.
    print(results)

Black, White, empty = 0, 1, -1

def debugPrint(check, msg):
    if check:
        print(msg)

def generateDebugMsg(debug, moveCount, B):
    gameCountMsg = "Game: " + str(debug['gameCount']).zfill(2) + "/" + str(debug['totalGames']).zfill(2)
    moveMsg = "Move: " + str(moveCount).zfill(3) 
    GenerationMsg = "Gen: " + str(debug['genCount']).zfill(3) 
    PlayersMsg = "B: " + str(B.pdn['Black']) + " | W: " + str(B.pdn['White'])
    msg = [GenerationMsg, gameCountMsg, moveMsg, PlayersMsg]
    msg = ' | '.join(msg)
    debugPrint(debug['printDebug'], msg)

def tournamentMatch(blackCPU, whiteCPU, gameID="NULL", dbURI=False, debug=False, multiProcessing=False):
    # initiate connection to mongoDB
    # this is needed because pymongo screams if you init prior fork.
    db = mongo.Mongo()
    if dbURI != False:
        db.initiate(dbURI)

    # assign colours
    blackCPU.assignColour(Black)
    whiteCPU.assignColour(White)

    # initiate checkerboard.
    B = checkers.CheckerBoard()
    # set the ID for this game.
    B.setID(gameID)
    B.setColours(blackCPU.id, whiteCPU.id)

    # add the game to mongo.
    mongoGame_id = db.write('games', B.pdn)

    # set game settings
    current_player = B.active
    choice = 0
    # Start the game loop.
    while not B.is_over():
        # print move status.
        if debug != False:
            generateDebugMsg(debug, str(B.turnCount), B)

        # game loop!
        if  B.turnCount % 2 != choice:
            botMove = blackCPU.make_move(B)
            B.make_move(botMove)
            if B.active == current_player:
                # Jumps must be taken; don't assign the next player.
                continue
            else:
                current_player = B.active
        else:
            botMove = whiteCPU.make_move(B)
            B.make_move(botMove)
            if B.active == current_player:
                # Jumps must be taken; don't assign the next player.
                continue
            else:
                current_player = B.active
        # print board.
        if debug != False:
            debugPrint(debug['printBoard'], B)
        # store the game to MongoDB.
        db.update('games', mongoGame_id, B.pdn)
    # once game is done, update the pdn with the results and return it.
    db.update('games', mongoGame_id, B.pdn)
    return B.pdn

# -----------

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
