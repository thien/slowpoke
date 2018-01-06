import checkers
import agent
import sys
import slowpoke as sp
import mongo
import human

Black, White, empty = 0, 1, -1

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
    print(B.pdn)
    print(B.AIBoardPos)
    # print(B.moves())
    print("--------")

# play 2 player game (human human)
def play2Player():
    cpu_1 = agent.Agent(human.Human())
    cpu_2 = agent.Agent(human.Human())
    playGame(player1, player2)

# play human robot
def playCPU(coefficents=[]):
    p1 = human.Human()
    player = agent.Agent(p1)

    slowpoke1 = sp.Slowpoke(4)
    if len(coefficents) > 0:
        slowpoke1.loadWeights(coefficents)
        print("Loaded coefficents")
    cpu = agent.Agent(slowpoke1)
    
    humanColour = Black
    botColour = White
    choice = 0
    while True:
        choice = input("Enter 0 to go first and 1 to go second: ")
        try:
            choice = int(choice)
            break
        except ValueError:
            print ("Please input 0 or 1.")
            continue
    
    if choice == 0:
        playGame(player, cpu)
    else:
        playGame(cpu, player)

def slowpokeGame(coefficents1, coefficents2):
    slowpoke1 = sp.Slowpoke(4)
    slowpoke2 = sp.Slowpoke(4)
    
    import magikarp
    magi = magikarp.Magikarp()
    
    slowpoke1.loadWeights(coefficents1)
    slowpoke2.loadWeights(coefficents2)

    print("Loaded coefficents")
    cpu_1 = agent.Agent(slowpoke1)
    cpu_2 = agent.Agent(slowpoke2)
    # debug = input("Would you like to step through game play? [Y/N]: ")
    # debug = 1 if debug.lower()[0] == 'y' else 0
    playGame(cpu_1, cpu_2)

def playGame(player1, player2, options={}):
    B = checkers.CheckerBoard()
    current_player = B.active

    choice = 0
    # take as input agents.
    while not B.is_over():
        print (B)
        if  B.turnCount % 2 != choice:
            print("blacks turn")
            B.make_move(player1.make_move(B, White))
        else:
            print("whites turn")
            B.make_move(player2.make_move(B, Black))
        # If jumps remain, then the board will not update current player
        if B.active == current_player:
            print ("Jumps must be taken.")
            continue
        else:
            current_player = B.active

    print (B)
    B.getWinnerMessage()
    return 0

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
            botMove = blackCPU.make_move(B, Black)
            B.make_move(botMove)
        else:
            botMove = whiteCPU.make_move(B, White)
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
    # play2Player()
    playCPU()
    # slowpokeGame()

if __name__ == '__main__':
    try:
        status = main()
        sys.exit(status)
    except KeyboardInterrupt:
        print ("Game terminated.")
        sys.exit(1)
