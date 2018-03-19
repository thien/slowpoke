import core.checkers as checkers
import agents.agent as agent
import core.mongo as mongo

import sys

Black, White, empty = 0, 1, -1

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


baseOptions = {
    'show_dialog' : True,
    'show_board' : True,
    'human_white' : False,
    'human_black' : False,
    'clear_screen_on_end' : True,
    'preload_moves' : []
}

def printPerspectiveBoard(B, options):
    if options['clear_screen_on_end']:
        print('\033c', end=None)

    # set default colour.
    blackPOV = None

    if options['human_black'] or options['human_white']:
        # we know that a person is playing.
        if options['human_black'] and options['human_white']:
            # both of them are playing, swap when needed.
            if B.turnCount % 2 != 0:
                # blacks turn
                if options['human_black']:
                    blackPOV = True
            else:
                if options['human_white']:
                    # generate board sprite
                    blackPOV = False
        else:
            if options['human_black']:
                blackPOV = True
            else:
                blackPOV = False
    else:
        # set a default colour.
        blackPOV = True

        # whites turn
    print(B.printBoard(blackPOV))

def playGame(black_player, white_player, options=baseOptions):
    B = checkers.CheckerBoard()
    current_player = B.active

    if len(options['preload_moves']) > 0:
        for i in options['preload_moves']:
            # get the index of the move we're going to use
            moveIndex = B.get_move_strings().index(i[1])
            move = B.get_moves()[moveIndex]
            # make move
            B.make_move(move)
    
    choice = 0
    # take as input agents.
    while not B.is_over():
        if options['show_board']:
            printPerspectiveBoard(B, options)
            # print(B.pdn['Moves'])
            # print(B.moves)
            # print(B.moves)
        if  B.turnCount % 2 != choice:
            if options['show_dialog']:
                print("blacks turn")
            B.make_move(black_player.make_move(B, Black))
        else:
            if options['show_dialog']:
                print("whites turn")
            B.make_move(white_player.make_move(B, White))
        # If jumps remain, then the board will not update current player
        if B.active == current_player:
            if options['show_dialog']:
                print ("Jumps must be taken.")
            continue
        else:
            current_player = B.active

    if options['show_board']:
        print (B)
        B.getWinnerMessage()
    return B

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
    db = None
    if dbURI != False:
        db = mongo.Mongo()
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
    if dbURI != False:
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
            B.make_move(blackCPU.make_move(B, Black))
        else:
            B.make_move(whiteCPU.make_move(B, White))
        if B.active == current_player:
            # Jumps must be taken; don't assign the next player.
            continue
        else:
            current_player = B.active

        # print board.
        if debug != False:
            debugPrint(debug['printBoard'], B)
        # store the game to MongoDB.
        # db.update('games', mongoGame_id, B.pdn)
    # once game is done, update the pdn with the results and return it.
    # db.update('games', mongoGame_id, B.pdn)
    return B.pdn

# -----------

def main():
    # handlePlayerOption()
    # play2Player()
    print("You shouldn't be able to load this program directly. It is only called.")
    print("Terminating..")
    # slowpokeGame()

if __name__ == '__main__':
    try:
        status = main()
        sys.exit(status)
    except KeyboardInterrupt:
        print ("Game terminated.")
        sys.exit(1)
