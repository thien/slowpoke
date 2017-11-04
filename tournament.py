import agent
import checkers
import slowpoke as sp
import operator
from random import randint


# TODO: Consider Multithreading
# TODO: Store results and player information into db

Black, White, empty = 0, 1, -1

def botGame(blackCPU, whiteCPU):
    # assign colours
    blackCPU.assignColour(Black)
    whiteCPU.assignColour(White)

    B = checkers.CheckerBoard()
    current_player = B.active

    choice = 0
    # Start the game loop.
    while not B.is_over():
        print("Game is currently on move:", B.turnCount)
        # game loop!
        if  B.turnCount % 2 != choice:
            botMove = blackCPU.make_move(B)
            B.make_move(botMove)
            if B.active == current_player:
                # print ("Jumps must be taken.")
                continue
            else:
                current_player = B.active
        else:
            botMove = whiteCPU.make_move(B)
            B.make_move(botMove)
            if B.active == current_player:
                # print ("Jumps must be taken.")
                continue
            else:
                current_player = B.active
        print(B)
    # once game is done, return the pgn
    return B.pdn

def sampleGame():
    # initiate agent for Slowpoke (we'll need this so we can make competitions.)
    bot1 = sp.Slowpoke()
    cpu1 = agent.Agent(bot1)

    bot2 = sp.Slowpoke()
    cpu2 = agent.Agent(bot2)

    # make them play a game.
    results = botGame(cpu1, cpu2)
    # print results.
    print(results)

def Tournament(tournamentSize, gameRounds):
    # initiate bots.
    participants = []
    for i in range(tournamentSize):
        bot = sp.Slowpoke()
        cpu = agent.Agent(bot)
        # add it to the list.
        participants.append(cpu)

    # make bots play each other.
    for i in range(len(participants)):
        for j in range(gameRounds):

            # choose a random number between 1 and participants.
            rand = randint(0, len(participants)-1)
            while (rand == i):
                rand = randint(0, len(participants)-1)

            # cpu1 is black, cpu2 is white
            cpu1 = participants[i]
            cpu2 = participants[rand]
            results = botGame(cpu1, cpu2)
            # allocate points for each player.
            if results["Winner"] == Black:
                participants[i].points += 1
                participants[rand].points -= 2
            elif results["Winner"] == White:
                participants[i].points -= 2
                participants[rand].points += 1
                
    # order the players by how good they are.
    players_ranking = sorted(participants, key=operator.attrgetter('points'))

    # return top 3
    for i in range(len(players_ranking)):
        print(players_ranking[i].points)


# def crossOver(cpu1, cpu2):


Tournament(6,5)