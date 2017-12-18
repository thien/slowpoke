# Slowpoke

Slowpoke is a checkboard playing program for my 3rd year dissertation.

### Running

To run the program, all that is needed to run is the following line in the terminal:

    python3 tournament.py

### Mongo Setup

    generation
        Count: ✔️
            An integer representing the gen count.
        Population: ✔️
            A list of Player ID's
        Players:
            PlayerID
            Score
        games:
            a list of gameID's
    games:
        gameID
        Black
        White
        Date
        Time
        FEN
        Result
        Moves

    players ✔️
        ID
        ELO
        Weights


### Champion Algorithm

- for champion:
    if its the 1st generation:
        default at 1000 elo
    else:
        with its elo score, play other champions; compute score difference:
        - 2 games for each champion (black and white)
        
    make it play the last 5 bots. 6 games each. (3 as black, 3 as white.)
    calculate elo.

- compute elo difference between last champion and current champion to measure relative performance increase.

- each champion needs to store generation count on it.