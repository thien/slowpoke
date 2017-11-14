# Zephyr

Zephyr is a checkboard playing program for my 3rd year dissertation.

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
