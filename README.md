# Zephyr

Zephyr is a checkboard playing program for my 3rd year dissertation.

### Mongo Setup

    Generations
        Count:
            An integer representing the gen count.
        Population:
            A list of Player ID's

    Tournaments
        generationID
        Scores
            {
                PlayerID
                Score
            }
        Games:
            A list of Game ID's

    Games
        Black
        White
        Date
        Time
        FEN
        Result
        Moves

    Players
        ID
        ELO
        Weights
