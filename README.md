# Slowpoke

Slowpoke is a checkboard playing program for my 3rd year dissertation. It's inspired by Blondie24 but also includes a series of modifications that allow it to be a system that can be taken seriously in 2017. It revolves around GANNs (Genetic Algorithms / Neural Networks) and moves are evaluated using a modified Monte-Carlo Tree Search. 

### Training

Training is called by running the python file:

    python3 simulate.py heavy

The program will attempt to utilise as many cores that the computer running the program has. I'm currently running this on a 128-core machine, which takes around 13 hours to finish (200 generations, 15 players per generation, 6ply).

### Playing a Champion

    python3 play.py

The program above also allows arguments; so you can quickly test the system:

    python3 play.py b=slowpoke w=slowpoke ply=8
