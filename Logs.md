# Logs

## 18/12/2017

- Currently having a test run to detect mongo errors.
- Implementing Default Options as a parameter for run.py
- Note that I will need to consider plyDepth as an option parameter for easy testing
- will also need to write to a local database because a live mongo writing is a bit messy rn.
- Default Options parameter now implemented
- PlyDepth paramter now added
- Error for champ games:

    Traceback (most recent call last):
    File "run.py", line 17, in <module>
        run()
    File "run.py", line 14, in run
        t.runGenerations()
    File "/Users/thien/Documents/GitHub/zephyr/tournament.py", line 301, in runGenerations
        self.runChampions()
    File "/Users/thien/Documents/GitHub/zephyr/tournament.py", line 167, in runChampions
        results = self.pool.map(self.poolChampGame, champGames)
    File "/usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/lib/python3.6/multiprocessing/pool.py", line 266, in map
        return self._map_async(func, iterable, mapstar, chunksize).get()
    File "/usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/lib/python3.6/multiprocessing/pool.py", line 644, in get
        raise self._value
    File "/usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/lib/python3.6/multiprocessing/pool.py", line 424, in _handle_tasks
        put(task)
    File "/usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/lib/python3.6/multiprocessing/connection.py", line 206, in send
        self._send_bytes(_ForkingPickler.dumps(obj))
    File "/usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/lib/python3.6/multiprocessing/reduction.py", line 51, in dumps
        cls(buf, protocol).dump(obj)
    File "/usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/lib/python3.6/multiprocessing/pool.py", line 528, in __reduce__
        'pool objects cannot be passed between processes or pickled'
    NotImplementedError: pool objects cannot be passed between processes or pickled

- will need to implement a proper round robin algorithm.
- champ games error is now fixed
- currently made the champ games primitive to see relative immediate performance. should be improved using Dantchev's description of premier league
- need to write up code for dashboard instead of dumping a bunch of random shit on the terminal
- made ELO redundant for now

## 19/12/2017

- currently working on updating the status
- could simplify the ID's of each agent with just a number; would shorten the amount of strings!
- simplified ID strings
- created initial status messages, will need to test this offline due to instabilities
- will need to write champion coefficents to a json file.
- debug console is completed now.
- There seems to be an issue which may be the case for running it on a mac:
 
    Traceback (most recent call last):
    File "run_lite.py", line 18, in <module>
        run()
    File "run_lite.py", line 15, in run
        t.runGenerations()
    File "/Users/thien/Documents/GitHub/zephyr/tournament.py", line 233, in runGenerations
        self.runChampions()
    File "/Users/thien/Documents/GitHub/zephyr/tournament.py", line 320, in runChampions
        pool = multiprocessing.Pool(processes=self.processors)
    File "/usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/lib/python3.6/multiprocessing/context.py", line 119, in Pool
        context=self.get_context())
    File "/usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/lib/python3.6/multiprocessing/pool.py", line 174, in __init__
        self._repopulate_pool()
    File "/usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/lib/python3.6/multiprocessing/pool.py", line 239, in _repopulate_pool
        w.start()
    File "/usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/lib/python3.6/multiprocessing/process.py", line 105, in start
        self._popen = self._Popen(self)
    File "/usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/lib/python3.6/multiprocessing/context.py", line 277, in _Popen
        return Popen(process_obj)
    File "/usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/lib/python3.6/multiprocessing/popen_fork.py", line 20, in __init__
        self._launch(process_obj)
    File "/usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/lib/python3.6/multiprocessing/popen_fork.py", line 66, in _launch
        parent_r, child_w = os.pipe()
    OSError: [Errno 24] Too many open files 

- This will need to be tested on a windows machine to make sure.
- This has been verified to be only an issue on macs. This isn't really a priority since the gruntwork will be on the windows machine.
- However, another error has been found:

    Traceback (most recent call last):
    File ".\run_lite.py", line 18, in <module>
        run()
    File ".\run_lite.py", line 15, in run
        t.runGenerations()
    File "C:\Users\Tnguy\Documents\GitHub\zephyr\tournament.py", line 231, in runGenerations
        population = ga.generateNewPopulation(players, self.population)
    File "C:\Users\Tnguy\Documents\GitHub\zephyr\genetic.py", line 32, in generateNewPopulation
        parent = roulette(players)
    File "C:\Users\Tnguy\Documents\GitHub\zephyr\genetic.py", line 89, in roulette
        probability = i.points / overallFitness
    ZeroDivisionError: division by zero

- This will need to be worked on tomorrow or sometime.
- will need to tackle the mongo no connection print.
- fixed pipe issue
- Should update to mongo periodically instead of per process..
- will also need to read the blondie24 paper to see their implementations.
- fixed mongo printing error
- initial file commit for heavy load (for testing)
- There's an issue with printing dates on windows. Mean gen time and remaining gen time are absent for some reason.

## 20/12/2017

- need to save champion to a file
- will also need to update the file every generation.
- update the mutation algorithm
- update the crossover algorithm
 - should have multiple kids
- next population generation method:
    for a population of 15:
        first 5 is the top 5 players of the previous gen
        next 4 is the children of 1st and 2nd place from previous gen
        next 4 is the children of 2nd and 3rd place from previous gen
        last 2 is direct mutation of 4th and 5th place from previous gen.

- currently, mira runs heavy loads with generations finishing every 30 minutes.

TODO:

- optimise neural network (done)
- create ravel and unravel algorithms (flatten the coefficents of the neural network) (DONE)
- update mutation algorithm
- write champs to a file
- update mongo writing
- update champion performance algorithm
- write current generation to file
- load from file

## 21/12/2017

- integrate population class (DONE)
- update mutation algorithm (DONE)
- update champion performance algorithm (DONE)
- write champs to a file (DONE)
- play champions from 2nd gen to determine performance (DONE)
- write current generation to file
- load from file
- update mongo writing
- print player ID's in debug log
- build web interface

https://arxiv.org/abs/1712.06567

## 22/12/2017

- improve neural network
- improve champion algorithm
- improve crossover
- improve mutation
- increase ply depth when moves are forced

     "The neural network topology chosen for the evolutionary checkers experiments. The net- works have 32 input nodes (blue) that correspond to the 32 possible positions on the board. The two hidden layers (green) comprise 40 and 10 hidden nodes, respectively. All input nodes are connected directly to the output node (red) with a weight of 1.0. Bias terms affect each hidden and output node as a threshold term (not pictured)."

- score should not consider previous champs weighting with the further previous champs.

## 03/01/2018

    After every neural network in the population played its five games as the red player, the fifteen neural networks with the highest point totals were saved as parents for the next generation. The remaining fifteen neural networks with the lowest point totals were killed o¤, victims of natural selection. Then, to begin the next generation, each surviving parent was copied to create a new o¤spring neural net- work, in which each weight of every o¤spring was varied at random, and the competition was started anew with the thirty members of the population.
    Playing at the edge of ai

    The only detail about our evolutionary process that I haven’t pro- vided concerns how o¤spring neural networks were created from their parents.You’ve probably heard of a “bell curve.”4 Kumar and I im- plemented a variation process whereby each weight of a surviving par- ent neural network was mutated using a bell curve.
The details of how to accomplish this procedure are presented in technical papers that we’ve published.5 The essence of the idea is to use a method that’s likely to generate values for an o¤spring’s weights

read p177!

## 06/01/2018

Crossover Algorithm:

  def crossOver(self, cpu1, cpu2, child1, child2, index1, index2):
    """
    Basic Crossover Algorithm for the GA.
    """
    mother = self.getWeights(cpu1)
    father = self.getWeights(cpu2)

    # pythonic crossover
    child1W = np.append(np.append(father[:index1], mother[index1:index2]), father[index2:])
    child2W = np.append(np.append(mother[:index1], father[index1:index2]), mother[index2:])
    
    # create new children with it
    self.setWeights(child1, child1W)
    self.setWeights(child2, child2W)

    # return the pair of children
    return (child1,child2)  

for each layer's weights:
    n = number of weights and biases in a given layer
    index1 = random integer[0 to n]
    index2 = random integer[0 to n]
    if index2 < index1:
        swap index1 and index2's values
    Weights(child1) = father[0 to index1] + mother[index1 to index 2] + father[index2 to n]
    Weights(child2) = mother[0 to index1] + father[index1 to index 2] + mother[index2 to n]
