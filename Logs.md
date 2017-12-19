# Logs

18/12/2017

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

19/12/2017

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