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