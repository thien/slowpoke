from __future__ import annotations

#!/usr/bin/python
import slowpoke.core.tournament as tournament
import sys
import os
from slowpoke import evaluator
from slowpoke import statistics


def train() -> None:
    from checkers_core import CheckerBoard as _RustCB

    print(f"[checkers-core] Rust backend active ({_RustCB.__module__})")
    options = {
        "mongoConfigPath": "config2.json",
        "Population": 15,
        "debugMode": False,  # if enabled, makes the system play randomly for testing purposes
        "printStatus": True,
        "connectMongo": False,
        "resultsLocation": os.path.join("..", "results"),
    }
    readyBool = False
    verifiedBool = False
    resume_folder = None
    # Check for arguments
    if "--resume" in sys.argv:
        idx = sys.argv.index("--resume")
        if idx + 1 < len(sys.argv):
            resume_folder = sys.argv[idx + 1]
            readyBool = True
            verifiedBool = True
        else:
            print("--resume requires a folder path")
            print("Terminating.")
            return
    elif len(sys.argv) > 1:
        # check arguments
        if "light" in sys.argv:
            print("You are about to load a light simulation.")
            options["ply_depth"] = 1
            options["NumberOfGenerations"] = 200
            verifiedBool = True

        elif "medium" in sys.argv:
            print("You are about to load a medium simulation.")
            options["ply_depth"] = 3
            options["NumberOfGenerations"] = 200
            verifiedBool = True

        elif "heavy" in sys.argv:
            print("You are about to load a heavy simulation.")
            options["ply_depth"] = 6
            options["NumberOfGenerations"] = 200
            verifiedBool = True

        elif "ohno" in sys.argv:
            print("You are about to load a really heavy simulation.")
            options["ply_depth"] = 8
            options["NumberOfGenerations"] = 1500
            verifiedBool = True

        elif "vheavy" in sys.argv:
            print("You are about to load a VERY HEAVY simulation (12 ply).")
            print("This will be extremely computationally intensive!")
            options["ply_depth"] = 12
            options["NumberOfGenerations"] = 500
            verifiedBool = True

        if "debug" in sys.argv:
            print("You are about to load a debug simulation.")
            options["ply_depth"] = 1
            options["debugMode"] = True
            options["NumberOfGenerations"] = 200
            verifiedBool = True

        # parallel threads option
        if "--parallel" in sys.argv:
            idx = sys.argv.index("--parallel")
            if idx + 1 < len(sys.argv):
                try:
                    options["num_parallel"] = int(sys.argv[idx + 1])
                    print(f"Using {options['num_parallel']} parallel threads.")
                except ValueError:
                    print("Invalid parallel count, using default (4).")
                    options["num_parallel"] = 4
            else:
                options["num_parallel"] = 4

        # check for user input
        if verifiedBool:
            print("Are you ready to run? Y/N")
            k = input()

            if k.upper() == "Y":
                print("dank")
                readyBool = True
        else:
            print("You didn't use an available option.")
            print("options: light, medium, heavy, vheavy, debug, ohno")
    else:
        # no arguments loaded; ask user for load type.
        print("You'll need to load some argument into this file. for instance:")
        print("     python3 simulate.py light")
        print("     python3 simulate.py vheavy --parallel 8")
    # run tournament
    if readyBool:
        if resume_folder:
            print(f"Resuming training from {resume_folder}")
            t = tournament.Generator.from_checkpoint(resume_folder)
            options["ply_depth"] = t.ply_depth  # used by evaluator below
        else:
            t = tournament.Generator(options)
        use_tui = "--no-tui" not in sys.argv
        if use_tui:
            from slowpoke.core.tui import TournamentApp

            app = TournamentApp(generator=t)
            app.run()
            if not app.generations_complete:
                print()
                print("=" * 60)
                print("Training did not complete.")
                print(f"To resume, run: python train.py --resume {t.save_location}")
                print("=" * 60)
                return
        else:
            try:
                t.run_generations()
            except KeyboardInterrupt:
                print()
                print("=" * 60)
                print("Training interrupted.")
                print(f"To resume, run: python train.py --resume {t.save_location}")
                print("=" * 60)
                return
        # create statistics
        stats = statistics.Statistics(t.folder_name)
        stats.load_statistics_file()
        stats.save_charts()
        # stats.average_num_moves_per_generation()
        # stats.getLearningRate()
        # stats.timeStatsPerGeneration()

        # evaluate performance
        su = evaluator.Evaluate(t.folder_name, options["ply_depth"])
        su.load_champions()
        games = su.create_games()
        su.evaluate(games)

        # create statistics of Gold Master
        stats.load_gm_file()
        stats.analyse_gm()
        # print that we're done.
        print("DONE!")
    else:
        print("Terminating.")


if __name__ == "__main__":
    train()
