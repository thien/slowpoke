from __future__ import annotations

import json
import os

import numpy as np

import core.storage as storage


class agentLoader:
    def __init__(self) -> None:
        self.basepath = os.path.join("..", "results")
        self.championFiletype = ".npz"
        self.championsFoldername = "champions"
        self.statisticsFilename = "statistics.json"
        self.systems = []
        self.cacheFilename = "menuCache.json"
        self.check_directory_change()

    def check_directory_change(self) -> None:
        cached = False
        # check if our cache file is there
        if self.cacheFilename in os.listdir(self.basepath):
            # load the file
            cache = self.load_cache()
            # count number of items
            if cache is not False:
                if "count" in cache.keys():
                    if self.count_directory_items() == cache["count"]:
                        if "systems" in cache.keys():
                            self.systems = cache["systems"]
                            cached = True
        if not cached:
            self.rebuild_cache_list()
            # when done, save to cache.
            self.save_cache()

    def load_cache(self) -> None:
        filepath = os.path.join(self.basepath, self.cacheFilename)
        cache = {}
        try:
            f = open(filepath, "r")
            cache = json.load(f)
            f.close()
            return cache
        except Exception:
            return False

    def save_cache(self) -> None:
        filepath = os.path.join(self.basepath, self.cacheFilename)
        cache = {"count": self.count_directory_items(), "systems": self.systems}
        if os.path.isfile(filepath):
            os.remove(filepath)
        with open(filepath, "w") as outfile:
            json.dump(cache, outfile)
        return True

    def count_directory_items(self) -> int:
        count = 0
        for system in os.listdir(self.basepath):
            sysdir = os.path.join(self.basepath, system)
            if os.path.isdir(sysdir):
                if self.championsFoldername in os.listdir(sysdir):
                    champPath = os.path.join(sysdir, self.championsFoldername)
                    for entry in os.listdir(champPath):
                        count += 1
        return count

    def load_coefficients(self, filepath: str) -> object:
        return json.load(filepath)

    def detect_agent_files(self, path: str) -> list:
        if os.path.isdir(path):
            contents = os.listdir(path)
            if self.statisticsFilename in contents:
                if self.championsFoldername in contents:
                    p = os.listdir(os.path.join(path, self.championsFoldername))
                    if len(p) > 0:
                        return True
        return False

    def get_latest_agent(self, agentPath: str) -> object:
        """
        returns path to the latest agent.
        """
        champPath = os.path.join(agentPath, "champions")
        directory = os.listdir(champPath)
        directory = sorted(
            [int(i.replace(self.championFiletype, "")) for i in directory]
        )
        latestAgent = str(directory[-1]) + self.championFiletype
        return os.path.join(champPath, latestAgent)

    def get_num_champions(self, path: str) -> int:
        return len(os.listdir(os.path.join(path, self.championsFoldername)))

    def scrape_stats(self, directory: str) -> list:
        statFilepath = os.path.join(directory, self.statisticsFilename)
        statistics = None

        try:
            f = open(statFilepath, "r")
            statistics = json.load(f)
            f.close()
        except Exception:
            return False

        ply_depth = str(statistics[0]["stats"][2][1])

        # finds the best agent.
        bestScore = 0
        bestGeneration = 0
        endScore = 0
        for j in range(len(statistics)):
            k = statistics[j]["stats"]
            generation = 0
            for i in k:
                if i[0] == "Generation":
                    generation = str(i[1].replace("/200", ""))
                if i[0] == "Cummulative Score":
                    if float(i[1]) > bestScore:
                        bestGeneration = generation
                        bestScore = round(float(i[1]), 2)
                    else:
                        endScore = round(float(i[1]), 2)

        stats = {
            "bestScore": bestScore,
            "bestGeneration": bestGeneration,
            "ply_depth": ply_depth,
            "OldestGen": self.get_num_champions(directory),
            "endScore": endScore,
        }
        return stats

    def rebuild_cache_list(self) -> list:
        # lets find all the items in the directory
        counter = 0
        print()
        for i in os.listdir(self.basepath):
            counter += 1
            print(
                "Loading Files in Directory: "
                + str(counter)
                + "/"
                + str(len(self.basepath))
                + "\r",
                end="",
            )
            agentPath = os.path.join(self.basepath, i)
            containsAgent = self.detect_agent_files(agentPath)
            if containsAgent:
                stats = self.scrape_stats(agentPath)
                stats["Name"] = i
                stats["latestAgentFile"] = self.get_latest_agent(agentPath)
                # only keep systems with long enough tings
                if stats["OldestGen"] > 1:
                    self.systems.append(stats)
                stats["baseDir"] = os.path.join(self.basepath, stats["Name"])
                stats["ChampDir"] = os.path.join(
                    stats["baseDir"], self.championsFoldername
                )
        print("", end="")
        # sort files by score
        self.systems = sorted(self.systems, key=lambda x: x["endScore"])[::-1]

    def load_agent_ui(self) -> None:
        print("Select the system to load, by the index:")
        chosen = False
        while not chosen:
            for j in range(len(self.systems)):
                i = self.systems[j]
                print("Index:" + str(j), "\t\t", end="")
                print("Ply:" + i["ply_depth"] + "\t", end="")
                print("#Champs:", str(i["OldestGen"]) + "\t", end="")
                print("Hi-Score:", str(i["bestScore"]) + "\t", end="")
                print("End-Score:", str(i["endScore"]) + "\t", end="")
                print("Foldername:", i["Name"])
            index = input("Select the Index: ")
            if int(index) not in [i for i in range(len(self.systems))]:
                print("This is an invalid input, please try again.")
            else:
                chosen = True
                system = self.systems[int(index)]
                return system

    @staticmethod
    def load_specific_agent(system):
        chosenAnswer = False
        option = False
        agentID = system["OldestGen"] - 1
        while not chosenAnswer:
            print("Would you like to choose a specific Agent?")
            print("By Default, it would choose the latest agent.")
            answer = input("Y/N: ")
            if answer in "ynYN":
                chosenAnswer = True
                if answer in "yY":
                    option = True
            else:
                print("You chose an invalid option, please try again.")
                print("---")

        if option:
            chosenAgent = False
            while not chosenAgent:
                print(
                    "Please choose a generation from 1 to "
                    + str(system["OldestGen"] + 1)
                )
                gen_id = input("Generation: ")
                try:
                    gen_id = int(gen_id)
                    if (gen_id < agentID + 1) and (gen_id > 0):
                        agentID = gen_id
                        chosenAgent = True
                except Exception:
                    print("That's an invalid response. Please try again..")
        return agentID

    def find_the_best(self, ply: int = 1) -> object:
        bestScore = 0
        bestSystem = None
        for i in self.systems:
            # print(i)
            if int(i["ply_depth"]) == ply:
                # now we iterate through the best score.
                if i["bestScore"] >= bestScore:
                    bestSystem = i
                    bestScore = i["bestScore"]
        return bestSystem

    def load_statistics_file(self, system: str) -> dict:
        filepath = os.path.join(system["baseDir"], self.statisticsFilename)
        stats = {}
        try:
            f = open(filepath, "r")
            stats = json.load(f)
            f.close()
            return stats
        except Exception:
            return False

    def load_agent_weights(self, system: str, pid: int) -> object:
        # Try .npz first (compressed numpy format)
        npz_path = os.path.join(system["ChampDir"], str(pid) + ".npz")
        if os.path.isfile(npz_path):
            try:
                data = storage.load_champion_npz(npz_path)
                return data.get("coefficients")
            except Exception:
                pass

        # Fallback to .json (legacy format)
        json_path = os.path.join(system["ChampDir"], str(pid) + ".json")
        try:
            with open(json_path, "r") as f:
                agent = json.load(f)
            return np.array(agent[str(pid)]["coefficents"])
        except Exception:
            print(f"I can't load agent {pid} from {system.get('ChampDir', '?')}")
            return False


if __name__ == "__main__":
    al = agentLoader()
    # al.rebuild_cache_list()
    # system = al.load_agent_ui()
    # theid = al.load_specific_agent(system)
    # print(system)
    # print("chosen", theid)
