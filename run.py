import tournament as t

def run():
    configpath = "config.json"
    ga = t.Generator(configpath,100, 15)
    ga.runGenerations()

if __name__ == "__main__":
    run()