import tournament as t

def run():
    #config variables
    configpath = "config.json"
    numberOfGenerations = 100
    population = 15

    # run tournament
    ga = t.Generator(configpath, numberOfGenerations, population)
    ga.runGenerations()

if __name__ == "__main__":
    run()