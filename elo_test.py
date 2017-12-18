import elo


inputElo = 1613
exp  = elo.expected(inputElo, 1609)
exp += elo.expected(inputElo, 1477)
exp += elo.expected(inputElo, 1388)
exp += elo.expected(inputElo, 1586)
exp += elo.expected(inputElo, 1720)

# # exp == 2.867
# A lost match #1, draw match #2, wins #3 and #4 and loses #5. Therefore the player's actual score is (0 + 0.5 + 1 + 1 + 0) = 2.5.

# We can now use this to calculate the new Elo rating for A:

results = [0,0.5,1,1,0]
results = sum(results)

result = elo.elo(inputElo, exp, results, k=32)  # 1601
print(int(result))

print("--")


inputElo = 1200

exp = elo.expected(inputElo, 1200)
exp += elo.expected(inputElo, 1204)
exp += elo.expected(inputElo, 1289)
exp += elo.expected(inputElo, 1241)
exp += elo.expected(inputElo, 1220)

results = [0,0.5,1,1,0]
results = sum(results)
result = elo.elo(inputElo, exp, results, k=32)  # 1601
print(result)

print("--")

inputElo = 1200

exp = elo.expected(inputElo, 1300)
exp += elo.expected(inputElo, 1300)

results = [0,1]
results = sum(results)
result = elo.elo(inputElo, exp, results, k=32)  # 1601
print(result)

exp = elo.expected(inputElo, 1300)
results = [1]
results = sum(results)
result = elo.elo(result, exp, results, k=32)  # 1601
print(result)
