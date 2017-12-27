import numpy as np
# ting = np.random.random_sample([3, 2])
# # normalise to a range from -0.2 to 0.2
# ting = (ting-0.5) * 0.4 

# print(ting)
# print(np.random.random_sample(10))

# we have a 1d array of the items
numberOfWeights = 1741
tau = 1 / np.sqrt( 2 * np.sqrt(numberOfWeights))

ting = np.random.random_sample([numberOfWeights])
# ting = np.ones([numberOfWeights])
print(ting)
ting = tau * ting
ting = np.exp(ting)
print(ting)

ine = np.random.random_sample([numberOfWeights, -0.2, 0.2])
print("ine", ine)
ine = ine * ting
ine = np.clip(ine, -1, 1)
print("ine", ine)
# we want to make a random multiplier of -0.2 to 0.2 or a random sample


# 