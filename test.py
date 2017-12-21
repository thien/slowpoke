import numpy as np
ting = np.random.random_sample([3, 2])
# normalise to a range from -0.2 to 0.2
ting = (ting-0.5) * 0.4 

print(ting)
print(np.random.random_sample(10))