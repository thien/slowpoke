# import timeit
import numpy as np

import datetime
x = np.random.random((5000, 5000)) - 0.5

u = None


print("tanh method:")
start = datetime.datetime.now().timestamp()
for i in range(100):
    u = np.tanh(x)
end = datetime.datetime.now().timestamp() - start
print(end)
print(u)

print("maxn method:")
start = datetime.datetime.now().timestamp()
for i in range(100):
    u = np.maximum(x, 0, x)
end = datetime.datetime.now().timestamp() - start
print(end)
print(u)

print("index method:")
start = datetime.datetime.now().timestamp()
for i in range(100):
    x[x<0] =0
    u = x
end = datetime.datetime.now().timestamp() - start
print(end)
print(u)
print("max method:")
start = datetime.datetime.now().timestamp()
for i in range(100):
    u = np.maximum(x, 0)
end = datetime.datetime.now().timestamp() - start
print(end)
print(u)
print("multiplication method:")
start = datetime.datetime.now().timestamp()
for i in range(100):
    u = x * (x > 0)
end = datetime.datetime.now().timestamp() - start
print(end)
print(u)
print("abs method:")
# %timeit -n10 (abs(x) + x) / 2
start = datetime.datetime.now().timestamp()
for i in range(100):
    u = (abs(x) + x) / 2
end = datetime.datetime.now().timestamp() - start
print(end)
print(u)