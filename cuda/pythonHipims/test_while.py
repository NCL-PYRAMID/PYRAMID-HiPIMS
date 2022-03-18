import numpy as np
t = 0
dt = 0.8
tout = 5
while t + dt < tout:
    t += dt
    print(t)
else:
    dt = tout - t
    print('finish')
    print(dt)
x = np.array([[0,1],[2,3]])
xx = x[:][1]
print(xx)