import numpy as np
import matplotlib.pyplot as plt

with open('output.txt') as f:
    content = f.readlines()


omega = content[0].split('-')
omega = omega[1:]
p = []
for i in omega:
    p.append(-float(i))

theta = np.arange(10, 81, 10)

plt.plot(theta, p)
plt.show()
