import glob
import numpy as np
import matplotlib.pyplot as plt
from pysph.solver.utils import load

data = load('*_0.hdf5')
particle_arrays = data['arrays']
ball = particle_arrays['ball']
index_of_middle_particle = len(ball.x) / 2

# get files in the directory
files = glob.glob("*100*.hdf5")
print(files)

y = []
for file in files:
    print(file)
    data = load(file)
    particle_arrays = data['arrays']
    ball = particle_arrays['ball']
    y.append(ball.y[index_of_middle_particle])

time = np.linspace(0, 1, len(y))
plt.plot(time, y)
plt.show()
