import pysph
import numpy as np
import matplotlib.pyplot as plt
from pysph.solver.utils import load
import sys


# get files in the directory
files = pysph.solver.utils.get_files(sys.argv[1])

data = load(files[0])
particle_arrays = data['arrays']
ball = particle_arrays['ball']
index_of_middle_particle = 0

y = []
v = []
for i in range(0, len(files)):
    data = load(files[i])
    particle_arrays = data['arrays']
    ball = particle_arrays['ball']
    y.append(ball.y[index_of_middle_particle])
    v.append(ball.v[index_of_middle_particle])

time = np.linspace(0, 0.5, len(y))
plt.plot(time, y)
plt.axis([0, 0.5, 0, 0.6])
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.grid(True, linestyle='-')
# plt.savefig("bouncing_ball_position_liu.svg")
# plt.savefig("bouncing_ball_position_liu.png")

plt.show()


plt.plot(time, v)
plt.axis([0, 0.5, -4, 4])
plt.xlabel("Time (s)")
plt.ylabel("Vertical velocity (m/s)")
plt.grid(True, linestyle='-')
# plt.savefig("bouncing_ball_velocity_liu.svg")
# plt.savefig("bouncing_ball_velocity_liu.png")

plt.show()
