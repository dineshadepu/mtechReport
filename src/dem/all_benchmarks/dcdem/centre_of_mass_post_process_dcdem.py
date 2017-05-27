from pysph.solver.utils import get_files, iter_output
import matplotlib.pyplot as plt
from pysph.solver.utils import load
from numpy import genfromtxt
import numpy as np


def get_com(file):
    data = load(file)
    t = data['solver_data']['t']
    pa = data['arrays']['cylinders']
    total_mass = np.sum(pa.m)
    moment_x = np.sum(pa.m * pa.x)
    moment_y = np.sum(pa.m * pa.y)
    x_com = moment_x / total_mass
    y_com = moment_y / total_mass
    return t, x_com / (26 * 1e-2), y_com / (26 * 1e-2)


def post_processing(folder):
    files = get_files(folder)
    t = []
    x = []
    y = []
    for file in files:
        current_t, xcm, ycm = get_com(file)
        if current_t > 0:
            current_t = current_t - 0.05
        t.append(current_t)
        x.append(xcm)
        y.append(ycm)
    print("Done")
    my_data = genfromtxt('xcom_dataset.csv', delimiter=',')
    plt.plot(t, x, label='x/L DEM')
    plt.plot(my_data[:, 0], my_data[:, 1], label="x/L experiment")

    my_data = genfromtxt('ycom_dataset.csv', delimiter=',')
    plt.plot(my_data[:, 0], my_data[:, 1], label="y/L experiment")
    plt.plot(t, y, label='y/L DEM')
    plt.xlabel("time")
    plt.ylabel("centre of mass(x, y) / L")
    plt.legend()
    # plt.show()
    plt.savefig('/Users/harishraja/iitbreport/doc/rnd/' + str(folder[:-1]) + '.png')
