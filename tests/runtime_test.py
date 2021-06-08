from mm1 import MM1
from core import Compose, Simulator

import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

normal = MM1(
    adjacent_states=lambda s: [1] if s == 0 else [s - 1, s + 1]
)

compose = Compose(normal)

linspace = np.linspace(0, 10000, num=11)

snormal = Simulator(normal)

scompose = Simulator(compose)

if __name__ == "__main__":

    tnormal = []
    tcompose = []

    for t in linspace:
        start = perf_counter()
        snormal.run(until=t)
        end = perf_counter()
        tnormal.append(end - start)

        start = perf_counter()
        scompose.run(until=t)
        end = perf_counter()
        tcompose.append(end - start)
        print('{} done'.format(t))

    plt.plot(linspace, tnormal, 'b', label='Regular M/M/1')
    plt.plot(linspace, tcompose, 'r--', label='Composed M/M/1')
    plt.xlabel('Simulation time')
    plt.ylabel('Runtime')
    plt.title('Runtime performance')
    plt.legend()

    plt.savefig('timing results/Runtime test.png')
    plt.show()
