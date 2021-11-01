from core import Compose, Simulator
from examples.mm1 import MM1

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

samples = 10
t = 100

if __name__ == "__main__":
    components = [1, 10, 25, 50, 75, 100]
    queues = [
        Compose(*[MM1() for j in range(i)]) for i in components
    ]
    times = []
    for queue in queues:
        res = []
        for sample in range(samples):
            start = perf_counter()
            Simulator(queue).run(until=t)
            stop = perf_counter()
            res.append(stop - start)
            print(sample)
        print('queue done')
        times.append(np.mean(res))

    print(times)
    plt.plot(components, times)
    plt.xlabel('components')
    plt.ylabel('runtime')
    plt.title('Composition Runtime')
    plt.savefig('timing results/Composition runtime test.png')
    plt.show()
