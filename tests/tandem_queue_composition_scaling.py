from core import Simulator
from examples.mm1k import MM1k
from functools import reduce
from operator import add
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

k = 10
samples = 10
t = 100

if __name__ == "__main__":
    res = []
    for components in range(2, 6):
        queues = [MM1k(k=k, _service_rate=2, _arrival_rate=1 if i == 0 else 2) for i in range(components)]
        queue = reduce(add, queues)
        queue.shared_events = [
            [('com', queues[i]), ('arr', queues[i + 1])] for i in range(components - 1)
        ]
        tmp = []
        for s in range(samples):
            start = perf_counter()
            Simulator(queue).run(until=t)
            stop = perf_counter()
            tmp.append(stop - start)
        res.append(np.mean(tmp))

    plt.plot(range(2, 6), res)
    plt.xlabel(r'$n$')
    plt.ylabel('Runtime')
    plt.title('Composition M/M/1/{0}, sim time={2}'.format(k, samples, t))
    plt.savefig('timing results/Composition k={0} sim time = {1}.png'.format(k, t))
    plt.show()
