from examples.tandem_queue import Tandem_queue2, Tandem_queue3, Tandem_queue4, Tandem_queue5
from time import perf_counter
from core import Simulator

import matplotlib.pyplot as plt
import numpy as np

k = 10
samples = 10
t = 100

if __name__ == "__main__":
    res = []

    def runs(queue):
        tmp = []
        for s in range(samples):
            start = perf_counter()
            Simulator(queue).run(until=t)
            stop = perf_counter()
            print('simulation {0} of {1} done'.format(s + 1, samples))
            tmp.append(stop - start)
        res.append(np.mean(tmp))

    runs(Tandem_queue2(k=k))
    print('tandem queue 2 finished')
    runs(Tandem_queue3(k=k))
    print('tandem queue 3 finished')
    runs(Tandem_queue4(k=k))
    print('tandem queue 4 finished')
    runs(Tandem_queue5(k=k))
    print('tandem queue 5 finished')

    plt.plot([2, 3, 4, 5], res)
    plt.xlabel(r'$n$')
    plt.ylabel('Runtime')
    plt.title('Normal M/M/1/{0}, sim time={1}'.format(k, t))
    plt.savefig('timing results/Normal k={0} sim time = {1}.png'.format(k, t))
    plt.show()
