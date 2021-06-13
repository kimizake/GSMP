from examples.mm1 import MM1
from core import Compose, Simulator

import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

samples = 10

q1 = MM1()
q2, q3, q4, q5 = MM1(), MM1(), MM1(), MM1()

c1 = Compose(q1)
c2 = q1 + q2
c3 = q1 + q2 + q3
c4 = q1 + q2 + q3 + q4
c5 = q1 + q2 + q3 + q4 + q5

tl = np.linspace(1000, 10000, num=10)

sn = Simulator(q1)

s1 = Simulator(c1)
s2 = Simulator(c2)
s3 = Simulator(c3)
s4 = Simulator(c4)
s5 = Simulator(c5)


if __name__ == "__main__":

    rn = []
    r1 = []
    r2 = []
    r3 = []
    r4 = []
    r5 = []

    for t in tl:
        tn, t1, t2, t3, t4, t5 = [], [], [], [], [], []
        for sample in range(samples):
            start = perf_counter()
            sn.run(until=t)
            end = perf_counter()
            tn.append(end - start)

            start = perf_counter()
            s1.run(until=t)
            end = perf_counter()
            t1.append(end - start)

            start = perf_counter()
            s2.run(until=t)
            end = perf_counter()
            t2.append(end - start)

            start = perf_counter()
            s3.run(until=t)
            end = perf_counter()
            t3.append(end - start)

            start = perf_counter()
            s4.run(until=t)
            end = perf_counter()
            t4.append(end - start)

            start = perf_counter()
            s5.run(until=t)
            end = perf_counter()
            t5.append(end - start)
            print('{0} done sample {1} of {2}'.format(t, sample + 1, samples))
        rn.append(np.mean(tn))
        r1.append(np.mean(t1))
        r2.append(np.mean(t2))
        r3.append(np.mean(t3))
        r4.append(np.mean(t4))
        r5.append(np.mean(t5))

    plt.plot(tl, rn, label='M/M/1 GSMP')
    plt.plot(tl, r1, label='M/M/1 Composed')
    plt.plot(tl, r2, label='2 parallel M/M/1')
    plt.plot(tl, r3, label='3 parallel M/M/1')
    plt.plot(tl, r4, label='4 parallel M/M/1')
    plt.plot(tl, r5, label='5 parallel M/M/1')
    plt.xlabel('Simulation time')
    plt.ylabel('Runtime')
    plt.title('Runtime performance')
    plt.legend()

    plt.savefig('timing results/Runtime test.png')
    plt.show()
