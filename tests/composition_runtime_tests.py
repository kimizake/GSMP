from core import Simulator
from mm1 import MM1

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

q1 = MM1(adjacent_states=lambda s: [1] if s == 0 else [s - 1, s + 1])
q2 = MM1(adjacent_states=lambda s: [1] if s == 0 else [s - 1, s + 1])
q3 = MM1(adjacent_states=lambda s: [1] if s == 0 else [s - 1, s + 1])
q4 = MM1(adjacent_states=lambda s: [1] if s == 0 else [s - 1, s + 1])
q5 = MM1(adjacent_states=lambda s: [1] if s == 0 else [s - 1, s + 1])

c2 = q1 + q2
c3 = q1 + q2 + q3
c4 = q1 + q2 + q3 + q4
c5 = q1 + q2 + q3 + q4 + q5

tl = np.linspace(0, 10000, num=11)

s2 = Simulator(c2)
s3 = Simulator(c3)
s4 = Simulator(c4)
s5 = Simulator(c5)

t2 = []
t3 = []
t4 = []
t5 = []

if __name__ == "__main__":
    for t in tl:
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

        print('done')

    plt.plot(tl, t2, label='2 parallel M/M/1')
    plt.plot(tl, t3, label='3 parallel M/M/1')
    plt.plot(tl, t4, label='4 parallel M/M/1')
    plt.plot(tl, t5, label='5 parallel M/M/1')
    plt.xlabel('Simulation time')
    plt.ylabel('Runtime')
    plt.title('Composition Runtime performance')
    plt.legend()

    plt.savefig('timing results/Composition runtime test.png')
    plt.show()
