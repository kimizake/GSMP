import matplotlib.pyplot as plt

from core import Simulator
from examples.mm1 import MM1
from examples.mm1ps import MM1PS

arrival = 1


def run(util, time):
    service = arrival / util
    fifo = MM1(service_rate=service)
    ps = MM1PS(service_rate=service)

    s1 = Simulator(fifo)
    s2 = Simulator(ps)

    return s1.run(until=time, estimate_probability=True), s2.run(until=time, estimate_probability=True)


if __name__ == "__main__":
    utils = [.2, .5, .8]
    times = [100, 1000, 10000]

    for t in times:
        for u in utils:
            d1, d2 = run(u, t)
            fifo_states, fifo_probs = zip(*d1)
            ps_states, ps_probs = zip(*d2)

            fig, ax = plt.subplots()
            ax.set_xlabel('state')
            ax.set_ylabel('probability')
            ax.set_title('M/M/1 FIFO vs. PS for simulation time {}'.format(t) + r' $\rho$ = {}'.format(u))
            ax.plot(fifo_states, fifo_probs, label='FIFO')
            ax.plot(ps_states, ps_probs, label='PS')
            ax.legend()
            fig.savefig('mm1 results/mm1 fifo v ps t={0} u={1}.png'.format(t, u))

            print('done')
    plt.show()
