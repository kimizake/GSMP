from core import Simulator
from examples.mm1 import MM1

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

t = 100000
alpha = 0.95

samples = 10
arrival = 1


def run(util1, util2, util3):
    service1 = arrival / util1
    service2 = arrival / util2
    service3 = arrival / util3

    q1 = MM1(service_rate=service1, arrival_rate=arrival)
    q2 = MM1(service_rate=service2, arrival_rate=arrival)
    q3 = MM1(service_rate=service3, arrival_rate=arrival)

    return Simulator(q1 + q2 + q3).run(until=t, estimate_probability=True)


def collect_data(util):

    probabilities = {
        1: [],
        2: [],
        3: []
    }

    for i in range(samples):
        data = run(util, util, util)
        print('sim {0} done of {1}'.format(i + 1, samples))

        states, probs = zip(*data)
        from operator import itemgetter
        x = max(states, key=itemgetter(0))[0]
        y = max(states, key=itemgetter(1))[1]
        z = max(states, key=itemgetter(2))[2]

        grid = np.zeros(shape=(x + 1, y + 1, z + 1))
        for state, prob in data:
            grid[state] = prob

        probabilities[1].append(np.sum(grid, axis=(1, 2)))
        probabilities[2].append(np.sum(grid, axis=(0, 2)))
        probabilities[3].append(np.sum(grid, axis=(0, 1)))

    x = min(len(i) for i in probabilities[1])
    y = min(len(i) for i in probabilities[2])
    z = min(len(i) for i in probabilities[3])

    ci1 = [
        st.t.interval(
            alpha=alpha, df=samples - 1,
            loc=np.mean([probabilities[1][i][state] for i in range(samples)]),
            scale=st.sem([probabilities[1][i][state] for i in range(samples)])
        ) for state in range(x)
    ]

    ci2 = [
        st.t.interval(
            alpha=alpha, df=samples - 1,
            loc=np.mean([probabilities[2][i][state] for i in range(samples)]),
            scale=st.sem([probabilities[2][i][state] for i in range(samples)])
        ) for state in range(y)
    ]

    ci3 = [
        st.t.interval(
            alpha=alpha, df=samples - 1,
            loc=np.mean([probabilities[3][i][state] for i in range(samples)]),
            scale=st.sem([probabilities[3][i][state] for i in range(samples)])
        ) for state in range(z)
    ]

    fig1, ax1 = plt.subplots()
    ax1.set_xlabel(r'$\rho$')
    ax1.set_ylabel('Probability')
    ax1.set_title('Queue 1, time = {}'.format(t))
    l1, u1 = zip(*ci1)
    ax1.fill_between(range(x), l1, u1, color='b', alpha=.5, label=r'$\rho$={}'.format(util))
    ax1.plot(range(x), [util**i * (1 - util) for i in range(x)], 'r--', label='steady state')
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.set_xlabel(r'$\rho$')
    ax2.set_ylabel('Probability')
    ax2.set_title('Queue 2, time = {}'.format(t))
    l2, u2 = zip(*ci2)
    ax2.fill_between(range(y), l2, u2, color='b', alpha=.5, label=r'$\rho$={}'.format(util))
    ax2.plot(range(y), [util ** i * (1 - util) for i in range(y)], 'r--', label='steady state')
    ax2.legend()

    fig3, ax3 = plt.subplots()
    ax3.set_xlabel(r'$\rho$')
    ax3.set_ylabel('Probability')
    ax3.set_title('Queue 3, time = {}'.format(t))
    l3, u3 = zip(*ci3)
    ax3.fill_between(range(z), l3, u3, color='b', alpha=.5, label=r'$\rho$={}'.format(util))
    ax3.plot(range(z), [util ** i * (1 - util) for i in range(z)], 'r--', label='steady state')
    ax3.legend()

    fig1.savefig('tandem queue results/1st pdf util={0} time={1}.png'.format(util, t))
    fig2.savefig('tandem queue results/2nd pdf util={0} time={1}.png'.format(util, t))
    fig3.savefig('tandem queue results/3rd pdf util={0} time={1}.png'.format(util, t))

    plt.show()


if __name__ == "__main__":
    for u in [.2, .5, .8]:
        collect_data(u)
        print('u {} done'.format(u))
