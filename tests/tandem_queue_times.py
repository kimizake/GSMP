from core import Simulator
from mm1 import MM1
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

t = 100000
alpha = 0.95


def func(util1, util2, util3):
    arrival = 1
    service1 = arrival / util1
    service2 = arrival / util2
    service3 = arrival / util3

    def adjacent_states(s):
        return [1] if s == 0 else [s - 1, s + 1]

    q1 = MM1(adjacent_states=adjacent_states, arrival_rate=arrival, service_rate=service1)
    q2 = MM1(adjacent_states=adjacent_states, arrival_rate=service1, service_rate=service2)
    q3 = MM1(adjacent_states=adjacent_states, arrival_rate=service2, service_rate=service3)

    queue = q1 + q2 + q3
    queue.shared_events = [
        [('com', q1), ('arr', q2)],
        [('com', q2), ('arr', q3)],
    ]

    arrival_times = {
        q1: deque(),
        q2: deque(),
        q3: deque()
    }
    response_times = {
        q1: [],
        q2: [],
        q3: []
    }

    def stream(data):
        event = data['event']
        process = data['process']
        simtime = data['time']

        if event == 'arr':
            if process == q1:
                arrival_times[q1].append(simtime)
            else:
                raise TypeError
        elif event == 'com':
            if process == q1:
                arrival_times[q2].append(simtime)     # shared event
                response_times[q1].append(
                    simtime - arrival_times[q1].popleft()
                )
            elif process == q2:
                arrival_times[q3].append(simtime)     # shared event
                response_times[q2].append(
                    simtime - arrival_times[q2].popleft()
                )
            else:
                response_times[q3].append(
                    simtime - arrival_times[q3].popleft()
                )

    simulation = Simulator(queue)
    from time import perf_counter
    start = perf_counter()
    data = simulation.run(until=t, plugin=stream, estimate_probability=True)
    end = perf_counter()

    def get_res(data, s):
        m = np.mean(data)
        ci = st.norm.interval(alpha=alpha, loc=m, scale=st.sem(data))
        h = ci[1] - m
        return (
            m,
            1 / (s - arrival),
            ci,
            h,
            100 * h / m
        )

    return {
        1: get_res(response_times[q1], service1),
        2: get_res(response_times[q2], service2),
        3: get_res(response_times[q3], service3)
    }, end - start, data


if __name__ == "__main__":
    utils = np.linspace(0.1, 0.9, num=9)

    runtimes = []

    e = []

    m1 = []
    c1 = []
    a1 = []

    m2 = []
    c2 = []
    a2 = []

    m3 = []
    c3 = []
    a3 = []

    for util in utils:
        res, runtime, data = func(util, util, util)

        runtimes.append(runtime)
        print('done')
        e.append(res[1][1])

        m1.append(res[1][0])
        c1.append(res[1][2])
        a1.append(res[1][4])

        m2.append(res[2][0])
        c2.append(res[2][2])
        a2.append(res[2][4])

        m3.append(res[3][0])
        c3.append(res[3][2])
        a3.append(res[3][4])

        if util in [.2, .5, .8]:
            states, probs = zip(*data)
            from operator import itemgetter
            x = max(states, key=itemgetter(0))[0]
            y = max(states, key=itemgetter(1))[1]
            z = max(states, key=itemgetter(2))[2]

            grid = np.zeros(shape=(x + 1, y + 1, z + 1))
            for state, prob in data:
                grid[state] = prob

            probabilities = [
                np.sum(grid, axis=(1, 2)),
                np.sum(grid, axis=(0, 2)),
                np.sum(grid, axis=(0, 1))
            ]

            expected = [
                [(1 - util) * util ** n for n in range(x + 1)],
                [(1 - util) * util ** n for n in range(y + 1)],
                [(1 - util) * util ** n for n in range(z + 1)]
            ]

            fig1, ax1 = plt.subplots()
            ax1.set_title('Queue 1 ' + r'$\rho$ = {0} simulation time = {1}'.format(util, t))
            fig2, ax2 = plt.subplots()
            ax2.set_title('Queue 2 ' + r'$\rho$ = {0} simulation time = {1}'.format(util, t))
            fig3, ax3 = plt.subplots()
            ax3.set_title('Queue 3 ' + r'$\rho$ = {0} simulation time = {1}'.format(util, t))

            ax1.plot(range(x + 1), probabilities[0], 'b', label='actual')
            ax1.plot(range(x + 1), expected[0], 'r--', label='expected')
            ax1.set_xlabel('states')
            ax1.set_ylabel('probability')
            ax1.legend()

            ax2.plot(range(y + 1), probabilities[1], 'b', label='actual')
            ax2.plot(range(y + 1), expected[1], 'r--', label='expected')
            ax2.set_xlabel('states')
            ax2.set_ylabel('probability')
            ax2.legend()

            ax3.plot(range(z + 1), probabilities[2], 'b', label='actual')
            ax3.plot(range(z + 1), expected[2], 'r--', label='expected')
            ax3.set_xlabel('states')
            ax3.set_ylabel('probability')
            ax3.legend()

            fig1.savefig('tandem queue results/1st pdf util={0} time={1}.png'.format(util, t))
            fig2.savefig('tandem queue results/2nd pdf util={0} time={1}.png'.format(util, t))
            fig3.savefig('tandem queue results/3rd pdf util={0} time={1}.png'.format(util, t))

    df1 = pd.DataFrame(data={
        r'$\rho$': utils,
        'exact': e,
        'estimate': m1,
        '{}% confidence interval'.format(100 * alpha): c1,
        'accuracy (%)': a1,
        'runtime': runtimes
    })

    df2 = pd.DataFrame(data={
        r'$\rho$': utils,
        'exact': e,
        'estimate': m2,
        '{}% confidence interval'.format(100 * alpha): c2,
        'accuracy (%)': a2,
        'runtime': runtimes
    })

    df3 = pd.DataFrame(data={
        r'$\rho$': utils,
        'exact': e,
        'estimate': m3,
        '{}% confidence interval'.format(100 * alpha): c3,
        'accuracy (%)': a3,
        'runtime': runtimes
    })

    df1.to_csv("tandem queue results/Tandem queue - 1st MM1 response times {}.csv".format(t))
    df2.to_csv("tandem queue results/Tandem queue - 2nd MM1 response times {}.csv".format(t))
    df3.to_csv("tandem queue results/Tandem queue - 3rd MM1 response times {}.csv".format(t))

    fig1, ax1 = plt.subplots()
    ax1.set_title('Queue 1 simulation time {}'.format(t))
    ax1.set_xlabel(r'$\rho$')
    ax1.set_ylabel('mean response time')
    ax1.fill_between(utils, [x[0] for x in c1], [x[1] for x in c1], color='b', alpha=.5, label='actual')
    ax1.plot(utils, e, 'r--', label='exact')
    ax1.legend()
    fig1.savefig('tandem queue results/1st response time time={}.png'.format(t))

    fig2, ax2 = plt.subplots()
    ax2.set_title('Queue 2 simulation time {}'.format(t))
    ax2.set_xlabel(r'$\rho$')
    ax2.set_ylabel('mean response time')
    ax2.fill_between(utils, [x[0] for x in c2], [x[1] for x in c2], color='b', alpha=.5, label='actual')
    ax2.plot(utils, e, 'r--', label='exact')
    ax2.legend()
    fig2.savefig('tandem queue results/2nd response time time={}.png'.format(t))

    fig3, ax3 = plt.subplots()
    ax3.set_title('Queue 3 simulation time {}'.format(t))
    ax3.set_xlabel(r'$\rho$')
    ax3.set_ylabel('mean response time')
    ax3.fill_between(utils, [x[0] for x in c3], [x[1] for x in c3], color='b', alpha=.5, label='actual')
    ax3.plot(utils, e, 'r--', label='exact')
    ax3.legend()
    fig3.savefig('tandem queue results/3rd response time time={}.png'.format(t))

    plt.show()

