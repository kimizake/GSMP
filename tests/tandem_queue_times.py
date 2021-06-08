from core import Simulator
from mm1 import MM1
from collections import deque

import numpy as np
import pandas as pd
import scipy.stats as st

t = 1000
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
    simulation.run(until=t, plugin=stream)
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
    }, end - start


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
        res, runtime = func(util, util, util)

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

    # df1.to_csv("tandem queue results/Tandem queue - 1st MM1 response times {}.csv".format(t))
    # df2.to_csv("tandem queue results/Tandem queue - 2nd MM1 response times {}.csv".format(t))
    # df3.to_csv("tandem queue results/Tandem queue - 3rd MM1 response times {}.csv".format(t))

