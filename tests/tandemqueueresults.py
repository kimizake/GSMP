from core import Simulator
from mm1 import MM1
from collections import deque

import numpy as np
import scipy.stats as st

time = 5000
alpha = 0.90


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

    class plugin:
        """
        plugin to track the response times
        """
        _arrival_times = {
            q1: deque(),
            q2: deque(),
            q3: deque()
        }
        _response_times = {
            q1: [],
            q2: [],
            q3: []
        }

        def stream(self, data):
            event = data['event']
            process = data['process']
            runtime = data['time']

            if event == 'arr':
                if process == q1:
                    self._arrival_times[q1].append(runtime)
                else:
                    raise TypeError
            elif event == 'com':
                if process == q1:
                    self._arrival_times[q2].append(runtime)     # shared event
                    self._response_times[q1].append(
                        runtime - self._arrival_times[q1].popleft()
                    )
                elif process == q2:
                    self._arrival_times[q3].append(runtime)     # shared event
                    self._response_times[q2].append(
                        runtime - self._arrival_times[q2].popleft()
                    )
                else:
                    self._response_times[q3].append(
                        runtime - self._arrival_times[q3].popleft()
                    )

        def get(self):
            return self._response_times

    simulation = Simulator(queue)
    plugin = plugin()
    simulation.run(until=time, plugin=plugin.stream)

    def get_res(data, s):
        m = np.mean(data)
        return (
            m, 1 / (s - arrival),
            st.norm.interval(alpha=alpha, loc=m, scale=st.sem(data))
        )

    responses = plugin.get()
    return {
        1: get_res(responses[q1], service1),
        2: get_res(responses[q2], service2),
        3: get_res(responses[q3], service3)
    }


if __name__ == "__main__":
    utils = np.linspace(0.1, 0.9, num=9)
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots()
    ax1.set_xlabel(r'$\rho$')
    ax1.set_ylabel('response time')
    ax1.set_title('Queue 1')
    mrt1 = []
    ex1 = []
    lrt1 = []
    urt1 = []

    fig2, ax2 = plt.subplots()
    ax2.set_xlabel(r'$\rho$')
    ax2.set_ylabel('response time')
    ax2.set_title('Queue 2')
    mrt2 = []
    ex2 = []
    lrt2 = []
    urt2 = []

    fig3, ax3 = plt.subplots()
    ax3.set_xlabel(r'$\rho$')
    ax3.set_ylabel('response time')
    ax3.set_title('Queue 3')
    mrt3 = []
    ex3 = []
    lrt3 = []
    urt3 = []

    for util in utils:
        r1 = func(util, 0.5, 0.5)[1]
        r2 = func(0.5, util, 0.5)[2]
        r3 = func(0.5, 0.5, util)[3]

        mrt1.append(r1[0])
        mrt2.append(r2[0])
        mrt3.append(r3[0])

        ex1.append(r1[1])
        ex2.append(r2[1])
        ex3.append(r3[1])

        lrt1.append(r1[2][0])
        lrt2.append(r2[2][0])
        lrt3.append(r3[2][0])

        urt1.append(r1[2][1])
        urt2.append(r2[2][1])
        urt3.append(r3[2][1])

    ax1.fill_between(utils, lrt1, urt1)
    ax1.plot(utils, mrt1, c='r', label='observed')
    ax1.plot(utils, ex1, '--', c='g', label='expected')
    ax1.legend()

    ax2.fill_between(utils, lrt2, urt2)
    ax2.plot(utils, mrt2, c='r', label='observed')
    ax2.plot(utils, ex2, '--', c='g', label='expected')
    ax2.legend()

    ax3.fill_between(utils, lrt3, urt3)
    ax3.plot(utils, mrt3, c='r', label='observed')
    ax3.plot(utils, ex3, '--', c='g', label='expected')
    ax3.legend()

    plt.show()
