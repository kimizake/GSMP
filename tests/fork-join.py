from examples.mm1 import MM1
from core import Simulator
from functools import reduce
from operator import add, and_
from collections import deque

K = 2               # Number of queues
simtime = 10000     # runs

util = 0.3
arrival = 1
service = arrival / util

qs = [MM1(adjacent_states=lambda state: [1] if state == 0 else [state - 1, state + 1],
          arrival_rate=arrival, service_rate=service) for _ in range(K)]


class FJ_Plugin:
    # Queue data structures
    _qarrival_times = {q: deque() for q in qs}
    _qresponse_times = {q: deque() for q in qs}
    # Metrics
    _jobs_seen = 0
    _mean_response_time = 0
    _response_times = []
    # Graphing
    _output = []

    def incoming_event(self, data):
        event = data['event']
        process = data['process']
        time = data['time']
        if event == 'com':
            self._qresponse_times[process].append(
                time - self._qarrival_times[process].popleft()
            )
        elif event == 'arr':
            for q in qs:
                self._qarrival_times[q].append(time)
        else:
            raise TypeError('unexpected event')

        def _job_complete():
            # Return True if none of the response time stacks of each queue are empty
            return reduce(and_, map(lambda _queue: len(_queue) > 0, self._qresponse_times.values()))

        if _job_complete():
            self._mean_response_time *= self._jobs_seen

            self._jobs_seen += 1
            response_time = max(map(
                lambda _queue: _queue.popleft(),
                self._qresponse_times.values()
            ))

            self._mean_response_time += response_time
            self._mean_response_time /= self._jobs_seen

            self._response_times.append(response_time)

            self._output.append((self._mean_response_time, time))

    def get_graph_data(self):
        return self._output

    def get_response_times(self):
        return self._response_times


if __name__ == "__main__":
    fj = reduce(add, qs)
    fj.shared_events = [[('arr', q) for q in qs]]   # Synchronise on arrivals
    plugin = FJ_Plugin()
    Simulator(fj).run(until=simtime, plugin=plugin.incoming_event)    # Aren't interested in default output

    H_2 = 1 + 0.5
    H_k = sum(1 / h for h in range(1, K + 1))
    upper_bound = H_k / (service - arrival)
    lower_bound = H_k / service
    approx = (H_k/H_2 + 4 / 11 * (1 - H_k / H_2) * util) * (12 - util) / 8 / (service - arrival)

    # make graphs
    from matplotlib import pyplot as plt
    mean_response_times, timeline = zip(*plugin.get_graph_data())
    plt.plot(timeline, mean_response_times)
    plt.plot(timeline, [upper_bound]*len(timeline), '--', label='upper')
    plt.plot(timeline, [lower_bound]*len(timeline), '--', label='lower')
    plt.plot(timeline, [approx]*len(timeline), '--', label='approximation')

    plt.ylabel('mean job response time')
    plt.xlabel('time')
    plt.title('Response time of Fork-join system with {} M/M/1 server queues'.format(K) + r' $\rho={}$'.format(util))
    # plt.savefig('./fork join results/{0}queues_{1}.png'.format(K, util))

    plt.show()

    # save data
    responses = plugin.get_response_times()
    import numpy as np
    import scipy.stats as st
    import pandas as pd

    mean = np.mean(responses)
    ci = st.norm.interval(alpha=0.99, loc=mean, scale=st.sem(responses))
    acc = 100 * (ci[1] - mean) / mean
    df = pd.DataFrame(data={
        r'$\rho$': [util],
        'approximation': [approx],
        'estimate': [mean],
        '{}% confidence interval'.format(100 * 0.99): [ci],
        'accuracy (%)': [acc],
    })
    # df.to_csv('fork join results/{0}queue_{1}.csv'.format(K, util))

