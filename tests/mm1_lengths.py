from examples.mm1 import MM1
from src.core import Simulator

import numpy as np
import scipy.stats as st


alpha = 0.95
arrival = 1
batches = 100
accuracy = 0.1


def func(util, total_time, warmup_time):

    service = arrival / util

    queue = MM1(adjacent_states=lambda s: [1] if s == 0 else [s - 1, s + 1],
                arrival_rate=arrival, service_rate=service)

    mean_job_count = util / (1 - util)
    mean_queue_length = util ** 2 / (1 - util)

    simulation = Simulator(queue)

    interval_size, _ = divmod(total_time, batches)

    class mm1plugin:

        _job_counts = []
        _queue_lengths = []

        _batch = 0
        _next_sample_time = np.random.rand() * interval_size

        def stream(self, _data):
            state = _data['old state']
            t = _data['time']

            if t < self._next_sample_time:
                return

            self._job_counts.append(state)
            self._queue_lengths.append(max(0, state - 1))
            self._batch += 1
            self._next_sample_time = self._batch * interval_size + np.random.rand() * interval_size

        def get(self):
            return (
                self._job_counts,
                self._queue_lengths
            )

    x = mm1plugin()
    simulation.run(until=total_time, warmup_until=warmup_time, plugin=x.stream)

    def format_data(_data, acc):
        mean = np.mean(_data)
        ci = st.norm.interval(alpha=alpha, loc=mean, scale=st.sem(_data))
        h = ci[1] - mean
        return (
            mean,
            acc,
            (mean - h, mean + h),
            h,
            mean * accuracy,
        )

    return list(map(format_data, x.get(), [mean_job_count, mean_queue_length]))


if __name__ == "__main__":
    utils = np.linspace(0.1, 0.9, num=9)
    time = 10000

    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots()
    ax1.set_title('M/M/1 mean population with {} batches'.format(batches))
    ax1.set_xlabel('utilisation - ' + r'$\rho$')
    ax1.set_ylabel('Population size')
    mean_job_data = []
    expected_job_data = []
    lower_job_data = []
    upper_job_data = []

    fig4, ax4 = plt.subplots()
    ax4.set_title('M/M/1 mean queue length with {} batches'.format(batches))
    ax4.set_xlabel(r'$\rho$')
    ax4.set_ylabel('Queue length')
    mean_queue_data = []
    expected_queue_data = []
    lower_queue_data = []
    upper_queue_data = []

    for util in utils:
        vals = func(util, time, 0)

        job_data = vals[0]
        mean_job_data.append(job_data[0])
        expected_job_data.append(job_data[1])
        lower_job_data.append(job_data[2][0])
        upper_job_data.append(job_data[2][1])

        queue_data = vals[1]
        mean_queue_data.append(queue_data[0])
        expected_queue_data.append(queue_data[1])
        lower_queue_data.append(queue_data[2][0])
        upper_queue_data.append(queue_data[2][1])

    ax1.fill_between(utils, lower_job_data, upper_job_data, color='b', alpha=.5, label='actual')
    ax1.plot(utils, expected_job_data, 'r--', label='steady state')
    ax1.legend()

    ax4.fill_between(utils, lower_queue_data, upper_queue_data, color='b', alpha=.5, label='actual')
    ax4.plot(utils, expected_queue_data, 'r--', label='steady state')
    ax4.legend()

    fig1.savefig("mm1 results/mm1 mean job batches {}.png".format(batches))
    fig4.savefig("mm1 results/mm1 mean queue length batches {}.png".format(batches))
    plt.show()

