from collections import deque
from src.mm1 import MM1
from src.core import Simulator

import numpy as np
import scipy.stats as st


def func(util, total_time):

    arrival = 1
    service = arrival / util

    queue = MM1(adjacent_states=lambda s: [1] if s == 0 else [s - 1, s + 1],
                arrival_rate=arrival, service_rate=service)

    warmup_time = 0

    mean_job_count = util / (1 - util)
    mean_response_length = 1 / (service - arrival)
    mean_waiting_time = util / (service - arrival)
    mean_queue_length = util ** 2 / (1 - util)

    simulation = Simulator(queue)

    class mm1plugin:

        _job_counts = []
        _response_times = []
        _waiting_times = []
        _queue_lengths = []
        _arrival_time = deque()
        _service_start_time = 0

        def stream(self, _data):
            old_state = _data['old state']
            new_state = _data['new state']
            event = _data['event']
            time = _data['time']

            if event == 'arr':
                self._arrival_time.append(time)
                if old_state == 0:
                    self._service_start_time = time
            elif event == 'com':
                arr_time = self._arrival_time.popleft()
                self._response_times.append(time - arr_time)

                service_time = time - self._service_start_time
                self._waiting_times.append(time - service_time - arr_time)

                if new_state > 0:
                    self._service_start_time = time

                self._job_counts.append(new_state)

                self._queue_lengths.append(
                    max(0, new_state - 1)   # One customer is in service
                )

        def get(self):
            return (
                self._job_counts,
                self._response_times,
                self._waiting_times,
                self._queue_lengths
            )

    x = mm1plugin()
    simulation.run(until=total_time, warmup_until=warmup_time, plugin=x.stream)

    def format_data(_data, acc):
        mean = np.mean(_data)
        return (
            mean,
            acc,
            st.norm.interval(alpha=0.90, loc=mean, scale=st.sem(_data))
        )

    return list(map(format_data, x.get(), [mean_job_count, mean_response_length, mean_waiting_time, mean_queue_length]))


if __name__ == "__main__":
    utils = np.linspace(0.1, 0.9, num=9)
    time = 10000

    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots()
    ax1.set_title('mean job count')
    ax1.set_xlabel('utilisation - ' + r'$\rho$')
    ax1.set_ylabel('job count')
    mean_job_data = []
    expected_job_data = []
    lower_job_data = []
    upper_job_data = []

    fig2, ax2 = plt.subplots()
    ax2.set_title('mean response time')
    ax2.set_xlabel('utilisation - ' + r'$\rho$')
    ax2.set_ylabel('response time')
    mean_response_data = []
    expected_response_data = []
    lower_response_data = []
    upper_response_data = []

    fig3, ax3 = plt.subplots()
    ax3.set_title('mean waiting time')
    ax3.set_xlabel('utilisation - ' + r'$\rho$')
    ax3.set_ylabel('waiting time')
    mean_waiting_data = []
    expected_waiting_data = []
    lower_waiting_data = []
    upper_waiting_data = []

    fig4, ax4 = plt.subplots()
    ax4.set_title('mean queueing time')
    ax4.set_xlabel('utilisation - ' + r'$\rho$')
    ax4.set_ylabel('queue length')
    mean_queue_data = []
    expected_queue_data = []
    lower_queue_data = []
    upper_queue_data = []

    for util in utils:
        vals = func(util, time)

        job_data = vals[0]
        mean_job_data.append(job_data[0])
        expected_job_data.append(job_data[1])
        lower_job_data.append(job_data[2][0])
        upper_job_data.append(job_data[2][1])

        response_data = vals[1]
        mean_response_data.append(response_data[0])
        expected_response_data.append(response_data[1])
        lower_response_data.append(response_data[2][0])
        upper_response_data.append(response_data[2][1])

        waiting_data = vals[2]
        mean_waiting_data.append(waiting_data[0])
        expected_waiting_data.append(waiting_data[1])
        lower_waiting_data.append(waiting_data[2][0])
        upper_waiting_data.append(waiting_data[2][1])

        queue_data = vals[3]
        mean_queue_data.append(queue_data[0])
        expected_queue_data.append(queue_data[1])
        lower_queue_data.append(queue_data[2][0])
        upper_queue_data.append(queue_data[2][1])

    ax1.fill_between(utils, lower_job_data, upper_job_data)
    ax1.plot(utils, mean_job_data, c='g', label='actual')
    ax1.plot(utils, expected_job_data, '--', c='r', label='expected')
    ax1.legend()

    ax2.fill_between(utils, lower_response_data, upper_response_data)
    ax2.plot(utils, mean_response_data, c='g', label='actual')
    ax2.plot(utils, expected_response_data, '--', c='r', label='expected')
    ax2.legend()

    ax3.fill_between(utils, lower_waiting_data, upper_waiting_data)
    ax3.plot(utils, mean_waiting_data, c='g', label='actual')
    ax3.plot(utils, expected_waiting_data, '--', c='r', label='expected')
    ax3.legend()

    ax4.fill_between(utils, lower_queue_data, upper_queue_data)
    ax4.plot(utils, mean_queue_data, c='g', label='actual')
    ax4.plot(utils, expected_queue_data, '--', c='r', label='expected')
    ax4.legend()

    fig1.savefig("mm1 mean job count time={}.png".format(time))
    fig2.savefig("mm1 mean response time time={}.png".format(time))
    fig3.savefig("mm1 mean waiting time time={}.png".format(time))
    fig4.savefig("mm1 mean queue length time={}.png".format(time))
    plt.show()

