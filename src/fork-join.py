from mm1 import MM1
from gsmp import Simulator
from functools import reduce
from operator import add, and_
from collections import deque

K = 5           # Number of queues
epochs = 10000  # runs

qs = [MM1(adjacent_states=lambda state: [1] if state == 0 else [state - 1, state + 1]) for _ in range(K)]


class FJ_Plugin:
    # Queue data structures
    _arrival_times = {q: deque() for q in qs}
    _response_times = {q: deque() for q in qs}
    # Metrics
    _jobs_seen = 0
    _mean_response_time = 0
    # Graphing
    _output = []

    def incoming_event(self, event, process, time):
        if event == 'com':
            self._response_times[process].append(
                time - self._arrival_times[process].popleft()
            )
        elif event == 'arr':
            self._arrival_times[process].append(time)
        else:
            raise TypeError('unexpected event')

        def _job_complete():
            # Return True if none of the response time stacks of each queue are empty
            return reduce(and_, map(lambda _queue: len(_queue) > 0, self._response_times.values()))

        if _job_complete():
            self._mean_response_time *= self._jobs_seen

            self._jobs_seen += 1
            response_time = max(map(
                lambda _queue: _queue.popleft(),
                self._response_times.values()
            ))

            self._mean_response_time += response_time
            self._mean_response_time /= self._jobs_seen

            self._output.append((self._mean_response_time, time))

    def get_results(self):
        return self._output


if __name__ == "__main__":
    fj = reduce(add, qs)
    plugin = FJ_Plugin()
    Simulator(fj).run(epochs=epochs, plugin=plugin.incoming_event)    # Aren't interested in default output
    mean_response_times, timeline = zip(*plugin.get_results())
    upper_bound = sum(1/i for i in range(1, K + 1))

    from matplotlib import pyplot as plt
    plt.plot(timeline, mean_response_times)
    plt.plot(timeline, [upper_bound]*len(timeline), '--')
    plt.ylabel('average job response time')
    plt.xlabel('time')
    plt.title('Response time of Fork-join system with {} M/M/1 server queues'.format(K))
    plt.show()
