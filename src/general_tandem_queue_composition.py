import numpy as np

from itertools import repeat
from functools import reduce
from operator import add
from gsmp import Simulator
from mm1k import MM1k

k = 10              # Queue Buffer size
arrival_rate = 1    # Network arrival rate
service_rate = 2    # Service rate of each node
num = 4             # Number of queues in network
epochs = 10000      # Number of paths

if __name__ == "__main__":
    # Define M/M/1/K queues
    queues = [MM1k('Queue {}'.format(i),
                   _k=k,
                   _arrival_rate=arrival_rate if i == 1 else service_rate,
                   _service_rate=service_rate) for i in range(1, num + 1)]
    # Compose them
    network = reduce(add, queues)
    network.shared_events = [[('com', queues[i]), ('arr', queues[i + 1])] for i in range(num - 1)]
    # Generate path
    states, holding_times, total_time = Simulator(network).run(epochs)
    # Project results onto grid data structure
    grid = np.zeros(shape=tuple(repeat(k + 1, times=num)))
    for _state, _holding_time in zip(states, holding_times):
        grid[_state] = _holding_time
    # Use grid to evaluate individual queue probabilities
    probabilities = [np.sum(grid, axis=tuple(filter(lambda x: x != i, range(num)))) / total_time for i in range(num)]
    # Plot these probabilities against the steady state results
    from utility import mmc_p, print_results
    expected_probability = mmc_p(arrival_rate / service_rate, 1, k)
    print_results(p=expected_probability, ys=[
        (p, 'Queue {0} with epochs {1}'.format(i + 1, epochs)) for i, p in enumerate(probabilities)
    ])
