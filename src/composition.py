from gsmp import Simulator
import numpy as np

# Changes to these parameters must be made in the tandem_queue and mm1k files
k = 20
arrival_rate = 1
service_rate = 2
epochs = 10000
warmup = 0


def get_state_probabilities(_states, _holding_times, _total_time):
    # Map the holding times onto a plane
    grid = np.zeros(shape=(k + 1, k + 1))
    for _state, _holding_time in zip(_states, _holding_times):
        grid[_state] = _holding_time

    first_queue_probabilities = sum(grid[i, :] for i in range(k + 1)) / _total_time     # Sum the rows
    second_queue_probabilities = sum(grid[:, i] for i in range(k + 1)) / _total_time    # Sum the columns

    return first_queue_probabilities, second_queue_probabilities


if __name__ == "__main__":
    # Define a simple tandem queue with 2 mm1k queues

    from mm1k import MM1k
    # Define 2 mm1k queues
    q1, q2 = MM1k('Queue 1'), MM1k('Queue 2')
    # Compose tandem queue
    tq1 = q1 + q2
    tq1.shared_events = [
        [('com', q1), ('arr', q2)]
    ]

    # Define tandem queue as normal gsmp
    from tandem_queue import Tandem_queue
    tq2 = Tandem_queue()

    # Run the simulations
    cmp_states, cmp_holding_times, cmp_total_time = Simulator(tq1).run(epochs, warmup=warmup)
    reg_states, reg_holding_times, reg_total_time = Simulator(tq2).run(epochs, warmup=warmup)

    # Obtain the estimated pdf for the two queues
    y1, y2 = get_state_probabilities(cmp_states, cmp_holding_times, cmp_total_time)     # results from composition
    y3, y4 = get_state_probabilities(reg_states, reg_holding_times, reg_total_time)     # results from regular approach

    # Plot the results
    from utility import mmc_p, print_results
    expected_probabilities = mmc_p(arrival_rate / service_rate, 1, k)
    print_results(p=expected_probabilities, ys=[
        (y1, 'Composed queue 1 epochs = {}'.format(epochs)),
        (y2, 'Composed queue 2 epochs = {}'.format(epochs)),
        (y3, 'Normal queue 3 epochs = {}'.format(epochs)),
        (y4, 'Normal queue 4 epochs = {}'.format(epochs)),
    ])

