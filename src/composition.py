from gsmp import Compose, Simulator
import matplotlib.pyplot as plt
import numpy as np

# Changes to these parameters must be made in the tandem_queue and mm1k files
k = 20
arrival_rate = 1
service_rate = 2
epochs = 10000
warmup = 0


def mmc_p(p0, n, c, u):
    # calculate probability that M/M/c/K queue has n customers given utilisation rho
    import math
    if n <= c:
        return p0 * u ** n / math.factorial(n)
    else:
        return p0 * u ** n / math.factorial(c) / c ** (n - c)


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
    # Define tandem queue as composition
    tq1 = Compose(q1, q2, synchros=[
        [('com', q1), ('arr', q2)],
    ])

    # Define tandem queue as normal gsmp
    from tandem_queue import Tandem_queue
    tq2 = Tandem_queue()

    # Run the simulations
    cmp_states, cmp_holding_times, cmp_total_time = Simulator(tq1).run(epochs, warmup=warmup)
    reg_states, reg_holding_times, reg_total_time = Simulator(tq2).run(epochs, warmup=warmup)

    # Obtain the estimated pdf for the two queues
    y1, y2 = get_state_probabilities(cmp_states, cmp_holding_times, cmp_total_time)     # results from composition
    y3, y4 = get_state_probabilities(reg_states, reg_holding_times, reg_total_time)     # results from regular approach

    # Code to generate the expected state probabilities
    import math
    rho = arrival_rate / service_rate
    c = 1
    p0 = 1 / math.fsum(
        [rho ** i / math.factorial(i) for i in range(c)] + [rho ** c / math.factorial(c - 1) / (c - rho)])
    # This is just the mm1k formula
    ps = [p0] + [mmc_p(p0, n, c, rho) for n in range(1, k+1)]

    # Create the graphs
    def print_plots(y, title):
        fig, ax = plt.subplots()
        ax.plot(range(k + 1), y, label='actual')
        ax.plot(range(k + 1), ps, label='expected')
        ax.set_xlabel('Number of jobs in system')
        ax.set_ylabel('Probability')
        ax.set_title(title)
        ax.legend()

    # plot results
    print_plots(y1, 'Queue 1 composition')
    print_plots(y2, 'Queue 2 composition')
    print_plots(y3, 'Queue 1 normal')
    print_plots(y4, 'Queue 2 normal')
    plt.show()
