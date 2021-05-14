from gsmp import Compose, Simulator
import matplotlib.pyplot as plt
import numpy as np

# Changes to these parameters must be made in the tandem_queue and mm1k files
k = 20
arrival_rate = 1
service_rate = 2
epochs = 20000


def mmc_p(p0, n, c, rho):
    """
    Calculate mmc
    """
    import math
    if n < c:
        return p0 * rho ** n / math.factorial(n)
    else:
        return p0 * rho ** n / math.factorial(c) / c ** (n - c)


if __name__ == "__main__":
    # Define a simple tandem queue with 2 mm1k queues

    from mm1k import MM1k
    # Define 2 mm1k queues
    q1, q2 = MM1k('Queue 1'), MM1k('Queue 2')
    # Define tandem queue as composition
    tq1 = Compose(q1, q2, synchros=[
        [('com', q1), ('arr', q2)],
    ])

    from tandem_queue import Tandem_queue
    # Define tandem queue as normal gsmp
    tq2 = Tandem_queue()

    # Run the simulations
    sim = Simulator(tq1)
    sim.simulate(epochs)

    sim2 = Simulator(tq2)
    sim2.simulate(epochs)

    # Code to generate the expected state probabilities
    import math
    rho = arrival_rate / service_rate
    c = 1
    p0 = 1 / math.fsum(
        [rho ** i / math.factorial(i) for i in range(c)] + [rho ** c / math.factorial(c - 1) / (c - rho)])

    # This is just the mm1k formula
    ps = [p0] + [mmc_p(p0, n, 1, rho) for n in range(1, k+1)]

    # Code to g
    def print_plots(y, title):
        fig, ax = plt.subplots()
        ax.plot(range(k + 1), y, label='actual')
        xrange = np.linspace(0, k, k + 1)
        ax.plot(xrange, ps, label='expected')
        ax.set_xlabel('Number of jobs in system')
        ax.set_ylabel('Probability')
        ax.set_title(title)
        ax.legend()

    # generate plots from composed queues
    def pull_names(ss):
        return list(map(lambda s: s.name, ss))

    def pull_times(ss):
        return map(lambda s: s.time_spent, ss)

    def pull_probs(ts, time):
        return list(map(lambda t: t / time, ts))

    x = pull_names(q1.get_states())
    y1 = pull_probs(pull_times(q1.get_states()), sim.total_time)
    y2 = pull_probs(pull_times(q2.get_states()), sim.total_time)
    print_plots(y1, 'Queue 1 composition')
    print_plots(y2, 'Queue 2 composition')

    # format results from tq2 test
    states = np.array(tq2.get_states()).reshape((k + 1, k + 1))     # convert flat list to 2d array
    # sum the time values across the rows and columns of the states 2d array to obtain the cumulative time spent
    # for each queue in each state
    t1 = [sum(map(lambda s: s.time_spent, states[i, :])) for i in range(k + 1)]
    t2 = [sum(map(lambda s: s.time_spent, states[:, i])) for i in range(k + 1)]
    del states

    y3 = pull_probs(t1, sim2.total_time)
    y4 = pull_probs(t2, sim2.total_time)

    print_plots(y3, 'Queue 1 normal')
    print_plots(y4, 'Queue 2 normal')

    plt.show()

