def mmc_p(utilisation, servers, buffer):
    """
    Return analytic probability distribution for M/M/c/K
    :param utilisation: Queue utilisation
    :param servers: Number of servers - c
    :param buffer: Number of values to generate
    :return: List of steady state probabilities
    """
    import math
    # π(0)
    p0 = 1 / math.fsum(
        [utilisation ** i / math.factorial(i) for i in range(servers)] +
        [utilisation ** servers / math.factorial(servers - 1) / (servers - utilisation)])

    def _mmc_p(n):
        # π(n)
        import math
        if n <= servers:
            return p0 * utilisation ** n / math.factorial(n)
        else:
            return p0 * utilisation ** n / math.factorial(servers) / servers ** (n - servers)

    return [p0] + [_mmc_p(n) for n in range(1, buffer + 1)]


def print_results(p=None, ys=None):
    from matplotlib import pyplot as plt

    k = len(p)

    # Create the graphs
    def graph(_y, _title):
        fig, ax = plt.subplots()
        ax.plot(range(k), p, '--', label='expected')
        ax.plot(range(k), _y, label='actual')
        ax.set_xlabel('Number of jobs in system')
        ax.set_ylabel('Probability')
        ax.set_title(_title)
        ax.legend()

    for data, title in ys:
        graph(data, title)

    plt.show()
