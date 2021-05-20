from gsmp import Gsmp

arrival = 1
service = 2


class MM1(Gsmp):

    def states(self):
        s = 0
        while True:
            yield s
            s += 1

    def events(self):
        return ['arr', 'com']

    def e(self, state):
        if state == 0:
            return ['arr']
        return self.events()

    def p(self, next_state, current_state, event):
        if event == 'arr':
            return int(next_state == current_state + 1)
        else:
            return int(next_state == current_state - 1)

    def f(self, next_state, new_event, current_state, winning_event):
        from numpy.random import exponential
        if new_event == 'arr':
            return exponential(1 / arrival)
        else:
            return exponential(1 / service)

    def r(self, state, event):
        return 1

    def s_0(self, state):
        return int(state == 0)

    def f_0(self, event, state):
        from numpy.random import exponential
        return exponential(1 / arrival)


rho = arrival / service

queue = MM1(
    infinite=True,
    adjacent_states=lambda state: [1] if state == 0 else [state - 1, state + 1]
)

if __name__ == "__main__":
    from gsmp import Simulator
    epochs = 10000

    states, holding_times, total_time = Simulator(queue).run(epochs)    # generate results

    expected_probabilities = list(map(       # Expected M/M/1 probabilities
        lambda n: (1 - rho) * rho ** n,
        states
    ))

    observed_probabilities = holding_times / total_time     # Estimate probabilities based on holding times

    # Make some graphs
    from matplotlib import pyplot as plt
    plt.plot(states, expected_probabilities, label='expected probabilities')
    plt.plot(states, observed_probabilities, label='observed probabilities')
    plt.xlabel('state')
    plt.ylabel('probability')
    plt.title('M/M/1 with traversals = {}'.format(epochs))
    plt.legend()
    plt.show()
