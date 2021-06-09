from core import Gsmp
from numpy.random import exponential

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

    def p(self, next_state, event, current_state):
        if event == 'arr':
            return int(next_state == current_state + 1)
        else:
            return int(next_state == current_state - 1)

    def f(self, next_state, new_event, current_state, winning_event):
        if new_event == 'arr':
            return exponential(1 / self._arrival)
        else:
            return exponential(1 / self._service)

    def r(self, state, event):
        return 1

    def s_0(self, state):
        return int(state == 0)

    def f_0(self, state, event):
        return exponential(1 / self._arrival)

    def __init__(self, adjacent_states=None, arrival_rate=arrival, service_rate=service):
        self._arrival = arrival_rate
        self._service = service_rate
        if adjacent_states is None:
            def adjacent_states(s):
                return [1] if s == 0 else [s - 1, s + 1]
        super().__init__(adjacent_states=adjacent_states)


rho = arrival / service

queue = MM1(
    adjacent_states=lambda state: [1] if state == 0 else [state - 1, state + 1]
)

if __name__ == "__main__":
    from core import Simulator
    epochs = 1000

    data = Simulator(queue).run(epochs=epochs, estimate_probability=True)  # generate results

    states, observed_probabilities = zip(*data)     # unpack data

    expected_probabilities = list(map(       # Expected M/M/1 probabilities
        lambda n: (1 - rho) * rho ** n,
        states
    ))

    from utility import print_results
    print_results(p=expected_probabilities, ys=[(observed_probabilities, 'M/M/1')])
