from src.gsmp import *
import numpy as np

N = 10

arrival_rate = 1   # arrival rate
service_rate = 2   # service rate

Events = {'arr': Event('arr'), 'com': Event('com')}
States = [State(str(i)) for i in range(N + 1)]


class mg1(GsmpSpec):
    def __init__(self, states, events):
        super().__init__(states, events)
        self.distributions = {
            'arr': np.random.exponential,
            'com': np.random.exponential,
        }

    def e(self, s: State) -> list:
        if s == States[0]:
            return [Event('arr')]
        elif s == States[-1]:
            return [Event('com')]
        elif s in States:
            return [Event('arr'), Event('com')]
        else:
            raise TypeError

    def p(self, _s: State, s: State, e: Event) -> float:
        if States.index(_s) == States.index(s) + 1:
            return 1 if e == Events['arr'] else 0
        elif States.index(_s) == States.index(s) - 1:
            return 1 if e == Events['com'] else 0
        else:
            raise TypeError

    def f(self, _s: State, _e: Event, s: State, e: Event, *args) -> float:
        if e == Events['arr']:
            if States.index(_s) == States.index(s) + 1:
                if States.index(s) in range(0, N) and _e == Events['arr']:
                    return self.distributions['arr'](1 / arrival_rate)
                if States.index(s) == 0 and _e == Events['com']:
                    return self.distributions['com'](1 / service_rate)
        elif e == Events['com']:
            if States.index(_s) == States.index(s) - 1:
                if States.index(s) in range(2, N + 1) and _e == Events['com']:
                    return self.distributions['com'](1 / service_rate)
                if States.index(s) == N and _e == Events['arr']:
                    return self.distributions['arr'](1 / arrival_rate)
        raise ValueError

    def r(self, s: State, e: Event) -> float:
        return 1 if e in self.events else 0

    def s_0(self, s: State) -> float:
        return 1 if s == State('0') else 0

    def f_0(self, e: Event, s: State, *args) -> float:
        if States.index(s) == 0 and e == Events['arr']:
            return self.distributions['arr'](1)
        else:
            raise ValueError


if __name__ == "__main__":
    spec = mg1(States, list(Events.values()))
    simulation = GsmpSimulation(spec)
    total_time = simulation.simulate(1000)
    from functools import reduce
    print(reduce(lambda x, y: x + y, [int(_s.label) * _s.time_spent for _s in States]) / total_time)
