from src.gsmp import *
import numpy as np

N = 20

es = {'arr': Event('arr'), 'com': Event('com')}
ss = [State(str(i)) for i in range(N + 1)]


class Mg1(GsmpSpec):
    def __init__(self, states, events, distribution, arrival_rate, service_rate):
        super().__init__(states, events)
        self.distributions = distribution
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate

    def e(self, s: State) -> list:
        if s == ss[0]:
            return [Event('arr')]
        elif s == ss[-1]:
            return [Event('com')]
        elif s in ss:
            return [Event('arr'), Event('com')]
        else:
            raise TypeError

    def p(self, _s: State, s: State, e: Event) -> float:
        if ss.index(_s) == ss.index(s) + 1:
            return 1 if e == es['arr'] else 0
        elif ss.index(_s) == ss.index(s) - 1:
            return 1 if e == es['com'] else 0
        else:
            return 0

    def f(self, _s: State, _e: Event, s: State, e: Event) -> float:
        if e == es['arr']:
            if ss.index(_s) == ss.index(s) + 1:
                if ss.index(s) in range(0, N) and _e == es['arr']:
                    return self.distributions['arr'](*self.arrival_rate)
                if ss.index(s) == 0 and _e == es['com']:
                    return self.distributions['com'](*self.service_rate)
        elif e == es['com']:
            if ss.index(_s) == ss.index(s) - 1:
                if ss.index(s) in range(2, N + 1) and _e == es['com']:
                    return self.distributions['com'](*self.service_rate)
                if ss.index(s) == N and _e == es['arr']:
                    return self.distributions['arr'](*self.arrival_rate)
        raise ValueError

    def r(self, s: State, e: Event) -> float:
        return 1 if e in self.events else 0

    def s_0(self, s: State) -> float:
        return 1 if s == State('0') else 0

    def f_0(self, e: Event, s: State) -> float:
        if ss.index(s) == 0 and e == es['arr']:
            return self.distributions['arr'](*self.arrival_rate)
        else:
            raise ValueError


if __name__ == "__main__":
    spec = Mg1(ss, list(es.values()), {'arr': np.random.exponential, 'com': np.random.exponential}, [1], [0.5])
    simulation = GsmpSimulation(spec)
    total_time = simulation.simulate(1000)
    from functools import reduce
    print(reduce(lambda x, y: x + y, [int(_s.label) * _s.time_spent for _s in ss]) / total_time)
