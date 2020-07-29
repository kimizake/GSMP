from src.gsmp import *
import numpy as np


class ggc(GsmpSpec):
    def __init__(self, states, events, c, k, dist, service_rates):
        self.c = c
        self.k = k
        self.dist = dist
        self.service_rates = service_rates
        super().__init__(states, events)

    def e(self, s: State) -> list:
        events = list()
        if s.label < self.k:
            events.append(Event('arr'))
        if s.label > 0:
            events = events + [Event(('com', i + 1)) for i in range(self.c)]
        return events

    def p(self, _s: State, s: State, e: Event) -> float:
        if e == Event('arr'):
            return 1 if _s.label == s.label + 1 and s.label in range(self.k) else 0
        else:
            (_, server) = e.label
            return 1 if _s.label == s.label - 1 and _s.label in range(self.k) else 0

    def f(self, _s: State, _e: Event, s: State, e: Event) -> float:
        if _e == e == Event('arr'):
            if _s.label == s.label + 1 and s.label in range(self.k):
                return self.dist['arr'](self.service_rates['arr'])
            else:
                raise TypeError
        if _e == e:
            if _s.label == s.label - 1 and _s.label in range(self.k):
                return self.dist['com'][e.label[1]](self.service_rates['com'][e.label[1]])
            else:
                raise TypeError
        if _e == Event('arr'):
            if s.label == self.k and _s.label == self.k - 1 and e.label[0] == 'com':
                return self.dist['arr'](self.service_rates['arr'])
            else:
                raise TypeError
        if e == Event('arr'):
            if s.label == 0 and _s.label == 1 and _e.label[0] == 'com':
                return self.dist['com'][_e.label[1]](self.service_rates['com'][_e.label[1]])
            else:
                raise TypeError
        raise TypeError

    def r(self, s: State, e: Event) -> float:
        return 1

    def s_0(self, s: State) -> float:
        return 1 if s == State(0) else 0

    def f_0(self, e: Event, s: State) -> float:
        if e == Event('arr') and s == State(0):
            return np.random.exponential(1)
        else:
            raise TypeError


if __name__ == "__main__":
    servers = 3
    queue_length = 5
    ss = [State(i) for i in range(queue_length + 1)]
    es = [Event('arr')] + [Event(('com', i + 1)) for i in range(servers)]
    mmc = ggc(
        ss, es, servers, queue_length,
        {
            'arr': np.random.exponential,
            'com': {
                1: np.random.exponential,
                2: np.random.exponential,
                3: np.random.exponential,
            }
        },
        {
            'arr': 1,
            'com': {
                1: 1 / 2,
                2: 1 / 2,
                3: 1 / 2,
            }
        }
    )
    sim = GsmpSimulation(mmc)
    ps = sim.simulate(1000)
    from operator import itemgetter

    for x in sorted(zip(ss, ps), key=itemgetter(1), reverse=True):
        print(x)
