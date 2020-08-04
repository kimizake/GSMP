from src.gsmp import *
import numpy as np
from itertools import chain, groupby, product, repeat
from numpy.linalg import norm
from operator import itemgetter


class ggc(GsmpSpec):
    def __init__(self, states, events, c, k, dist, arrival_rate, service_rate):
        self.c = c
        self.k = k
        self.dist = dist
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        super().__init__(states, events)

    def e(self, s: State) -> list:
        events = list()
        (queue_length, *servers) = s.label
        if queue_length < self.k:
            events.append(Event('arr'))
        for server, status in enumerate(servers):
            if status == 1:
                events.append(Event(('com', server)))
        return events

    def p(self, _s: State, s: State, e: Event) -> float:
        (old_queue, *old_servers) = s.label
        (new_queue, *new_servers) = _s.label
        if e == Event('arr'):
            if old_queue < self.k and new_queue == old_queue + 1:
                return int(np.all(old_servers == new_servers == np.ones(self.c)))
            elif old_queue == new_queue == 0 and norm(old_servers, ord=1) + 1 == norm(new_servers, ord=1) and len(
                    np.where(np.not_equal(old_servers, new_servers) == 1)[0]) == 1:
                return 1 / (self.c - norm(old_servers, ord=1))
        else:
            (_, server) = e.label
            if old_queue - 1 == new_queue and old_queue > 0:
                return int(np.all(old_servers == new_servers == np.ones(self.c)))
            elif old_queue == new_queue == 0:
                return int(old_servers[server] == 1 and new_servers[server] == 0 and np.all(
                    np.where(np.not_equal(old_servers, new_servers)) == np.array(server)
                ))
        return 0

    def f(self, _s: State, _e: Event, s: State, e: Event) -> float:
        if _e == Event('arr'):
            return self.dist['arr'](self.arrival_rate)
        else:
            return self.dist['com'](self.service_rate)

    def r(self, s: State, e: Event) -> float:
        return 1

    def s_0(self, s: State) -> float:
        return int(s == State((0, *repeat(0, self.c))))

    def f_0(self, e: Event, s: State) -> float:
        if e == Event('arr') and s == State((0, *repeat(0, self.c))):
            return np.random.exponential(1)
        else:
            raise TypeError


def pi(c, k, l, m):
    u = l / m
    p0 = 1 / (np.sum([u ** i / np.math.factorial(i) for i in range(c + 1)])
              + u ** c / np.math.factorial(c) *
              np.sum([u ** i / c ** i for i in range(1, k - c + 1)]))
    yield p0
    for i in range(1, c + 1):
        yield p0 * u ** i / np.math.factorial(i)
    for i in range(c + 1, k + 1):
        yield p0 * u ** i / c ** (i - c) / np.math.factorial(c)


if __name__ == "__main__":
    c = 1
    k = 3
    ss = list(chain(
        (State((0, *t)) for t in product(range(2), repeat=c)),
        (State((i + 1, *repeat(1, c))) for i in range(k))
    ))
    es = [Event('arr')] + [Event(('com', i)) for i in range(c)]
    mmc = ggc(
        ss, es, c, k,
        {
            'arr': np.random.exponential,
            'com': np.random.exponential
        },
        1, 1 / 2
    )
    sim = GsmpSimulation(mmc)
    ps = sim.simulate(5000)

    _ss = map(lambda s: sum(s.label), ss)
    _ps = map(lambda i: sum(map(lambda j: j[1], i)), (list(g) for _, g in groupby(sorted(zip(_ss, ps)), key=itemgetter(0))))

    for y in sorted(zip(range(c + k), _ps, pi(c, c + k, 1, 2)), key=itemgetter(0)):
        print(y)
