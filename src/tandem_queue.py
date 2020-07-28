from src.gsmp import *
import numpy as np
from itertools import chain, product

d = 5
c = 5


def gen_vectors():
    arr = np.zeros(d)
    for i in range(d):
        for j in range(c):
            yield np.copy(arr)
            arr[i] += 1
    yield arr


class tandem_queue(GsmpSpec):
    def __init__(self, states, events):
        super(tandem_queue, self).__init__(states, events)

    def e(self, s: State) -> list:
        def func(index, num):
            out = []
            if index == 0 and num < c:
                out.append((index, 'arr'))
            if num > 0:
                out.append((index, 'com'))
            return out
        return [Event(i) for i in chain(*map(func, range(d), s.label))]

    def p(self, _s: State, s: State, e: Event) -> float:
        (q, event) = e.label
        if event == 'arr' and q == 0:
            if _s.label[q] == s.label[q] + 1 and _s.label[q] <= c:
                return 1
        if event == 'com':
            if _s.label[q] == s.label[q] - 1 and _s.label[q] >= 0:
                return 1
        return 0

    def f(self, _s: State, _e: Event, s: State, e: Event) -> float:
        pass

    def r(self, s: State, e: Event) -> float:
        return 1

    def s_0(self, s: State) -> float:
        return 1 if s == State(np.zeros(d)) else 0

    def f_0(self, e: Event, s: State) -> float:
        if s == State(np.zeros(d)) and e == Event((0, 'arr')):
            return np.random.exponential(1)
        raise ValueError


es = [Event((0, 'arr'))] + [Event(i) for i in product(list(range(d)), ['com'])]
ss = [State(i) for i in gen_vectors()]

if __name__ == "__main__":
    queue = tandem_queue(ss, es)
    sim = GsmpSimulation(queue)
    sim.simulate(10)
