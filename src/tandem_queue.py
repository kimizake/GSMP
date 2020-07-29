from src.gsmp import *
import numpy as np
from itertools import chain, product

d = 3
c = 5


class tandem_queue(GsmpSpec):
    def __init__(self, states, events):
        super(tandem_queue, self).__init__(states, events)

    def e(self, s: State) -> list:
        def func(queue, capacity, capacity_of_next_queue):
            out = []
            if queue == 0 and capacity < c:
                out.append((queue, 'arr'))
            if capacity > 0 and capacity_of_next_queue < c:
                out.append((queue, 'com'))
            return out
        return [Event(i) for i in chain(*map(func, range(d), s.label, np.append(s.label[1:], 0)))]

    def p(self, _s: State, s: State, e: Event) -> float:
        (queue, event) = e.label
        if event == 'arr' and queue == 0:
            if _s.label[queue] == s.label[queue] + 1 and _s.label[queue] <= c:
                return 1 if np.all(np.where(_s.label != s.label)[0] == np.array([queue])) else 0
        if event == 'com':
            if _s.label[queue] == s.label[queue] - 1 and _s.label[queue] >= 0:
                if not queue + 1 >= d:
                    if _s.label[queue + 1] == s.label[queue + 1] + 1 and _s.label[queue + 1] <= c:
                        return 1 if np.all(np.where(_s.label != s.label) == np.array([queue, queue + 1])) else 0
                    else:
                        return 0
                return 1 if np.all(np.where(_s.label != s.label) == np.array([queue])) else 0
        return 0

    def f(self, _s: State, _e: Event, s: State, e: Event) -> float:
        # (queue, event) = e.label
        (_queue, _event) = _e.label
        # if event == _event == 'arr':
        #     assert queue == _queue == 0, "events in different queues"
        #     assert np.where(_s.label != s.label) == (np.array([0]),), "state mismatch"
        #     return np.random.exponential(1)
        # if event == _event == 'com':
        #     assert queue == _queue, "events in different queues"
        #     if queue >= d:
        #         assert np.where(_s.label != s.label) == (np.array([queue]),), "state mismatch"
        #     else:
        #         assert np.where(_s.label != s.label) == (np.array([queue, queue + 1]),), "state mismatch"
        #     return np.random.exponential(1 / 2)
        # if event == 'arr' and _event == 'com':
        #     assert queue == _queue, "events in different queues"
        #
        # if event == 'com' and _event == 'arr':
        #     assert queue == _queue, "events in different queues"
        # raise TypeError
        return np.random.exponential(1) if _event == 'arr' else np.random.exponential(1 / 2)

    def r(self, s: State, e: Event) -> float:
        return 1

    def s_0(self, s: State) -> float:
        return 1 if np.all(s.label == np.zeros(d)) else 0

    def f_0(self, e: Event, s: State) -> float:
        if s == State(np.zeros(d)) and e == Event((0, 'arr')):
            return np.random.exponential(1)
        raise ValueError


es = [Event((0, 'arr'))] + [Event(i) for i in product(list(range(d)), ['com'])]
ss = [State(i) for i in np.array(list(product(range(c + 1), repeat=d)))]

if __name__ == "__main__":
    q = tandem_queue(ss, es)
    sim = GsmpSimulation(q)
    dist = sim.simulate(1000)
    from operator import itemgetter
    for x in sorted(zip(ss, dist), key=itemgetter(1), reverse=True):
        print(x)
