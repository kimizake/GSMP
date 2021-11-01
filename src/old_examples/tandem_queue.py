from src.gsmp import *
import numpy as np
from itertools import chain, product
from ggc import pi as mm1_pi

d = 3
c = 1


class tandem_queue(Gsmp):
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
        return [Event(i) for i in chain(*map(func, range(d), s.name, np.append(s.name[1:], 0)))]

    def p(self, _s: State, s: State, e: Event) -> float:
        (queue, event) = e.name
        if event == 'arr' and queue == 0:
            if _s.name[queue] == s.name[queue] + 1 and _s.name[queue] <= c:
                return 1 if np.all(np.where(_s.name != s.name)[0] == np.array([queue])) else 0
        if event == 'com':
            if _s.name[queue] == s.name[queue] - 1 and _s.name[queue] >= 0:
                if not queue + 1 >= d:
                    if _s.name[queue + 1] == s.name[queue + 1] + 1 and _s.name[queue + 1] <= c:
                        return int(np.all(np.where(np.not_equal(_s.name, s.name))[0] == np.array([queue, queue + 1])))
                    else:
                        return 0
                return int(np.all(np.where(np.not_equal(_s.name, s.name))[0] == np.array([queue])))
        return 0

    def f(self, _s: State, _e: Event, s: State, e: Event) -> float:
        # (queue, event) = e.label
        (_queue, _event) = _e.name
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
        return 1 if np.all(s.name == np.zeros(d)) else 0

    def f_0(self, e: Event, s: State) -> float:
        if s == State(np.zeros(d)) and e == Event((0, 'arr')):
            return np.random.exponential(1)
        raise ValueError


es = [Event((0, 'arr'))] + [Event(i) for i in product(list(range(d)), ['com'])]
ss = [State(i) for i in list(product(range(c + 1), repeat=d))]

mm1_ps = {i: p for i, p in enumerate(mm1_pi(1, c, 1, 2))}


def pi(*ks):
    return np.prod(list(mm1_ps[k] for k in ks))


if __name__ == "__main__":
    q = tandem_queue(ss, es)
    sim = Simulator(q)
    dist = sim.simulate(1000)
    from operator import itemgetter
    exp = map(lambda s: pi(*s.name), ss)
    for x in sorted(zip(ss, dist, exp), key=itemgetter(2), reverse=True):
        print(x)
