from gsmp import Gsmp
from numpy.random import exponential

k = 10
arrival_rate = 1
service_rate = 2

avg_arrival_time = 1 / arrival_rate
avg_service_time = 1 / service_rate


class Tandem_queue(Gsmp):
    def states(self):
        from itertools import product
        return product(range(k + 1), repeat=2)  # cartesian product

    def events(self):
        return ['arr', 'com1', 'com2']

    def e(self, s):
        (x, y) = s
        es = []
        if x < k:
            es.append('arr')
        if x > 0 and y < k:
            es.append('com1')
        if y > 0:
            es.append('com2')
        return es

    def p(self, _s, s, e):
        x1, y1 = s
        x2, y2 = _s
        if e == 'arr':
            return bool(x1 + 1 == x2 and y1 == y2)
        elif e == 'com1':
            return bool(x1 - 1 == x2 and y1 + 1 == y2)
        else:
            return bool(x1 == x2 and y1 - 1 == y2)

    def f(self, _s, _e, s, e):
        if _e == 'arr':
            return exponential(avg_arrival_time)
        else:
            return exponential(avg_service_time)

    def r(self, s, e):
        return 1

    def s_0(self, s):
        return int(s == (0, 0))

    def f_0(self, e, s):
        return exponential(avg_arrival_time)
