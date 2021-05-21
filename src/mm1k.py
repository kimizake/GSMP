from gsmp import Gsmp
from numpy.random import exponential

k = 20
arrival_rate = 1
service_rate = 2
avg_arrival_time = 1 / arrival_rate
avg_service_time = 1 / service_rate


class MM1k(Gsmp):
    def states(self):
        return range(k + 1)

    def events(self):
        return ['arr', 'com']

    def e(self, s):
        es = []
        if s < k:
            es.append('arr')
        if s > 0:
            es.append('com')
        return es

    def p(self, _s, e, s):
        if e == 'arr':
            return int(s + 1 == _s)
        else:
            return int(s - 1 == _s)

    def f(self, _s, _e, s, e):
        if _e == 'arr':
            return exponential(avg_arrival_time)
        else:
            return exponential(avg_service_time)

    def r(self, s, e):
        return 1

    def s_0(self, s):
        return int(s == 0)

    def f_0(self, s, e):
        return exponential(avg_arrival_time)

    def __repr__(self):
        return str(self.name)

    def __init__(self, name):
        self.name = name
        super().__init__()
