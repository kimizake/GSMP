from core import Gsmp
from numpy.random import exponential

k = 20
arrival_rate = 1
service_rate = 2


class MM1k(Gsmp):
    def states(self):
        return range(self._k + 1)

    def events(self):
        return ['arr', 'com']

    def e(self, s):
        es = []
        if s < self._k:
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
            return exponential(self._avg_arrival_time)
        else:
            return exponential(self._avg_service_time)

    def r(self, s, e):
        return 1

    def s_0(self, s):
        return int(s == 0)

    def f_0(self, s, e):
        return exponential(self._avg_arrival_time)

    def __repr__(self):
        if self.name is None:
            return hex(id(self))
        return str(self.name)

    def __init__(self, name=None, k=k, _arrival_rate=arrival_rate, _service_rate=service_rate):
        self.name = name
        self._k = k
        self._avg_arrival_time = 1 / _arrival_rate
        self._avg_service_time = 1 / _service_rate
        super().__init__()


if __name__ == "__main__":
    from core import Simulator
    res = Simulator(MM1k()).run(until=1000, estimate_probability=True)
    print(res)
