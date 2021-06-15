from numpy.random import exponential
from core import Gsmp

arrival, service = 1, 2


class MM1PS(Gsmp):

    def states(self):
        s = 0
        while True:
            yield s
            s += 1

    def events(self):
        yield "arr"
        s = 1
        while True:
            yield "com {}".format(s)
            s += 1

    def e(self, s):
        if s == 0:
            return ["arr"]
        else:
            return ["arr"] + ["com {}".format(i + 1) for i in range(s)]

    def p(self, _s, e, s):
        if e == "arr":
            return int(_s == s + 1)
        else:
            return int(_s == s - 1)

    def f(self, _s, _e, s, e):
        if _e == "arr":
            return exponential(1 / self._arrival)
        else:
            return exponential(1 / self._service)

    def r(self, s, e):
        if e == "arr":
            return 1
        else:
            return 1 / s

    def s_0(self, s):
        return int(s == 0)

    def f_0(self, s, e):
        return exponential(1 / self._arrival)

    def __init__(self,
                 service_rate=service,
                 arrival_rate=arrival,
                 adjacent_states=None):
        if adjacent_states is None:
            def adjacent_states(n):
                return [1] if n == 0 else [n - 1, n + 1]
            self._service = service_rate
            self._arrival = arrival_rate
            super().__init__(adjacent_states=adjacent_states)
