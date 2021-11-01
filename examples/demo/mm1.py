from core import Gsmp
from numpy.random import exponential

arrival, service = 1, 2


class MM1(Gsmp):
    def states(self):
        s = 0
        while True:
            yield s
            s += 1

    def events(self):
        return ['arr', 'com']

    def e(self, s):
        if s == 0:
            return ['arr']
        return ['arr', 'com']

    def p(self, _s, e, s):
        if e == 'arr':
            return int(_s == s + 1)
        return int(_s == s - 1)

    def f(self, _s, _e, s, e):
        if _e == 'arr':
            return exponential(1 / arrival)
        return exponential(1 / service)

    def r(self, s, e):
        return 1

    def s_0(self, s):
        return int(s == 0)

    def f_0(self, s, e):
        return exponential(1 / arrival)
