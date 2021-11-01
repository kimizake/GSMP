from core import Gsmp
from numpy.random import exponential
from itertools import product

k = 20
arrival_rate = 1
service_rate = 2
avg_arrival_time = 1 / arrival_rate
avg_service_time = 1 / service_rate


class Tandem_queue(Gsmp):

    def f(self, _s, _e, s, e):
        if _e == "arr":
            return exponential(avg_arrival_time)
        else:
            return exponential(avg_service_time)

    def r(self, s, e):
        return 1

    def f_0(self, s, e):
        return exponential(avg_arrival_time)

    def __init__(self, k=k):
        self._k = k
        super().__init__()


class Tandem_queue2(Tandem_queue):
    def states(self):
        # cartesian product
        return product(range(self._k + 1), repeat=2)

    def events(self):
        return ["arr", "com1", "com2"]

    def e(self, s):
        (x, y) = s
        es = []
        if x < self._k:
            es.append("arr")
        if x > 0 and y < self._k:
            es.append("com1")
        if y > 0:
            es.append("com2")
        return es

    def p(self, _s, e, s):
        x1, y1 = s
        x2, y2 = _s
        if e == "arr":
            return bool(x1 + 1 == x2 and y1 == y2)
        elif e == "com1":
            return bool(x1 - 1 == x2 and y1 + 1 == y2)
        else:
            return bool(x1 == x2 and y1 - 1 == y2)

    def s_0(self, s):
        return int(s == (0, 0))


class Tandem_queue3(Tandem_queue):
    def states(self):
        # cartesian product
        return product(range(self._k + 1), repeat=3)

    def events(self):
        return ["arr", "com1", "com2", "com3"]

    def e(self, s):
        (x, y, z) = s
        es = []
        if x < self._k:
            es.append("arr")
        if x > 0 and y < self._k:
            es.append("com1")
        if y > 0 and z < self._k:
            es.append("com2")
        if z > 0:
            es.append("com3")
        return es

    def p(self, _s, e, s):
        x1, y1, z1 = s
        x2, y2, z2 = _s
        if e == "arr":
            return bool(x1 + 1 == x2 and y1 == y2 and z1 == z2)
        elif e == "com1":
            return bool(x1 - 1 == x2 and y1 + 1 == y2 and z1 == z2)
        elif e == "com2":
            return bool(x1 == x2 and y1 - 1 == y2 and z1 + 1 == z2)
        else:
            return bool(x1 == x2 and y1 == y2 and z1 - 1 == z2)

    def s_0(self, s):
        return int(s == (0, 0, 0))


class Tandem_queue4(Tandem_queue):
    def states(self):
        # cartesian product
        return product(range(self._k + 1), repeat=4)

    def events(self):
        return ["arr", "com1", "com2", "com3", "com4"]

    def e(self, s):
        (x, y, z, a) = s
        es = []
        if x < self._k:
            es.append("arr")
        if x > 0 and y < self._k:
            es.append("com1")
        if y > 0 and z < self._k:
            es.append("com2")
        if z > 0 and a < self._k:
            es.append("com3")
        if a > 0:
            es.append("com4")
        return es

    def p(self, _s, e, s):
        x1, y1, z1, a1 = s
        x2, y2, z2, a2 = _s
        if e == "arr":
            return bool(x1 + 1 == x2 and
                        y1 == y2 and
                        z1 == z2 and
                        a1 == a2)
        elif e == "com1":
            return bool(x1 - 1 == x2 and
                        y1 + 1 == y2 and
                        z1 == z2 and
                        a1 == a2)
        elif e == "com2":
            return bool(x1 == x2 and
                        y1 - 1 == y2 and
                        z1 + 1 == z2 and
                        a1 == a2)
        elif e == "com3":
            return bool(x1 == x2 and
                        y1 == y2 and
                        z1 - 1 == z2 and
                        a1 + 1 == a2)
        else:
            return bool(x1 == x2 and
                        y1 == y2 and
                        z1 == z2 and
                        a1 - 1 == a2)

    def s_0(self, s):
        return int(s == (0, 0, 0, 0))


class Tandem_queue5(Tandem_queue):
    def states(self):
        # cartesian product
        return product(range(self._k + 1), repeat=5)

    def events(self):
        return ["arr", "com1", "com2", "com3", "com4", "com5"]

    def e(self, s):
        (x, y, z, a, b) = s
        es = []
        if x < self._k:
            es.append("arr")
        if x > 0 and y < self._k:
            es.append("com1")
        if y > 0 and z < self._k:
            es.append("com2")
        if z > 0 and a < self._k:
            es.append("com3")
        if a > 0 and b < self._k:
            es.append("com4")
        if b > 0:
            es.append("com5")
        return es

    def p(self, _s, e, s):
        x1, y1, z1, a1, b1 = s
        x2, y2, z2, a2, b2 = _s
        if e == "arr":
            return bool(x1 + 1 == x2 and
                        y1 == y2 and
                        z1 == z2 and
                        a1 == a2 and
                        b1 == b2)
        elif e == "com1":
            return bool(x1 - 1 == x2 and
                        y1 + 1 == y2 and
                        z1 == z2 and
                        a1 == a2 and
                        b1 == b2)
        elif e == "com2":
            return bool(x1 == x2 and
                        y1 - 1 == y2 and
                        z1 + 1 == z2 and
                        a1 == a2 and
                        b1 == b2)
        elif e == "com3":
            return bool(x1 == x2 and
                        y1 == y2 and
                        z1 - 1 == z2 and
                        a1 + 1 == a2 and
                        b1 == b2)
        elif e == "com4":
            return bool(x1 == x2 and
                        y1 == y2 and
                        z1 == z2 and
                        a1 - 1 == a2 and
                        b1 + 1 == b2)
        else:
            return bool(x1 == x2 and
                        y1 == y2 and
                        z1 == z2 and
                        a1 == a2 and
                        b1 - 1 == b2)

    def s_0(self, s):
        return int(s == (0, 0, 0, 0, 0))
