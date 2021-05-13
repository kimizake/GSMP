from gsmp import State, Event, Gsmp, Simulator
from itertools import product
from numpy.random import exponential
from operator import itemgetter
from ggc import pi
from math import factorial, prod

k = 10

ss = list(
    State(i) for i in product(range(k + 1), repeat=2)
)
es = [
    Event('a'), Event('c1'), Event('c2')
]


class tandem_queue(Gsmp):

    def e(self, s: State) -> list:
        (a, b) = s.name
        res = list()
        if a < k:
            res.append(Event('a'))
        if a > 0 and b < k:
            res.append(Event('c1'))
        if b > 0:
            res.append(Event('c2'))
        return res

    def p(self, _s: State, s: State, e: Event) -> float:
        (a, b) = s.name
        (_a, _b) = _s.name
        if e.name == 'a':
            return int(b == _b and _a == a + 1 and _a <= k)
        if e.name == 'c1':
            return int(_a == a - 1 and _a >= 0 and _b == b + 1 and _b <= k)
        if e.name == 'c2':
            return int(a == _a and _b == b - 1 and _b >= 0)
        return 0

    def f(self, _s: State, _e: Event, s: State, e: Event) -> float:
        if _e.name == 'a':
            return exponential(scale=1)
        if _e.name == 'c1':
            return exponential(scale=.5)
        if _e.name == 'c2':
            return exponential(scale=.5)
        return 0

    def r(self, s: State, e: Event) -> float:
        return 1

    def s_0(self, s: State) -> float:
        return int(s == State((0, 0)))

    def f_0(self, e: Event, s: State) -> float:
        return exponential(scale=1)


def mmc_pi(c, k, p):
    p0 = 1 / (sum((c * p) ** i / factorial(i) for i in range(c)) + (c * p) ** c / factorial(c) / (1 - p))
    if k == 0:
        return p0
    elif 0 < k < c:
        return p0 * (c * p) ** k / factorial(k)
    elif c <= k:
        return p0 * (c * p) ** k * c ** (c - k) / factorial(c)
    else:
        raise Exception


if __name__ == "__main__":
    res = Simulator(tandem_queue(ss, es)).simulate(10000)

    q1 = pi(1, k, 1, 2)
    q2 = pi(1, k, 1, 2)

    _q = list(product(q1, q2))
    m = list(map(prod, _q))

    def ps(*ks):
        return prod(mmc_pi(1, k, .5) for k in ks)

    exp = list(map(lambda s: ps(*s.name), ss))

    for x in sorted(zip(ss, res, m, exp), key=itemgetter(2), reverse=True):
        print(x)
