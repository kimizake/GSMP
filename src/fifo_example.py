from src.gsmp import *
from src.bitmap import BitMap
import numpy as np

N = 10

l = 1   # arrival rate
m = 2   # service rate


Events = {'arr': Event('arr'), 'com': Event('com')}

bitmap = BitMap(list(Events.values()))

States = [State(
    str(0), bitmap.format([Events['arr']])
)]

States.extend([State(
    str(i), bitmap.all()
) for i in range(1, N)])

States.append(State(
    str(N), bitmap.format([Events['com']])
))


def r(state, event):
    if event in bitmap.get(state.events):
        return 1
    else:
        raise ValueError


def p(ss, s, e):
    if e == Events['arr']:
        return [1 if States.index(_s) == States.index(s) + 1 else 0 for _s in ss]
    elif e == Events['com']:
        return [1 if States.index(_s) == States.index(s) - 1 else 0 for _s in ss]
    else:
        raise ValueError


distributions = {
    'arr': np.random.exponential,
    'com': np.random.exponential,
}


def f(_s, _e, s, e):
    if e == Events['arr']:
        if States.index(_s) == States.index(s) + 1:
            if States.index(s) in range(0, N) and _e == Events['arr']:
                return distributions['arr'](1/l)
            if States.index(s) == 0 and _e == Events['com']:
                return distributions['com'](1/m)
    elif e == Events['com']:
        if States.index(_s) == States.index(s) - 1:
            if States.index(s) in range(2, N+1) and _e == Events['com']:
                return distributions['com'](1/m)
            if States.index(s) == N and _e == Events['arr']:
                return distributions['arr'](1/l)
    raise ValueError


def s_0(s):
    return 1 if s.label == '0' else 0


def f_0(s, e):
    if States.index(s) == 0 and e == Events['arr']:
        return distributions['arr'](1)
    else:
        raise ValueError


if __name__ == "__main__":
    simulation = Gsmp(States, list(Events.values()), p, r, f, s_0, f_0)
    total_time = simulation.simulate(1000)
    from functools import reduce
    print(reduce(lambda x, y: x + y, [int(_s.label) * _s.time_spent for _s in States]) / total_time)
    """
    lambda = 1
    s = 0.5
    rho = 0.5
    var(s) = 0.25
    L = 0.5 + (0.25 + 0.25)/2(0.5) = 1
    """
