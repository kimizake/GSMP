from gsmp import *
from bitmap import BitMap
import numpy as np

N = 5

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
    'arr': np.random.uniform,
    'com': np.random.uniform,
}


def f(_s, _e, s, e):
    if e == Events['arr']:
        if States.index(_s) == States.index(s) + 1:
            if States.index(s) in range(0, N) and _e == Events['arr']:
                return distributions['arr']()
            if States.index(s) == 0 and _e == Events['com']:
                return distributions['com']()
    elif e == Events['com']:
        if States.index(_s) == States.index(s) - 1:
            if States.index(s) in range(2, N+1) and _e == Events['com']:
                return distributions['com']()
            if States.index(s) == N and _e == Events['arr']:
                return distributions['arr']()
    raise ValueError


def f_0(s, e):
    if States.index(s) == 0 and e == Events['arr']:
        return distributions['arr'](5)
    else:
        raise ValueError


if __name__ == "__main__":
    simulation = Gsmp(States, list(Events.values()), p, r, f, States[0], f_0)
    simulation.simulate(30)
