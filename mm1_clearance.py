from gsmp import *
from bitmap import BitMap
import numpy as np

N = 20

Events = {'arr': Event('arr'), 'com': Event('com'), 'clr': Event('clr')}

bitmap = BitMap(list(Events.values()))

States = [State(
    str(0), bitmap.format([Events['arr'], Events['clr']])
)]

States.extend([State(
    str(i), bitmap.all()
) for i in range(1, N)])

States.append(State(
    str(N), bitmap.format([Events['com'], Events['clr']])
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
    elif e == Events['clr']:
        return [1 if States.index(_s) == 0 else 0 for _s in ss]
    else:
        raise ValueError


distributions = {
    'arr': np.random.weibull,
    'com': np.random.weibull,
    'clr': np.random.weibull,
}


def f(_s, _e, s, e):
    if e == Events['arr']:
        if States.index(_s) == States.index(s) + 1:
            if States.index(s) in range(0, N) and _e == Events['arr']:
                return 0.5 * distributions['arr'](10)
            if States.index(s) == 0 and _e == Events['com']:
                return 0.5 * distributions['com'](10)
    elif e == Events['com']:
        if States.index(_s) == States.index(s) - 1:
            if States.index(s) in range(2, N+1) and _e == Events['com']:
                return 0.5 * distributions['com'](10)
            if States.index(s) == N and _e == Events['arr']:
                return 0.5 * distributions['arr'](10)
    elif e == Events['clr']:
        if States.index(_s) == 0:
            if _e == Events['clr']:
                return 0.5 * distributions['clr'](10)
            if _e == Events['arr'] and States.index(s) == N - 1:
                return 0.5 * distributions['arr'](10)
    raise ValueError


def f_0(s, e):
    if States.index(s) == 0 and e == Events['arr']:
        return 0.5 * distributions['arr'](10)
    elif States.index(s) == 0 and e == Events['clr']:
        return 0.5 * distributions['clr'](10)
    else:
        raise ValueError


if __name__ == "__main__":
    simulation = Gsmp(States, list(Events.values()), p, r, f, States[0], f_0)
    simulation.simulate(300)
