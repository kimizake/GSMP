from gsmp import *
import numpy as np

N = 5

Events = {'arr': Event('arr'), 'com': Event('com')}
States = [State(chr(0), [Events['arr']])]

States.extend([(i, State(chr(i), Events.values())) for i in range(1, N)])

States.append(State(chr(N), [Events['com']]))


def r(state, event):
    if event in state.events:
        return 1
    else:
        raise ValueError


def p(_s, s, e):
    if e == Events['arr'] and States.index(_s) == State.index(s) + 1:
        return 1
    elif e == Events['com'] and States.index(_s) == States.index(s) - 1:
        return 1
    return 0


distributions = {
    'arr': np.random.uniform,
    'com': np.random.uniform,
}


def f(_s, _e, s, e):
    if e == Events['arr']:
        if States.index(_s) == States.index(s) + 1:
            if States.index(s) in range(0, N) and _e == Events['arr']:
                return distributions['arr'](5, 10)
            if States.index(s) == 0 and _e == Events['com']:
                return distributions['com'](5, 10)
    elif e == Events['com']:
        if States.index(_s) == States.index(s) - 1:
            if States.index(s) in range(2, N+1) and _e == Events['com']:
                return distributions['com'](5, 10)
            if States.index(s) == N and _e == Events['arr']:
                return distributions['arr'](5, 10)
    raise ValueError


def f_0(s, e):
    if States.index(s) == 0 and e == Events['arr']:
        return distributions['arr'](5, 10)
    else:
        raise ValueError
