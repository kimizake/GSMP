from gsmp import *
import numpy as np

N = 5

Events = {'arr': Event('arr'), 'com': Event('com')}
States = dict()

States[0] = State(chr(0), [Events['arr']])

States.update([(i, State(chr(i), Events.values())) for i in range(1, N)])

States[N] = State(chr(N), Events['com'])


def r(state, event):
    if event in state.events:
        return 1
    else:
        return 1000     # Arbitrarily high


def winning_state(states, event):
    rates = map(lambda s: r(s, event), states)
    return states[np.where(rates == np.min(rates))[0][0]]


def get_key_from_value(value):
    for k, v in States.items():
        if v == value:
            return k


def p(new_states, states, e):
    state = winning_state(states, e)

    def helper(new_state):
        if e == Events['arr'] and get_key_from_value(new_state) == get_key_from_value(state) + 1:
            return 1
        elif e == Events['com'] and get_key_from_value(new_state) == get_key_from_value(state) - 1:
            return 1
        return 0

    return map(lambda s: helper(s), new_states)


distributions = {
    'arr': np.random.uniform,
    'com': np.random.uniform,
}


def f(new_states, new_event, old_states, old_events):

    def helper(new_state, old_state, old_event):
        if old_event == Events['arr']:
            if get_key_from_value(new_state) == get_key_from_value(old_state) + 1:
                if get_key_from_value(old_state) in range(0, N) and new_event == Events['arr']:
                    return distributions['arr'](5, 10)
                if get_key_from_value(old_state) == 0 and new_event == Events['com']:
                    return distributions['com'](5, 10)
        elif old_event == Events['com']:
            if get_key_from_value(new_state) == get_key_from_value(old_state) - 1:
                if get_key_from_value(old_state) in range(2, N+1) and new_event == Events['com']:
                    return distributions['com'](5, 10)
                if get_key_from_value(old_state) == N and new_event == Events['arr']:
                    return distributions['arr'](5, 10)
        return 1000

    return np.min(map(lambda _s, s, e: helper(_s, s, e), new_states, old_states, old_events))


def f_0(s, e):
    if get_key_from_value(s) == 0 and e == Events['arr']:
        return distributions['arr'](5, 10)
    else:
        raise TypeError