from queues import queue1, queue2, queue3
from numpy.random import exponential


def f(new_state, new_event, old_state, trigger_event):
    if new_event == ("com", queue1):
        return exponential(0.5)
    elif new_event == ("arr", queue3):
        return exponential(0.5)
    else:
        return None     # unreachable


def r(state, event):
    return 1


def f_0(state, event):
    if state == (0, 0, 0) and event == ():
        return 1
    else:
        return 0
