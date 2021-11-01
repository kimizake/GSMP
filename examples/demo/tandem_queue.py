from examples.demo.mm1 import MM1
from numpy.random import exponential

arrival, service = 1, 2


def adj_state(s):
    return [1] if s == 0 else [s - 1, s + 1]


queue1 = MM1(adjacent_states=adj_state)
queue2 = MM1(adjacent_states=adj_state)
queue3 = MM1(adjacent_states=adj_state)


def f(_s, _e, s, e):
    if _e == ('com', queue1):
        return exponential(1 / service)
    if _e == ('com', queue2):
        return exponential(1 / service)
    raise ValueError     # Unreachable


tandem_queue = queue1 + queue2 + queue3

tandem_queue.shared_events = [
    [('com', queue1), ('arr', queue2)],
    [('com', queue2), ('arr', queue3)]
]

tandem_queue.f = f
