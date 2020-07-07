import numpy as np
from functools import reduce

DISTRIBUTION = {
    0: np.random.uniform
}


def get_items_from_binary_list(b, l):
    s = str(b)[2:]
    assert len(s) == len(l)
    return [l[int(i)] for i in s if i == '1']


def convert_sub_list_to_binary(s, l):
    assert set(s).issubset(set(l))
    s = '0b'
    s += ['1' if i in s else '0' for i in l]
    return bin(int(s, 2))


class Event:
    def __init__(self, label):
        self.label = label
        self.clock = 0
        self.active = False

    def set_clock(self, clock):
        self.clock = clock

    def __eq__(self, other):
        if isinstance(self, other):
            return self.label == other.label
        return False


class State:
    def __init__(self, label, events):
        self.label = label
        self.events = events

    def __eq__(self, other):
        if isinstance(self, other):
            return self.label == other.label
        return False


class Gsmp:
    def __init__(self, S, E, P, R, F, S_0, F_0):
        """
        :param S: hashmap of all State objects in simulation
        :param E: list of all Event objects in simulation
        :param P: function taking args (s', s, e)
        :param R: function taking args (s, e)
        :param F: function taking args (s', e', s, e)
        :param S_0: Initial state
        :param F_0: Initial clock distribution
        """
        self.states = S
        self.events = E
        self.probabilities = P
        self.rates = R
        self.clock_distribution = F

        self.current_states = S_0

        tmp = convert_sub_list_to_binary([], E)
        self.old_events = tmp
        self.cancelled_events = tmp
        self.new_events = S_0.events
        self.active_events = S_0.events

        for _e in get_items_from_binary_list(self.new_events, self.events.values()):
            try:
                _e.set_clock(F_0(S_0, _e))
            except TypeError:
                pass

    def set_current_state(self, new_states):
        import operator
        e = reduce(operator.or_, map(lambda s: s.events, self.current_states))
        e_prime = reduce(operator.or_, map(lambda s: s.events, new_states))
        self.old_events = e & e_prime
        self.cancelled_events = e ^ self.old_events
        self.new_events = e_prime ^ self.old_events
        self.active_events = self.old_events | self.new_events
        self.current_states = new_states

    def set_old_clock(self, old_states, t):
        old_events = get_items_from_binary_list(self.old_events, self.events.values())
        for e in old_events:
            e.clock -= t * self.rates(old_states, e)

    def set_new_clocks(self, old_states, winning_events):
        new_states = self.current_states
        for _e in get_items_from_binary_list(self.new_events, self.events.values()):
            _e.set_clock(
                self.clock_distribution(new_states, _e, old_states, winning_events)     # We can determine which
            )

    def simulate(self, epochs):
        if epochs > 0:
            old_states = self.current_states
            active_events = get_items_from_binary_list(self.active_events, self.events.values())

            """
            Determine winning event(s)
            Note we know if an event exists in multiple 'current states' then the 'winning state' for that event is the 
            one with the highest rate!
            Therefore when supplied with a list of states, the probability, rates and distribution functions choose the
            'winning state'. 
            TODO: For now assume that there is only one state with a maximum rate, and generalise later...
            Also note that this means that multiple events can fire at the same time only if they are different events.
            """
            tmp = [event.clock / self.rates(old_states, event) for event in active_events]
            event_index = np.where(tmp == np.amin(tmp))[0]
            winning_events = [active_events[i] for i in event_index]
            time_elapsed = np.amin(tmp)

            """
            Determine next states
            """
            def pick_new_state(winning_event):
                rand = np.random.uniform()
                ps = self.probabilities(self.states.values(), old_states, winning_event)
                new_state_index = np.max(np.clip(ps, 0, rand))
                new_state = self.states[new_state_index]
                return new_state
            new_states = map(pick_new_state, winning_events)

            """
            update state
            """
            self.set_current_state(new_states)

            """
            update old clocks
            """
            self.set_old_clock(old_states, time_elapsed)

            """
            update new clocks
            """
            self.set_new_clocks(old_states, winning_events)

            """
            misc
            """
            ...

            return self.simulate(epochs - 1)
        return self
