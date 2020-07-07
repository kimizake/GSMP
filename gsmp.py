import numpy as np

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
        :param S: list of all State objects in simulation
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

        self.current_state = S_0

        tmp = convert_sub_list_to_binary([], E)
        self.old_events = tmp
        self.cancelled_events = tmp
        self.new_events = S_0.events
        self.active_events = S_0.events

        for _e in get_items_from_binary_list(self.new_events, self.events.values()):
            try:
                _e.set_clock(F_0(S_0, _e))
            except ValueError:
                pass

    def set_current_state(self, new_state):
        e = self.current_state.events
        e_prime = new_state.events
        self.old_events = e & e_prime
        self.cancelled_events = e ^ self.old_events
        self.new_events = e_prime ^ self.old_events
        self.active_events = self.old_events | self.new_events
        self.current_state = new_state

    def set_old_clock(self, s, t):
        e = get_items_from_binary_list(self.old_events, self.events.values())
        for _e in e:
            _e.clock -= t * self.rates(s, e)

    def set_new_clocks(self, s, e):
        _s = self.current_state
        for _e in get_items_from_binary_list(self.new_events, self.events.values()):
            _e.set_clock(
                self.clock_distribution(_s, _e, s, e)
            )

    def simulate(self, epochs):
        if epochs > 0:
            old_state = self.current_state
            active_events = get_items_from_binary_list(self.active_events, self.events.values())

            """
            Determine winning event
            """
            tmp = []
            for event in active_events:
                try:
                    tmp.append(event.clock / self.rates(old_state, event))
                except ZeroDivisionError:
                    pass
                except ValueError:
                    pass
            time_elapsed = np.amin(tmp)
            event_index = np.where(tmp == time_elapsed)[0]
            winning_events = [active_events[i] for i in event_index]
            winning_event = np.random.choice(winning_events)

            """
            Determine next states
            """
            rand = np.random.uniform()
            ps = self.probabilities(self.states, old_state, winning_event)
            new_state_index = np.max(np.clip(ps, 0, rand))
            new_state = self.states[new_state_index]

            """
            update state
            """
            self.set_current_state(new_state)

            """
            update old clocks
            """
            self.set_old_clock(old_state, time_elapsed)

            """
            update new clocks
            """
            self.set_new_clocks(old_state, winning_event)

            """
            misc
            """
            ...

            return self.simulate(epochs - 1)
