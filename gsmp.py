import numpy as np
from bitmap import BitMap


class Event:
    def __init__(self, label):
        self.label = label
        self.clock = 0
        self.current_time = 0

    def set_clock(self, clock):
        self.clock = clock
        self.current_time = clock

    def tick_down(self, time):
        self.current_time -= time
        self.current_time = self.current_time % self.clock
        if self.current_time == 0:
            self.current_time = self.clock

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self.label == other.label
        return False

    def __hash__(self):
        return hash(self.label)

    def __repr__(self):
        return self.label


class State:
    def __init__(self, label, events):
        self.label = label
        self.events = events
        self.visits = 0

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self.label == other.label
        return False

    def __hash__(self):
        return hash(self.label)

    def __repr__(self):
        return self.label


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

        self.initial_state = S_0
        self.initial_distribution = F_0

        self.current_state = S_0
        self.old_events = 0
        self.cancelled_events = 0
        self.new_events = 0
        self.active_events = 0

        self.bitmap = BitMap(E)
        self.set_initial_state()

    def set_initial_state(self):
        self.current_state = self.initial_state
        self.old_events = 0
        self.cancelled_events = 0
        self.new_events = self.initial_state.events
        self.active_events = self.initial_state.events

        for _e in self.bitmap.get(self.new_events):
            try:
                _e.set_clock(self.initial_distribution(self.initial_state, _e))
            except ValueError:
                pass

    def set_current_state(self, new_state):
        e = self.current_state.events
        e_prime = new_state.events
        old_events = e & e_prime
        new_events = e_prime ^ old_events
        self.old_events = old_events
        self.cancelled_events = e ^ old_events
        self.new_events = new_events
        self.active_events = old_events | new_events
        self.current_state = new_state

    def set_old_clock(self, s, e, t):
        for _e in self.bitmap.get(self.old_events):
            _e.tick_down(t * self.rates(s, e))

    def set_new_clocks(self, s, e):
        _s = self.current_state
        for _e in self.bitmap.get(self.new_events):
            try:
                _e.set_clock(
                    self.clock_distribution(_s, _e, s, e)
                )
            except ValueError as E:
                print(E)

    def simulate(self, epochs):
        while epochs > 0:
            old_state = self.current_state
            active_events = self.bitmap.get(self.active_events)

            """
            Determine winning event
            """
            tmp = []
            for event in active_events:
                try:
                    tmp.append(event.current_time / self.rates(old_state, event))
                except ZeroDivisionError:
                    pass
                except ValueError:
                    pass
            time_elapsed = np.amin(tmp)
            event_index = np.where(tmp == time_elapsed)[0]
            winning_events = [active_events[i] for i in event_index]
            winning_event = np.random.choice(winning_events)

            """
            Determine next state
            """
            ps = self.probabilities(self.states, old_state, winning_event)
            new_state = np.random.choice(self.states, p=ps)

            new_state.visits += 1

            if new_state == self.initial_state:
                self.set_initial_state()
            else:
                self.set_current_state(new_state)
                self.set_old_clock(old_state, winning_event, time_elapsed)
                self.set_new_clocks(old_state, winning_event)

            """
            print output
            """
            # print("s:{0}, \te:{1}, \ts':{2},".format(old_state, winning_event, self.current_state))

            epochs -= 1
