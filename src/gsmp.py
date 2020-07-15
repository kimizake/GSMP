import numpy as np
from src.bitmap import BitMap


class Event:
    def __init__(self, label):
        self.label = label
        self.clock = None

    def set_clock(self, clock):
        self.clock = clock

    def tick_down(self, time):
        self.clock -= time
        assert self.clock >= 0

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
        self.time_spent = 0

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self.label == other.label
        return False

    def __hash__(self):
        return hash(self.label)

    def __repr__(self):
        return self.label


class Gsmp:
    def __init__(self, s, e, p, r, f, s_0, f_0):
        """
        :param s: list of all State objects in simulation
        :param e: list of all Event objects in simulation
        :param p: function taking args (s', s, e)
        :param r: function taking args (s, e)
        :param f: function taking args (s', e', s, e)
        :param s_0: Initial state setting function
        :param f_0: Initial clock distribution function
        """
        self.states = s
        self.events = e
        self.probabilities = p
        self.rates = r
        self.clock_distribution = f

        self.initial_state = np.random.choice(
            s,
            p=[s_0(_s) for _s in s]
        )
        self.initial_distribution = f_0

        self.current_state = self.initial_state
        self.old_events = 0
        self.cancelled_events = 0
        self.new_events = 0
        self.active_events = 0

        self.bitmap = BitMap(e)
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

    def set_current_state(self, new_state, winning_event):
        e = self.current_state.events - self.bitmap.positions[winning_event]
        e_prime = new_state.events
        old_events = e & e_prime
        new_events = e_prime ^ old_events
        self.old_events = old_events
        self.cancelled_events = e ^ old_events
        self.new_events = new_events
        self.active_events = old_events | new_events
        self.current_state = new_state

    def set_old_clock(self, s, t):
        for _e in self.bitmap.get(self.old_events):
            _e.tick_down(t * self.rates(s, _e))

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
        total_time = 0
        while epochs > 0:
            old_state = self.current_state
            active_events = self.bitmap.get(self.active_events)

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

            old_state.time_spent += time_elapsed
            total_time += time_elapsed

            """
            Determine next state
            """
            ps = self.probabilities(self.states, old_state, winning_event)
            new_state = np.random.choice(self.states, p=ps)

            if new_state == self.initial_state:
                self.set_initial_state()
            else:
                self.set_current_state(new_state, winning_event)
                self.set_old_clock(old_state, time_elapsed)
                self.set_new_clocks(old_state, winning_event)

            """
            print output
            """
            # print("s:{0}, \te:{1}, \ts':{2},".format(old_state, winning_event, self.current_state))

            epochs -= 1
        return total_time
