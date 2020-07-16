import numpy as np
from src.bitmap import BitMap
from abc import ABCMeta, abstractmethod


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
    def __init__(self, label):
        self.label = label
        self.events = 0
        self.time_spent = 0

    def set_events(self, events):
        self.events = events

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self.label == other.label
        return False

    def __hash__(self):
        return hash(self.label)

    def __repr__(self):
        return self.label


class GsmpSpec(metaclass=ABCMeta):
    def __init__(self, states, events):
        self.states = states
        self.events = events
        bitmap = BitMap(events)
        for state in states:
            state.set_events(bitmap.format(self.e(state)))

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'e') and callable(subclass.e) and
                hasattr(subclass, 'p') and callable(subclass.p) and
                hasattr(subclass, 'f') and callable(subclass.f) and
                hasattr(subclass, 'r') and callable(subclass.r) and
                hasattr(subclass, 's_0') and callable(subclass.s_0) and
                hasattr(subclass, 'f_0') and callable(subclass.f_0) or
                NotImplementedError)

    @abstractmethod
    def e(self, s: State) -> list:
        """Returns the set of events scheduled to occur in state"""
        raise NotImplementedError

    @abstractmethod
    def p(self, _s: State, s: State, e: Event) -> float:
        """Returns the probability of the next state being _s when event e occurs in state s"""
        raise NotImplementedError

    @abstractmethod
    def f(self, _s: State, _e: Event, s: State, e: Event, *args) -> float:
        """Returns the distribution function (evaluated at x) used to set the clock for new event _e
         when event e triggers a transition from state s to state _s"""
        raise NotImplementedError

    @abstractmethod
    def r(self, s: State, e: Event) -> float:
        """Returns the rate at which the clock for event e runs down in state s"""
        raise NotImplementedError

    @abstractmethod
    def s_0(self, s: State) -> float:
        """Returns the probability of state s being the initial state"""
        raise NotImplementedError

    @abstractmethod
    def f_0(self, e: Event, s: State, *args) -> float:
        """Returns the distribution function (evaluated at x) to set the clock of event e in initial state s"""
        raise NotImplementedError


class GsmpSimulation:
    def __init__(self, spec: GsmpSpec):
        self.specification = spec

        self.initial_state = np.random.choice(
            spec.states,
            p=[spec.s_0(_s) for _s in spec.states]
        )
        self.initial_distribution = spec.f_0

        self.current_state = self.initial_state
        self.old_events = 0
        self.cancelled_events = 0
        self.new_events = self.current_state.events
        self.active_events = self.new_events

        self.bitmap = BitMap(spec.events)
        self.set_new_clocks(None, None)

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
            _e.tick_down(t * self.specification.r(s, _e))

    def set_new_clocks(self, s, e):
        _s = self.current_state
        for _e in self.bitmap.get(self.new_events):
            try:
                _e.set_clock(
                    self.specification.f_0(_e, _s) if _s == self.initial_state else
                    self.specification.f(_s, _e, s, e)
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
                    tmp.append(event.clock / self.specification.r(old_state, event))
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
            new_state = np.random.choice(
                self.specification.states,
                p=[self.specification.p(_s, old_state, winning_event) for _s in self.specification.states]
            )

            self.set_current_state(new_state, winning_event)
            self.set_old_clock(old_state, time_elapsed)
            self.set_new_clocks(old_state, winning_event)

            epochs -= 1
        return total_time
