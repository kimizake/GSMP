import numpy as np
from bitmap import BitMap
from abc import ABCMeta, abstractmethod
from itertools import chain, product


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
        return str(self.label)


class State:
    def __init__(self, label):
        self.label = label
        self.events = None
        self.time_spent = 0

    def set_events(self, events):
        self.events = events

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return np.all(self.label == other.label)
        return False

    def __hash__(self):
        return hash(self.label)

    def __repr__(self):
        return str(self.label)


class Gsmp(metaclass=ABCMeta):
    def __init__(self, states, events):

        #TODO: add a save feature after pre-processing

        self.states = states
        self.events = events

        # define adjacent states
        for state in states:
            es = self.e(state)
            state.adjacent_nodes = list(
                {_s if self.p(_s, state, e) != 0 else None for _s in states for e in es} - {None}
            )

        self.initial_state = np.random.choice(
            states,
            p=[self.s_0(state) for state in states]
        )

        # dfs prune states
        visited = dict((state, False) for state in states)
        def visit(s):
            if not visited[s]:
                visited[s] = True
                for _s in s.adjacent_nodes:
                    visit(_s)
        visit(self.initial_state)
        for state in states:
            if not visited[state]:
                self.states.remove(state)
        del visited

        # generate bitmap
        self.bitmap = BitMap(events)
        for state in self.states:
            state.set_events(self.bitmap.format(self.e(state)))

        # set trackers
        self.current_state = self.initial_state
        self.old_events = 0
        self.cancelled_events = 0
        self.new_events = self.current_state.events
        self.active_events = self.new_events
        self.set_new_clocks(None, None)

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
    def f(self, _s: State, _e: Event, s: State, e: Event) -> float:
        """Returns the distribution function used to set the clock for new event _e
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
    def f_0(self, e: Event, s: State) -> float:
        """Returns the distribution function to set the clock of event e in initial state s"""
        raise NotImplementedError

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
            _e.tick_down(t * self.r(s, _e))

    def set_new_clocks(self, s, e):
        _s = self.current_state
        for _e in self.bitmap.get(self.new_events):
            try:
                _e.set_clock(
                    self.f_0(_e, _s) if _s == self.initial_state else
                    self.f(_s, _e, s, e)
                )
            except ValueError as E:
                print(E)

class GsmpComposition(Gsmp):

    class CompoundState(State):
        def __init__(self, states):
            super().__init__(states)

        def __getitem__(self, item):
            return self.label[item]

    class CompoundEvent(Event):
        def __init__(self, g, e, i):
            super().__init__(e.label)
            self.i = i
            self.e = e
            self.g = g

        def __eq__(self, other):
            if isinstance(self, type(other)):
                return self.g == other.g and self.e == other.e
            elif isinstance(other, tuple):
                return other == (self.g, self.e)
            return False

        def __hash__(self):
            return hash(self.e)

        def __iter__(self):
            return iter((self.g, self.e))

        def __repr__(self):
            return repr((self.i, self.e))

    def __init__(self, syncs, *gsmps):
        self.nodes = gsmps  # List of gsmp
        self.syncs = syncs  # List of tuples - equivalency of events from different gsmp nodes
        self.states = list(
            chain.from_iterable(map(lambda s: (g, s), g.states) for g in gsmps)
        )
        self.events = list(
            chain.from_iterable(map(lambda e: (g, e), g.events) for g in gsmps)
        )

    def bitmap(self, item):
        g, _ = item
        return g.bitmap

    def e(self, s) -> list:
        g, s = s
        return list(map(lambda e: (g, e), g.e(s)))

    def p(self, _s, s, e) -> float:
        g, s = s
        _g, _s = _s
        __g, e = e
        if g == _g == __g:
            return g.p(_s, s, e)
        if __g == g:
            ...
        if __g == _g:
            ...

        return 0

    def f(self, _s, _e, s, e) -> float:
        es = [e]
        _es = [_e]
        try:
            es.extend(self.syncs[(e.g, e.e)])
        except KeyError:
            pass
        try:
            _es.extend(self.syncs[(_e.g, _e.e)])
        except KeyError:
            pass
        finally:
            for k, v in es:
                for _k, _v in _es:
                    if k == _k:
                        i = self.gsmps.index(k)
                        return k.f(_s[i], _v, s[i], v)
        raise Exception

    def r(self, s, e) -> float:
        g, _e = e
        i = self.gsmps.index(g)
        return g.r(s[i], _e)

    def s_0(self, s) -> float:
        from math import prod
        return prod(map(lambda x, g: g.s_0(x), s.label, self.gsmps))

    def f_0(self, e, s) -> float:
        g, _e = e
        i = self.gsmps.index(g)
        return g.f_0(_e, s[i])


class Simulator:
    def __init__(self, gsmp: Gsmp):
        self.g = gsmp
        self.total_time = 0

    def simulate(self, epochs):
        while epochs > 0:
            old_state = self.g.current_state
            active_events = self.g.bitmap.get(self.g.active_events)

            """
            Determine winning event
            """
            tmp = []
            for event in active_events:
                try:
                    tmp.append(event.clock / self.g.r(old_state, event))
                except ZeroDivisionError:
                    pass
                except ValueError:
                    pass
            try:
                time_elapsed = np.amin(tmp)
            except Exception:
                pass
            event_index = np.where(tmp == time_elapsed)[0]
            winning_events = [active_events[i] for i in event_index]
            winning_event = np.random.choice(winning_events)

            old_state.time_spent += time_elapsed
            self.total_time += time_elapsed

            """
            Determine next state
            """
            new_state = np.random.choice(
                self.g.current_state.adjacent_nodes,
                p=[self.g.p(_s, old_state, winning_event) for _s in self.g.current_state.adjacent_nodes]
            )

            self.g.set_current_state(new_state, winning_event)
            self.g.set_old_clock(old_state, time_elapsed)
            self.g.set_new_clocks(old_state, winning_event)

            # print("s={0}, e={1}, s'={2}".format(old_state, winning_event, new_state))

            epochs -= 1
        return map(lambda s: s.time_spent / self.total_time, self.g.states)
