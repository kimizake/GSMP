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

        visited = dict((state, False) for state in states)

        self.initial_state = np.random.choice(
            states,
            p=[self.s_0(state) for state in states]
        )

        # dfs prune states

        def visit(s):
            if not visited[s]:
                visited[s] = True
                for _s in s.adjacent_nodes:
                    visit(_s)

        visit(self.initial_state)

        for state in states:
            if not visited[state]:
                self.states.remove(state)

        del(visited)

        # generate bitmap

        bitmap = BitMap(events)
        for state in self.states:
            state.set_events(bitmap.format(self.e(state)))
            # print("s={0}, E(s)={1}".format(state, self.e(state)))

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
        self.gsmps = gsmps
        self.syncs = syncs
        states = list(self.CompoundState(i) for i in product(*(g.states for g in gsmps)))  # Enumerate cartesian product of all state spaces
        events = list(self.CompoundEvent(g, event, gsmps.index(g)) for g in gsmps for event in g.events)   # Union of all Event sets
        for _, v in syncs.items():
            events = list(filter(lambda e: not e == v, events))
        super().__init__(states, events)

    def e(self, s) -> list:
        es = list(chain.from_iterable(map(lambda g, v: map(lambda e: (g, e), g.e(v)), self.gsmps, s)))

        def filter_blocked_events(event):
            try:
                events = self.syncs[event]
                for e in events:
                    if e not in es:
                        return False
            except KeyError:
                pass
            return True
        es = list(filter(filter_blocked_events, es))
        return list(filter(lambda e: e in es, self.events))

    def p(self, _s, s, e) -> float:
        p = list(int(x == y) for x, y in zip(_s, s))

        es = [e]
        try:
            es.extend(self.syncs[(e.g, e.e)])
            pass
        except KeyError:
            pass
        finally:
            for _g, _e in es:
                i = self.gsmps.index(_g)
                p[i] = _g.p(_s[i], s[i], _e)

        import math
        return math.prod(p)

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
        self.gsmp = gsmp

        self.initial_state = np.random.choice(
            gsmp.states,
            p=[gsmp.s_0(_s) for _s in gsmp.states]
        )
        self.initial_distribution = gsmp.f_0

        self.current_state = self.initial_state
        self.old_events = 0
        self.cancelled_events = 0
        self.new_events = self.current_state.events
        self.active_events = self.new_events

        self.bitmap = BitMap(gsmp.events)
        self.set_new_clocks(None, None)

        self.total_time = 0

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
            _e.tick_down(t * self.gsmp.r(s, _e))

    def set_new_clocks(self, s, e):
        _s = self.current_state
        for _e in self.bitmap.get(self.new_events):
            try:
                _e.set_clock(
                    self.gsmp.f_0(_e, _s) if _s == self.initial_state else
                    self.gsmp.f(_s, _e, s, e)
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
                    tmp.append(event.clock / self.gsmp.r(old_state, event))
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
                self.current_state.adjacent_nodes,
                p=[self.gsmp.p(_s, old_state, winning_event) for _s in self.current_state.adjacent_nodes]
            )

            self.set_current_state(new_state, winning_event)
            self.set_old_clock(old_state, time_elapsed)
            self.set_new_clocks(old_state, winning_event)

            # print("s={0}, e={1}, s'={2}".format(old_state, winning_event, new_state))

            epochs -= 1
        return map(lambda s: s.time_spent / self.total_time, self.gsmp.states)
