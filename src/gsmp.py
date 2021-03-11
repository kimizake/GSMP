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

    def get_states(self):
        return self.states

    def get_current_state(self):
        return self.current_state

    def get_active_events(self):
        return self.bitmap.get(self.active_events)

    def get_new_state(self, o, e):
        return np.random.choice(
            self.current_state.adjacent_nodes,
            p=[self.p(_s, o, e) for _s in self.current_state.adjacent_nodes]
        )

    def choose_winning_event(self, o, es):
        tmp = []
        for e in es:
            try:
                tmp.append(e.clock / self.r(o, e))
            except ZeroDivisionError:
                pass
            except ValueError:
                pass
        try:
            t = np.amin(tmp)
        except Exception:
            pass
        event_index = np.where(tmp == t)[0]
        # Usually there is only 1 winning event, but in the case of a tie, randomly select a winner
        winning_events = [es[i] for i in event_index]
        return np.random.choice(winning_events), t

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

    @staticmethod
    def update_state_time(s, t):
        s.time_spent += t


class GsmpComposition:

    def __init__(self, *args):
        self.nodes = args
        # TODO: check preconditions for composition

    def find_gsmp(self, e):
        """
            return list of indexes of gsmp nodes which contain event e
            TODO: optimise with hash-map.
        """
        out = []
        for i, node in enumerate(self.nodes):
            if e in node.events: out.append(i)
        return out

    def get_states(self):
        return list(chain.from_iterable(node.get_states() for node in self.nodes))

    def get_current_state(self):
        return list(n.get_current_state() for n in self.nodes)

    def get_active_events(self):
        # TODO: filtering for shared events
        return dict((i, n.get_active_events()) for i, n in enumerate(self.nodes))

    def get_new_state(self, o, e):
        """
            Given a list of old states and a singular event,
            go to the active gsmps and call new state
            Return a list of just the updated states
        """
        return list(self.nodes[i].get_new_state(o[i], e) for i in self.find_gsmp(e))

    def choose_winning_event(self, o, es):
        """
        :param o: list of current states in each node
        :param es: hashmap of active events
        :return: winning, time passed
        """
        ttl = dict(self.nodes[index].choose_winning_event(o[index], events) for index, events in es.items())
        winner = min(ttl, key=ttl.get)
        return winner, ttl[winner]

    def set_current_state(self, s, e):
        """
            Given a list of updated states, we now need to enumerate to update nodes.
        """
        for i, j in enumerate(self.find_gsmp(e)):
            self.nodes[j].set_current_state(s[i], e)

    def set_old_clock(self, s, t):
        """
            s is a list of old states, which will be a 1-1 mapping with our gsmp nodes
        """
        for i, node in enumerate(self.nodes):
            node.set_old_clock(s[i], t)

    def set_new_clocks(self, s, e):
        """
            Given the winning event a set of old states (1-1 map with gsmp nodes)
            go to revelevant nodes and update event clocks
        """
        for i in self.find_gsmp(e):
            self.nodes[i].set_new_clocks(s[i], e)

    @staticmethod
    def update_state_time(s, t):
        for _s in s:
            _s.time_spent += t


class Simulator:
    def __init__(self, gsmp: Gsmp):
        self.g = gsmp
        self.total_time = 0

    def simulate(self, epochs):
        while epochs > 0:
            old_state = self.g.get_current_state()
            active_events = self.g.get_active_events()

            winning_event, time_elapsed = self.g.choose_winning_event(old_state, active_events)

            # Increment timing metrics
            self.g.update_state_time(old_state, time_elapsed)
            self.total_time += time_elapsed

            # Select new state
            new_state = self.g.get_new_state(old_state, winning_event)

            # Update trackers for next iteration
            self.g.set_current_state(new_state, winning_event)
            self.g.set_old_clock(old_state, time_elapsed)
            self.g.set_new_clocks(old_state, winning_event)

            # print("s={0}, e={1}, s'={2}".format(old_state, winning_event, new_state))

            epochs -= 1
        return map(lambda s: s.time_spent / self.total_time, self.g.get_states())
