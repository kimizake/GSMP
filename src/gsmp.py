import numpy as np
from bitmap import BitMap
from abc import ABCMeta, abstractmethod


class Event:
    def __init__(self, node, name):
        self.name = name
        self.clock = None
        self.frozen = False
        self.synchronised = False
        self.__hash_value = name, node

    def get_name(self):
        return self.name

    def set_clock(self, clock):
        self.clock = clock

    def tick_down(self, time):
        if self.frozen:
            return
        assert self.clock >= time
        self.clock -= time

    def set_hash_value(self, *args):
        self.__hash_value = tuple(args)

    def __eq__(self, other):
        return isinstance(other, Event) and hash(self) == hash(other)

    def __hash__(self):
        return hash(self.__hash_value)

    def __repr__(self):
        return str(self.name)


class State:
    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.events = None
        self.time_spent = 0

    def get_name(self):
        return self.name

    def set_events(self, events):
        self.events = events

    def __eq__(self, other):
        return isinstance(other, State) and self.index == other.index

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return str(self.name)


class Gsmp(metaclass=ABCMeta):

    def __init__(self):
        #TODO: add a save feature after pre-processing

        # Construct state and event objects
        self.__states = [State(index, name) for index, name in enumerate(self.states())]
        self.__events = [Event(self, name) for name in self.events()]

        # define adjacent states
        for state in self.__states:
            es = self._e(state)
            state.adjacent_nodes = list(
                {_s if self._p(_s, state, e) != 0 else None for _s in self.__states for e in es} - {None}
            )

        # choose initial state
        self.initial_state = np.random.choice(
            self.__states,
            p=[self._s_0(state) for state in self.__states]
        )

        # dfs prune states
        visited = {s: False for s in self.__states}

        def visit(s):
            if not visited[s]:
                visited[s] = True
                for _s in s.adjacent_nodes:
                    visit(_s)
        visit(self.initial_state)
        for state in self.__states:
            if not visited[state]:
                self.__states.remove(state)
        del visited

        # generate bitmap
        self.bitmap = BitMap(self.__events)
        for state in self.__states:
            state.set_events(self.bitmap.format(self._e(state)))

        # set trackers
        self.current_state = self.initial_state
        self.old_events = 0
        self.cancelled_events = 0
        self.new_events = self.current_state.events
        self.active_events = self.new_events
        self.set_new_clocks(None, None)

    # Define interface through abstract methods
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'states') and callable(subclass.states) and
                hasattr(subclass, 'events') and callable(subclass.events) and
                hasattr(subclass, 'e') and callable(subclass.e) and
                hasattr(subclass, 'p') and callable(subclass.p) and
                hasattr(subclass, 'f') and callable(subclass.f) and
                hasattr(subclass, 'r') and callable(subclass.r) and
                hasattr(subclass, 's_0') and callable(subclass.s_0) and
                hasattr(subclass, 'f_0') and callable(subclass.f_0) or
                NotImplementedError)

    @abstractmethod
    def states(self) -> list:
        """Define the countable set of states"""
        raise NotImplementedError

    @abstractmethod
    def events(self) -> list[str]:
        """Define the finite set of events"""
        raise NotImplementedError

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

    # Define wrapper functions
    @staticmethod
    def get_names(*args):
        def get_name(param):
            return param.get_name()
        return tuple(map(get_name, args))

    def _e(self, *args):
        return [Event(self, name) for name in self.e(*Gsmp.get_names(*args))]

    def _p(self, *args):
        return self.p(*Gsmp.get_names(*args))

    def _f(self, *args):
        return self.f(*Gsmp.get_names(*args))

    def _r(self, *args):
        return self.r(*Gsmp.get_names(*args))

    def _s_0(self, *args):
        return self.s_0(*Gsmp.get_names(*args))

    def _f_0(self, *args):
        return self.f_0(*Gsmp.get_names(*args))

    def get_states(self):
        return self.__states

    def get_events(self):
        return self.__events
        # return dict(event.get_hash_value() for event in self.__events)

    def get_current_state(self):
        return self.current_state

    def get_active_events(self):
        # return list(filter(
        #     lambda event: not event.frozen, self.bitmap.get(self.active_events)
        # ))
        return list(self.bitmap.get(self.active_events))

    def get_new_state(self, o, e):
        try:
            return np.random.choice(
                self.current_state.adjacent_nodes,
                p=[self._p(_s, o, e) for _s in self.current_state.adjacent_nodes]
            )
        except ValueError as v:
            raise v

    def choose_winning_event(self, o, es):
        tmp = []
        for e in es:
            try:
                tmp.append(e.clock / self._r(o, e))
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
        try:
            e = self.current_state.events - self.bitmap.positions[winning_event]
            e_prime = new_state.events
            old_events = e & e_prime
            new_events = e_prime ^ old_events
            self.old_events = old_events
            self.cancelled_events = e ^ old_events
            self.new_events = new_events
            self.active_events = old_events | new_events
            self.current_state = new_state
        except KeyError as k:
            raise KeyError

    def set_old_clock(self, s, t):
        for _e in self.bitmap.get(self.old_events):
            _e.tick_down(t * self._r(s, _e))

    def set_new_clocks(self, s, e):
        _s = self.current_state
        for _e in self.bitmap.get(self.new_events):
            try:
                _e.set_clock(
                    self._f_0(_e, _s) if _s == self.initial_state else
                    self._f(_s, _e, s, e)
                )
            except ValueError as E:
                print(E)

    @staticmethod
    def update_state_time(s, t):
        s.time_spent += t


class GsmpComposition:
    """
    Normal event handling can be deferred to sub-process with relevant vector components
    However synchronised events may have more complicated setting functions,
    therefore the user has the choice of yielding setter functions for probability, clock distributions and rates.
    """

    def __init__(self, *args, synchros=None, p=None, f=None, r=None):
        self.nodes = args

        # Configure the synchronised events to have hashing equality
        for synchro in synchros:
            name, node = synchro[0]
            {event.get_name(): event for event in node.get_events()}[name].synchronised = True
            for _name, _node in synchro[1:]:
                e = {event.get_name(): event for event in _node.get_events()}[_name]
                e.synchronised = True
                v = _node.bitmap.positions.pop(e)
                e.set_hash_value(name, node)
                _node.bitmap.positions[e] = v

        # TODO: check preconditions for composition

        # Create hash from event to node indexes
        find_gsmp = {}
        for i, n in enumerate(self.nodes):
            es = n.get_events()
            for e in es:
                if e in find_gsmp:
                    find_gsmp[e].append((i, e))
                else:
                    find_gsmp[e] = [(i, e)]
        self.find_gsmp = find_gsmp

        self.state_times = dict()

    # def get_states(self):
    #     return list(chain.from_iterable(node.get_states() for node in self.nodes))

    def get_current_state(self):
        """
        Return current state vector
        """
        return list(n.get_current_state() for n in self.nodes)

    def get_active_events(self):
        """
        Return some data structure with current event information
        """
        active_events = {i: n.get_active_events() for i, n in enumerate(self.nodes)}

        """
        If the nodes disagree on whether a synchronised event is active in the current state,
        then that event must be frozen and filtered out of the active events
        """
        from itertools import filterfalse
        for i, es in active_events.items():
            def is_event_active_in_all_states(e):
                if e.synchronised:
                    for j, _e in self.find_gsmp[e]:
                        if e not in active_events[j]:
                            e.frozen = True
                            return False
                        if j < i:
                            return False
                e.frozen = False
                return True
            es[:] = filter(is_event_active_in_all_states, es)

        return {k: v for k, v in active_events.items() if v}

    def get_new_state(self, o, e):
        """
        Given a list of old states and a singular event,
        go to the active gsmps and call new state
        Return a list of just the updated states
        """
        return list(self.nodes[i].get_new_state(o[i], _e) for i, _e in self.find_gsmp[e])

    def choose_winning_event(self, o, es):
        """
        Use the active event data structure to pick the winner
        :param o: list of current states in each node
        :param es: hashmap of active events
        :return: winning, time passed
        """
        ttl = dict(self.nodes[index].choose_winning_event(o[index], events) for index, events in es.items())
        # from operator import itemgetter
        # return min(ttl, key=itemgetter(1))
        winner = min(ttl, key=ttl.get)
        return winner, ttl[winner]

    def set_current_state(self, s, e):
        """
            Given a list of updated states, we now need to enumerate to update nodes.
        """
        for i, (index, event) in enumerate(self.find_gsmp[e]):
            self.nodes[index].set_current_state(s[i], event)

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
        for i, _e in self.find_gsmp[e]:
            self.nodes[i].set_new_clocks(s[i], _e)

    def update_state_time(self, s, t):
        for _s in s:
            _s.time_spent += t
        s = tuple(s)
        if s not in self.state_times:
            self.state_times[s] = t
        else:
            self.state_times[s] += t

    def get_probability_distribution(self, total_time):
        return list(map(lambda state, holding_time: (state, holding_time / total_time), self.state_times.items()))


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

            #print("event {0} fired at time {1} ------- new state {2}".format(winning_event, self.total_time, self.g.get_current_state()))

            epochs -= 1
        # return map(lambda s: s.time_spent / self.total_time, self.g.get_states())
