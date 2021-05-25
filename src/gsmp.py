from abc import ABCMeta, abstractmethod
import numpy as np
from bitmap import BitMap
from itertools import chain, filterfalse, product
import operator
from functools import cache


class Event:
    def __init__(self, node, name):
        self._name = name
        self._node = node
        self._shared = False
        self._shared_events = {node: name}
        self._total_shared_events = 1

    def get_name(self, process=None):
        if process is None and not self._shared:
            return self._name
        return self._shared_events[process]

    def get_default_process(self):
        return self._node

    @cache
    def get_shared_process(self, event):
        # Called with event where it is known that they share exactly one common process
        return (set(self._shared_events) & set(event.get_shared_events())).pop()

    @property
    def shared(self):
        return self._shared

    @shared.setter
    def shared(self, val):
        self._shared = val

    def add_shared_event(self, node, name):
        self._shared_events[node] = name
        self._total_shared_events += 1

    def get_shared_events(self):
        return self._shared_events

    def total_shared_events(self):
        return self._total_shared_events

    def __eq__(self, other):
        return isinstance(other, Event) and (
            (self._name == other._name and self._node == other._node) or
            (self._shared and
             other._node in self._shared_events and self._shared_events[other._node] == other._name) or
            (other._shared and
             self._node in other._shared_events and other._shared_events[self._node] == self._name)
        )

    def __hash__(self):
        return hash((self._name, self._node))

    def __repr__(self):
        return str(self._name + '@' + str(self._node))


class SimulationObject(metaclass=ABCMeta):

    @classmethod
    def __subclasscheck__(cls, subclass):
        return (
                hasattr(subclass, 'reset') and callable(subclass.reset) and
                hasattr(subclass, 'get_states') and callable(subclass.get_states) and
                hasattr(subclass, 'get_current_state') and callable(subclass.get_current_state) and
                hasattr(subclass, 'get_active_events') and callable(subclass.get_active_events) and
                hasattr(subclass, 'choose_winning_event') and callable(subclass.choose_winning_event) and
                hasattr(subclass, 'get_new_state') and callable(subclass.get_new_state) and
                hasattr(subclass, 'set_current_state') and callable(subclass.set_current_state) and
                hasattr(subclass, 'set_old_clocks') and callable(subclass.set_old_clocks) and
                hasattr(subclass, 'set_new_clocks') and callable(subclass.set_new_clocks) or
                NotImplementedError
        )

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def get_states(self):
        raise NotImplementedError

    @abstractmethod
    def get_current_state(self):
        raise NotImplementedError

    @abstractmethod
    def get_active_events(self, s):
        raise NotImplementedError

    @abstractmethod
    def choose_winning_event(self, o, e):
        raise NotImplementedError

    @abstractmethod
    def get_new_state(self, o, e):
        raise NotImplementedError

    @abstractmethod
    def set_current_state(self, n):
        raise NotImplementedError

    @abstractmethod
    def set_old_clocks(self, t, n, e, o):
        raise NotImplementedError

    @abstractmethod
    def set_new_clocks(self, n, e, o):
        raise NotImplementedError


class GsmpWrapper(SimulationObject):
    """
    Proxy class to add simulation function to user defined GSMP
    """
    __gsmp__ = None
    _events = None
    _current_state = None

    def __init__(self, obj, adjacent_states=None):
        # TODO: add a save feature after pre-processing

        if not isinstance(obj, Gsmp):
            raise TypeError('wrapped object must be of type %s' % Gsmp)
        self.__gsmp__ = obj

        # Construct state and event objects
        self._events = [Event(self, name) for name in self.__gsmp__.events()]

        self._clock = None
        self._bitmap = None

        if adjacent_states is not None:
            self._get_adj_states = adjacent_states
        else:
            # The default get_adjacent_state function has to iterate over the entire state space and event set
            self._get_adj_states = lambda s: {
                _s for _s in self.get_states() for e in self._bitmap.get(self._e(s)) if self._p(_s, e, s) != 0
            }

    def reset(self):
        # choose initial state
        def take_candidates(_iter):
            _sum = 0
            while _sum < 1:
                s, p = next(_iter)
                _sum += p
                yield s, p
            if _sum > 1:
                raise ValueError('Invalid pdf for initial state distribution')
            return

        # only take states
        states, probabilities = zip(*take_candidates(map(lambda s: (s, self._s_0(s)), self.get_states())))
        initial_state = states[np.random.choice(
            len(states),        # Because numpy requires 1-D array-like value, select from the index
            p=probabilities
        )]
        del states, probabilities

        if self._clock:
            del self._clock
        if self._bitmap:
            del self._bitmap
        self._clock = {}
        self._bitmap = BitMap(self._events)
        # set trackers
        self._current_state = initial_state
        # set initial clocks
        for event in self.get_active_events(initial_state):
            self.set_clock(initial_state, event, None, None, f=self._f_0(initial_state, event))

    def set_clock(self, next_state, new_event, old_state, trigger_event, f=None):
        if f is None:
            f = self._f(next_state, new_event, old_state, trigger_event)
        self._clock[new_event] = f

    def reduce_clock(self, state, event, time=0, r=None):
        if r is None:
            r = self._r(state, event)
        self._clock[event] -= time * r

    def time_delta(self, state, event, r=None):
        if r is None:
            r = self._r(state, event)
        return self._clock[event] / r

    def swap_event(self, original, new):
        # replace original in event list
        i = self._events.index(original)
        self._events[i] = new
        # clear memory
        del original

    @cache
    def _e(self, state):
        return self._bitmap.format([Event(self, name) for name in self.__gsmp__.e(state)])

    def _p(self, next_state, event, old_state):
        return self.__gsmp__.p(next_state, event.get_name(process=self), old_state)

    def _f(self, next_state, new_event, old_state, trigger_event):
        return self.__gsmp__.f(next_state, new_event.get_name(process=self),
                               old_state, trigger_event.get_name(process=self))

    def _r(self, state, event):
        return self.__gsmp__.r(state, event.get_name(process=self))

    def _s_0(self, state):
        return self.__gsmp__.s_0(state)

    def _f_0(self, state, event):
        return self.__gsmp__.f_0(state, event.get_name(process=self))

    def get_states(self):
        return self.__gsmp__.states()

    @cache
    def get_adj_states(self, state):
        return list(self._get_adj_states(state))

    def get_events(self):
        return self._events

    def get_current_state(self):
        return self._current_state

    def get_active_events(self, state):
        return self._bitmap.get(self._e(state))

    def get_old_events(self, new_state, trigger_event, old_state):
        e1 = self._e(old_state)
        try:
            e1 -= self._bitmap.positions[trigger_event]
        except KeyError:
            pass
        e2 = self._e(new_state)
        return self._bitmap.get(e1 & e2)

    def get_new_events(self, new_state, trigger_event, old_state):
        e1 = self._e(old_state)
        try:
            e1 -= self._bitmap.positions[trigger_event]
        except KeyError:
            pass
        e2 = self._e(new_state)
        return self._bitmap.get(e2 - (e1 & e2))

    def get_new_state(self, o, e):
        adj_states = self.get_adj_states(o)
        return adj_states[np.random.choice(
            len(adj_states),
            p=[self._p(_s, e, o) for _s in adj_states]
        )]

    def choose_winning_event(self, o, es):
        tmp = list(map(lambda e: (e, self.time_delta(o, e)), es))
        return min(tmp, key=operator.itemgetter(1))

    def set_current_state(self, new_state):
        self._current_state = new_state

    def set_old_clocks(self, time, new_state, trigger_event, old_state):
        for old_event in self.get_old_events(new_state, trigger_event, old_state):
            self.reduce_clock(old_state, old_event, time=time)

    def set_new_clocks(self, next_state, trigger_event, old_state):
        for new_event in self.get_new_events(next_state, trigger_event, old_state):
            self.set_clock(next_state, new_event, old_state, trigger_event)

    def __add__(self, other):
        return Compose(self, other)


class Gsmp(GsmpWrapper, metaclass=ABCMeta):

    def __init__(self, adjacent_states=None):
        super().__init__(self, adjacent_states=adjacent_states)

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
    def states(self):
        """
        Define the countable set of states
        :return: Iterable of state identifiers
        """
        raise NotImplementedError

    @abstractmethod
    def events(self):
        """
        Define the finite set of events
        :return: Iterable of event identifiers
        """
        raise NotImplementedError

    @abstractmethod
    def e(self, s):
        """
        Returns the set of events scheduled to occur in state
        :param s: state identifier
        :return: Iterable of event identifiers
        """
        raise NotImplementedError

    @abstractmethod
    def p(self, _s, e, s):
        """
        Returns the probability of the next state being _s when event e occurs in state s
        :param _s: next state identifier
        :param e: event identifier
        :param s: current state identifier
        :return: float between 0 and 1
        """
        raise NotImplementedError

    @abstractmethod
    def f(self, _s, _e, s, e):
        """
        Returns the distribution function used to set the clock for new event _e
         when event e triggers a transition from state s to state _s
        :param _s: new state identifier
        :param _e: new event identifier
        :param s: old state identifier
        :param e: transition event identifier
        :return: sample from random distribution
        """
        raise NotImplementedError

    @abstractmethod
    def r(self, s, e):
        """
        Returns the rate at which the clock for event e runs down in state s
        :param s: state identifier
        :param e: event identifier
        :return: float
        """
        raise NotImplementedError

    @abstractmethod
    def s_0(self, s):
        """
        Returns the probability of state s being the initial state
        :param s: state identifier
        :return: float between 0 and 1
        """
        raise NotImplementedError

    @abstractmethod
    def f_0(self, s, e):
        """
        Returns the distribution function to set the clock of event e in initial state s
        :param s: state identifier
        :param e: event identifier
        :return: float
        """
        raise NotImplementedError


class Compose(SimulationObject):
    """
    Normal event handling can be deferred to sub-process with relevant vector components
    However synchronised events may have more complicated setting functions,
    therefore the user has the choice of yielding setter functions for probability, clock distributions and rates.
    """

    _f = None
    _r = None
    _shared_events = None

    def __init__(self, *args, shared_events=None, f=None, r=None):
        _nodes = []
        for arg in args:
            # Parse arguments
            if isinstance(arg, Gsmp):
                _nodes.append(arg)
            elif isinstance(arg, Compose):
                _nodes.extend(arg.nodes)
            else:
                raise TypeError('Bad composition')
        # Nodes data-structure tracks sub-processes and their 'position' in the state vector
        self.nodes = {arg: i for i, arg in enumerate(_nodes)}
        self.shared_events = shared_events
        self.override_clocks(f=f, r=r)

    @property
    def shared_events(self):
        return self._shared_events

    @shared_events.setter
    def shared_events(self, val):
        self._shared_events = val

    def override_clocks(self, f=None, r=None):
        # Override clock functions.
        if f:
            self._f = lambda _s, _e, s, e: f(_s, _e.get_name(), s, e.get_name())
        if r:
            self._r = lambda s, e: r(s, e.get_name())

    def reset(self):
        total_events = {
            (e.get_name(), e.get_default_process()): e
            for e in chain.from_iterable(node.get_events() for node in self.nodes)
        }

        # Configure the synchronised events to have hashing equality
        for event_mapping in self.shared_events:
            name, node = event_mapping[0]   # Take default as first one
            e = total_events[(name, node)]
            for _name, _node in event_mapping[1:]:
                _e = total_events[(_name, _node)]
                e.add_shared_event(_node, _name)
                _node.swap_event(_e, e)
            e.shared = True  # take e as the default for this synchronisation

        del total_events

        for node in self.nodes:
            node.reset()

    def get_states(self):
        # This takes the cartesian product of the states, therefore it is really slow
        return product(*(node.get_states() for node in self.nodes))

    def get_current_state(self):
        """
        Return current state vector
        """
        return tuple(n.get_current_state() for n in self.nodes)

    @staticmethod
    def _get_events(events):
        events = list(chain.from_iterable(events))    # flatten the list
        from collections import Counter
        event_counter = Counter(events)

        def is_illegal(e):
            return e.shared and (
                    # when the event has a different status across its nodes,
                    e.total_shared_events() != event_counter[e]
            )

        return set(filterfalse(is_illegal, events))    # filter out 'illegal' events

    def get_active_events(self, state):
        """
        :return: list of all active events
        """
        return Compose._get_events(node.get_active_events(state[i]) for node, i in self.nodes.items())

    def get_old_events(self, new_state, trigger_event, old_state):
        return Compose._get_events(node.get_old_events(new_state[i], trigger_event, old_state[i])
                                   for node, i in self.nodes.items())

    def get_new_events(self, new_state, trigger_event, old_state):
        # TODO look for a smarter way

        new_active_events = self.get_active_events(new_state)
        old_events = self.get_old_events(new_state, trigger_event, old_state)
        return new_active_events - old_events

    def get_new_state(self, o, e):
        """
        return list of new states
        """
        # Get e's parent subprocesses
        nodes = e.get_shared_events()

        def get_new_node_state(node):
            if node in nodes:                                       # when event 'e' is in gsmp 'node'
                return node.get_new_state(o[self.nodes[node]], e)   # that 'node' will enter a new state
            return node.get_current_state()                         # otherwise its state doesn't change
        return tuple(map(get_new_node_state, self.nodes))

    def choose_winning_event(self, o, es):
        """
        Use the active event data structure to pick the winner
        :param o: list of current states in each node
        :param es: list of active events
        :return: winning, time passed
        """
        def get_time_deltas(event):
            parent = event.get_default_process()
            if event.shared and self._r is not None:
                return event, parent.get_clock(event) / self._r(o, event)
            return event, parent.time_delta(o[self.nodes[parent]], event)

        time_deltas = list(map(get_time_deltas, es))  # calculate how long each active event spends in current state
        return min(time_deltas, key=operator.itemgetter(1))     # Return event with smallest time_delta

    def set_current_state(self, s):
        """
        set current state to s, and update all event status trackers
        """
        for node, i in self.nodes.items():
            node.set_current_state(s[i])

    def set_old_clocks(self, time, new_state, trigger_event, old_state):
        """
        Decrement all old clocks
        """
        for event in self.get_old_events(new_state, trigger_event, old_state):

            r = self._r(old_state, event) if event.shared and self._r is not None else None
            parent = event.get_default_process()
            i = self.nodes[parent]
            parent.reduce_clock(old_state[i], event, time=time, r=r)

    def set_new_clocks(self, new_state, trigger_event, old_state):
        """
        Set all new clocks
        """
        for new_event in self.get_new_events(new_state, trigger_event, old_state):
            f = self._f(new_state, new_event, old_state, trigger_event) \
                if new_event.shared and self._f is not None else None
            parent = new_event.get_default_process()
            i = self.nodes[parent]
            if new_event.shared and self._f is not None:
                f = self._f(new_state, new_event, old_state, trigger_event)
            elif new_event.shared:
                # Niche edge case where trigger event is unknown to default process of new_event
                # Need to work out the shared process between new event and trigger event
                # And get clock setting function from there
                _parent = trigger_event.get_shared_process(new_event)
                j = self.nodes[_parent]
                f = _parent._f(new_state[j], new_event, old_state[j], trigger_event)

            parent.set_clock(new_state[i], new_event, old_state[i], trigger_event, f=f)

    def __add__(self, other):
        return Compose(*self.nodes, other)


class Simulator:
    def __init__(self, gsmp: SimulationObject):
        self.g = gsmp

    def run(self, epochs, warmup=0):
        """
        Run a new simulation, generates sample paths through the GSMP state spaces
        :param epochs: number of event firings
        :param warmup: 'warmup' epochs
        :return: observed states, holding times, total simulation time
        """
        self.g.reset()
        self._run(warmup)
        return self._run(epochs)

    def _run(self, epochs):
        state_holding_times = {}
        total_time = 0
        while epochs > 0:
            old_state, trigger_event, new_state, time_delta = self._generate_path()

            # Increment timing metrics
            if old_state not in state_holding_times:
                state_holding_times[old_state] = 0
            state_holding_times[old_state] += time_delta
            total_time += time_delta

            # print("event {0} fired at time {1} ------- new state {2}".format(
            # winning_event, self.total_time, self.g.get_current_state()))

            epochs -= 1
        return (
            list(state_holding_times.keys()),     # observed states
            list(state_holding_times.values()),   # holding times
            total_time                            # total simulation time
        )

    def _generate_path(self):
        # Get old state and active events
        old_state = self.g.get_current_state()
        active_events = self.g.get_active_events(old_state)
        # Select trigger event and find time from last event
        trigger_event, time_delta = self.g.choose_winning_event(old_state, active_events)
        # Select new state
        new_state = self.g.get_new_state(old_state, trigger_event)
        # Update clocks for next loop
        self.g.set_old_clocks(time_delta, new_state, trigger_event, old_state)
        self.g.set_new_clocks(new_state, trigger_event, old_state)

        self.g.set_current_state(new_state)

        return old_state, trigger_event, new_state, time_delta
