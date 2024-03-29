from abc import ABCMeta, abstractmethod
import numpy as np
from itertools import chain, filterfalse, product
import operator
from functools import cache


class Event:
    def __init__(self, node, name):
        self._name = name
        self._node = node
        self._shared = False
        self._shared_events = {node: name}

    def get_name(self, process=None):
        if process is None or not self._shared:
            return self._name
        return self._shared_events[process]

    def get_default_process(self):
        return self._node

    @cache
    def get_shared_process(self, event):
        # Called with event where it is known that they share at least one common process
        return (set(self._shared_events) & set(event.get_shared_events())).pop()

    @property
    def shared(self):
        return self._shared

    @shared.setter
    def shared(self, val):
        self._shared = val

    def add_shared_event(self, node, name):
        self._shared_events[node] = name

    def get_shared_events(self):
        return self._shared_events

    def total_shared_events(self):
        return len(self._shared_events)

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
    _clock = None

    def __init__(self, obj, adjacent_states=None):
        if not isinstance(obj, Gsmp):
            raise TypeError('wrapped object must be of type %s' % Gsmp)
        self.__gsmp__ = obj
        self._events = {}

        if adjacent_states is None:
            def adjacent_states(s):
                return {
                    _s for _s in self.get_states() for e in self._e(s) if self._p(_s, e, s) != 0
                }
        self._get_adj_states = adjacent_states

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
        self._clock = {}
        # set trackers
        self._current_state = initial_state
        # set initial clocks
        for event in self.get_active_events(initial_state):
            self.set_clock(initial_state, event, None, None, f=self._f_0(initial_state, event))

    def get_clock(self, event):
        return self._clock[event]

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

    def get_event(self, e):
        if e not in self._events:
            self.set_event(e, Event(self, e))
        return self._events[e]

    def set_event(self, name, obj):
        self._events[name] = obj

    @cache
    def _e(self, state):
        return set(map(self.get_event, self.__gsmp__.e(state)))

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

    def get_current_state(self):
        return self._current_state

    def get_active_events(self, state):
        return self._e(state)

    def get_old_events(self, new_state, trigger_event, old_state):
        return self._e(new_state) & (self._e(old_state) - {trigger_event})

    def get_new_events(self, new_state, trigger_event, old_state):
        return self._e(new_state) - ((self._e(old_state) - {trigger_event}) & self._e(new_state))

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
    _f_0 = None
    _shared_events = None

    def __init__(self, *args, shared_events=None, f=None, r=None, f_0=None):
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
        self.shared_events = [] if shared_events is None else shared_events
        self.f = f
        self.r = r
        self._f_0 = f_0

    @property
    def shared_events(self):
        return self._shared_events

    @shared_events.setter
    def shared_events(self, val):
        seen = []
        for e in chain.from_iterable(val):
            if e not in seen:
                seen.append(e)
            else:
                raise ValueError('Duplicate shared events')
        del seen
        self._shared_events = val

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, func):
        self._f = lambda _s, _e, s, e: func(
            _s, (_e.get_name(), _e.get_default_process()), s, (e.get_name(), e.get_default_process())
        )

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, func):
        self._r = lambda s, e: func(s, (e.get_name(), e.get_default_process()))

    @property
    def f_0(self):
        return self._f_0

    @f_0.setter
    def f_0(self, func):
        self._f_0 = lambda s, e: func(s, (e.get_name(), e.get_default_process()))

    def reset(self):
        # Configure the synchronised events to have hashing equality
        for event_mapping in self.shared_events:
            name, node = event_mapping[0]   # Take default as first one
            e = node.get_event(name)
            for _name, _node in event_mapping[1:]:
                e.add_shared_event(_node, _name)
                # replace instances in _node
                _e = _node.get_event(_name)
                _node.set_event(_name, e)
                del _e
                _node._e.cache_clear()
            e.shared = True

        for node in self.nodes:
            node.reset()

        for event in self.get_active_events(self.get_current_state()):
            # set initial clocks of shared events
            if event.shared and self.f_0 is not None:
                parent = event.get_default_process()
                val = self.f_0(self.get_current_state(), event)
                parent.set_clock(None, event, None, None, f=val)

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
            if event.shared and self.r is not None:
                return event, parent.get_clock(event) / self.r(o, event)
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

            r = self.r(old_state, event) if event.shared and self.r is not None else None
            parent = event.get_default_process()
            i = self.nodes[parent]
            parent.reduce_clock(old_state[i], event, time=time, r=r)

    def set_new_clocks(self, new_state, trigger_event, old_state):
        """
        Set all new clocks
        """
        for new_event in self.get_new_events(new_state, trigger_event, old_state):
            f = self.f(new_state, new_event, old_state, trigger_event) \
                if new_event.shared and self.f is not None else None
            parent = new_event.get_default_process()
            i = self.nodes[parent]
            if new_event.shared and self.f is not None:
                f = self.f(new_state, new_event, old_state, trigger_event)
            elif new_event.shared:
                # When default clock setting is used,
                # New event and trigger event are guaranteed to be shared by at least one process
                # However this process is not guaranteed to be the default for either event
                _parent = trigger_event.get_shared_process(new_event)
                j = self.nodes[_parent]
                f = _parent._f(new_state[j], new_event, old_state[j], trigger_event)

            parent.set_clock(new_state[i], new_event, old_state[i], trigger_event, f=f)

    def __add__(self, other):
        return Compose(*self.nodes, other)


class Simulator:
    def __init__(self, gsmp: SimulationObject):
        self.g = gsmp

    def run(self, until=float('inf'), epochs=float('inf'),
            warmup_epochs=0, warmup_until=0,
            estimate_probability=False,
            plugin=None):
        """
        Run a new simulation, generates sample paths through the GSMP state spaces
        :param until: simulation runtime
        :param epochs: number of event firings
        :param warmup_epochs: 'warmup' epochs
        :param warmup_until: 'warmup' time
        :param estimate_probability: boolean for returning state probability distribution
        :param plugin: function pointer for parsing live event transitions
        :return: [(observed state, probability)]
        """
        self.g.reset()
        self._run(epochs=warmup_epochs)
        self._run(until=warmup_until)
        return self._run(until=until, epochs=epochs,
                         estimate_probability=estimate_probability, _plugin=plugin)

    def _run(self, until=float('inf'), epochs=float('inf'),
             estimate_probability=False, _plugin=None):
        if estimate_probability:
            state_holding_times = {}
        total_time = 0
        while epochs > 0 and total_time < until:
            old_state, trigger_event, new_state, time_delta = self._generate_path()

            # Increment timing metrics
            if estimate_probability:
                if old_state not in state_holding_times:
                    state_holding_times[old_state] = 0
                state_holding_times[old_state] += time_delta
            total_time += time_delta
            if _plugin is not None:
                # send transition data to plugin
                _plugin({
                    'old state': old_state,
                    'new state': new_state,
                    'event': trigger_event.get_name(),
                    'process': trigger_event.get_default_process(),
                    'time': total_time,
                    'time delta': time_delta
                })

            epochs -= 1
        if estimate_probability:
            return [(state, time / total_time) for state, time in state_holding_times.items()]
        return

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
