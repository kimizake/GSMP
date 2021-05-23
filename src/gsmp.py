from abc import ABCMeta, abstractmethod
import numpy as np
from bitmap import BitMap
from functools import cache


class Event:
    def __init__(self, node, name):
        self.name = name
        self.clock = None
        self._frozen = False
        self.shared = False
        self._equal_event = None
        self._node = node

    def get_name(self):
        return self.name

    def set_clock(self, clock):
        if self._frozen:
            return
        self.clock = clock

    def get_clock(self):
        return self.clock if not self._frozen else None

    def set_equal_event(self, obj):
        assert isinstance(obj, Event)
        self._equal_event = obj

    def suspend_clock(self):
        self._frozen = True
        self.clock = None

    def is_frozen(self):
        return self._frozen

    def tick_down(self, time):
        if self._frozen:
            return
        assert self.clock >= time
        self.clock -= time

    def __eq__(self, other):
        return isinstance(other, Event) and (
            (self.name == other.name and self._node == other._node) or (hash(self) == hash(other))
        )

    def __hash__(self):
        return hash(self._equal_event) if self._equal_event else hash((self.name, self._node))

    def __repr__(self):
        return str('event ' + self.name + ' at ' + str(self._node))


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
                hasattr(subclass, 'set_new_clocks') and callable(subclass.set_new_clocks) and
                hasattr(subclass, 'parse_state') and callable(subclass.parse_state) or
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
    def get_active_events(self):
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
    current_state = None
    old_events = None
    new_events = None
    active_events = None

    def __init__(self, obj, adjacent_states=None):
        # TODO: add a save feature after pre-processing

        if not isinstance(obj, Gsmp):
            raise TypeError('wrapped object must be of type %s' % Gsmp)
        self.__gsmp__ = obj

        # Construct state and event objects
        self._events = [Event(self, name) for name in self.__gsmp__.events()]

        # Create event bitmap - the precondition here is that the event set is finite and < bit count
        self.bitmap = BitMap(self._events)

        if adjacent_states is not None:
            self._get_adj_states = adjacent_states
        else:
            # The default get_adjacent_state function has to iterate over the entire state space and event set
            self._get_adj_states = lambda s: {
                _s for _s in self.get_states() for e in self.bitmap.get(self._e(s)) if self._p(_s, e, s) != 0
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

        # set trackers
        self.current_state = initial_state
        # set initial clocks
        for event in self.get_active_events():
            event.set_clock(self._f_0(initial_state, event))

    @cache
    def _e(self, state):
        return self.bitmap.format([Event(self, name) for name in self.__gsmp__.e(state)])

    def _p(self, next_state, event, old_state):
        return self.__gsmp__.p(next_state, event.get_name(), old_state)

    def _f(self, next_state, new_event, old_state, trigger_event):
        return self.__gsmp__.f(next_state, new_event.get_name(), old_state, trigger_event.get_name())

    def _r(self, state, event):
        return self.__gsmp__.r(state, event.get_name())

    def _s_0(self, state):
        return self.__gsmp__.s_0(state)

    def _f_0(self, state, event):
        return self.__gsmp__.f_0(state, event.get_name())

    def get_states(self):
        return self.__gsmp__.states()

    @cache
    def get_adj_states(self, state):
        return list(self._get_adj_states(state))

    def get_events(self):
        return self._events

    def get_current_state(self):
        return self.current_state

    def get_active_events(self):
        return list(self.bitmap.get(self._e(self.current_state)))

    def get_old_events(self, new_state, trigger_event, old_state):
        e1 = self._e(old_state)
        try:
            e1 -= self.bitmap.positions[trigger_event]
        except KeyError:
            pass
        e2 = self._e(new_state)
        return self.bitmap.get(e1 & e2)

    def get_new_events(self, new_state, trigger_event, old_state):
        e1 = self._e(old_state)
        try:
            e1 -= self.bitmap.positions[trigger_event]
        except KeyError:
            pass
        e2 = self._e(new_state)
        return self.bitmap.get(e2 - (e1 & e2))

    def get_new_state(self, o, e):
        adj_states = self.get_adj_states(o)
        return adj_states[np.random.choice(
            len(adj_states),
            p=[self._p(_s, e, o) for _s in adj_states]
        )]

    def choose_winning_event(self, o, es):
        tmp = list(map(lambda e: e.get_clock() / self._r(o, e), es))
        t = np.amin(tmp)
        event_index = np.where(tmp == t)[0]
        # Usually there is only 1 winning event, but in the case of a tie, randomly select a winner
        winning_events = [es[i] for i in event_index]
        return winning_events[np.random.choice(len(winning_events))], t

    def set_current_state(self, new_state):
        self.current_state = new_state

    def set_old_clocks(self, time, new_state, trigger_event, old_state):
        for old_event in self.get_old_events(new_state, trigger_event, old_state):
            old_event.tick_down(time * self._r(old_state, old_event))

    def set_new_clocks(self, new_state, trigger_event, old_state):
        for new_event in self.get_new_events(new_state, trigger_event, old_state):
            new_event.set_clock(self._f(new_state, new_event, old_state, trigger_event))


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
    shared_events = None
    find_gsmp = None

    def __init__(self, *args, synchros=None, f=None, r=None):
        self.nodes = args

        # Wrap the stochastic functions.
        if f:
            self._f = lambda _s, _e, s, e: f(_s, _e.get_name(), s, e.get_name())
        if r:
            self._r = lambda s, e: r(s, e.get_name())

        self._synchros = synchros

    def reset(self):
        for node in self.nodes:
            node.reset()

        def get_event_from_tuple(_name, _node):
            return {event.get_name(): event for event in _node.get_events()}[_name]

        # Configure the synchronised events to have hashing equality
        for synchro in self._synchros:
            name, node = synchro[0]
            e = get_event_from_tuple(name, node)
            e.shared = True
            for _name, _node in synchro[1:]:
                _e = get_event_from_tuple(_name, _node)
                _e.shared = True
                _e.suspend_clock()
                _e.set_equal_event(e)

        self.shared_events = {get_event_from_tuple(*ss[0]): ss for ss in self._synchros}

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
        # This field holds information about the way events match up to nodes
        # By storing the index instead of a pointer to the node, it makes it easy to work with state vectors.
        self.find_gsmp = find_gsmp

    def get_states(self):
        from itertools import product
        # This takes the cartesian product of the states, therefore it is really slow
        return product(*(node.get_states() for node in self.nodes))

    @staticmethod
    def parse_state(states):
        return tuple(state.get_name() for state in states)

    def get_current_state(self):
        """
        Return current state vector
        """
        return tuple(n.get_current_state() for n in self.nodes)

    def _get_events(self, events):
        from itertools import chain, filterfalse
        events = list(chain.from_iterable(events))    # flatten the list
        from collections import Counter
        event_counter = Counter(events)

        def is_illegal(e):
            return e.shared and (
                    # when the event has a different status across its nodes,
                    len(self.shared_events[e]) != event_counter[e]
                    # or it's been suspended and is therefore a duplicate
                    or e.is_frozen()
            )

        return list(filterfalse(is_illegal, events))    # filter out 'illegal' events

    def get_active_events(self):
        """
        :return: list of all active events
        """
        return self._get_events(node.get_active_events() for node in self.nodes)

    def get_old_events(self, new_state, trigger_event, old_state):
        return self._get_events(node.get_old_events(new_state[i], trigger_event, old_state[i])
                                for i, node in enumerate(self.nodes))

    def get_new_events(self, new_state, trigger_event, old_state):
        return self._get_events(node.get_new_events(new_state[i], trigger_event, old_state[i])
                                for i, node in enumerate(self.nodes))

    def get_new_state(self, o, e):
        """
        return list of new states
        """
        # for event e, find the index which process nodes it belongs to, and its alias within that node
        indices, events = zip(*self.find_gsmp[e])

        def get_new_node_state(i, node):
            if i in indices:                            # when event 'e' is in gsmp 'node'
                event = events[indices.index(i)]
                return node.get_new_state(o[i], event)  # that 'node' will enter a new state
            return node.get_current_state()             # otherwise its state doesn't change
        from itertools import starmap
        new_states = tuple(starmap(get_new_node_state, enumerate(self.nodes)))
        del indices, events
        return new_states

    def choose_winning_event(self, o, es):
        """
        Use the active event data structure to pick the winner
        :param o: list of current states in each node
        :param es: list of active events
        :return: winning, time passed
        """
        def get_time_deltas(event):
            if event.shared and self._r is not None:
                return event.get_clock() / self._r(o, event)
            i, _event = self.find_gsmp[event][0]  # default behavior is to go to the first gsmp node
            return event.get_clock() / self.nodes[i]._r(o[i], _event)

        time_deltas = list(map(get_time_deltas, es))  # calculate how long each active event spends in current state
        winning_time = np.amin(time_deltas)  # take the minimum time
        winning_indexes = np.where(time_deltas == winning_time)[0]  # In case multiple events 'win'
        winning_events = [es[i] for i in winning_indexes]  # Randomly select one of those events
        return winning_events[np.random.choice(len(winning_events))], winning_time  # Typically only one event wins.

    def set_current_state(self, s):
        """
        set current state to s, and update all event status trackers
        """
        for i, node in enumerate(self.nodes):
            node.set_current_state(s[i])

    def set_old_clocks(self, time, new_state, trigger_event, old_state):
        """
        iterate over all the old events in each node, and update clocks
        """
        for event in self.get_old_events(new_state, trigger_event, old_state):
            if event.shared and self._r is not None:
                time_delta = time * self._r(old_state, event)
            else:
                i, _event = self.find_gsmp[event][0]
                time_delta = time * self.nodes[i]._r(old_state[i], _event)
            event.tick_down(time_delta)

    def set_new_clocks(self, new_state, trigger_event, old_state):
        """
        iterate over all the new events in each node, and set new clocks
        """
        for i, _e in self.find_gsmp[trigger_event]:
            node = self.nodes[i]
            for new_event in node.get_new_events(new_state[i], trigger_event, old_state[i]):
                if new_event.shared and self._f is not None:
                    new_event.set_clock(self._f(
                        new_state, new_event,
                        old_state, trigger_event
                    ))
                else:
                    new_event.set_clock(node._f(
                        new_state[i], new_event,
                        old_state[i], _e
                    ))


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
        active_events = self.g.get_active_events()
        # Select trigger event and find time from last event
        trigger_event, time_delta = self.g.choose_winning_event(old_state, active_events)
        # Select new state
        new_state = self.g.get_new_state(old_state, trigger_event)
        # Update clocks for next loop
        self.g.set_old_clocks(time_delta, new_state, trigger_event, old_state)
        self.g.set_new_clocks(new_state, trigger_event, old_state)

        self.g.set_current_state(new_state)

        return old_state, trigger_event, new_state, time_delta
