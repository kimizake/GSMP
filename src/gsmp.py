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

    def tick_down(self, time):
        if self._frozen:
            return
        assert self.clock >= time
        self.clock -= time

    def __eq__(self, other):
        return isinstance(other, Event) and hash(self) == hash(other)

    def __hash__(self):
        return hash(self._equal_event) if self._equal_event else hash((self.name, self._node))

    def __repr__(self):
        return str('event ' + self.name + ' at ' + str(self._node))


class State:
    def __init__(self, node, value):
        self._node = node
        self._val = value
        self.events = None

    def get_name(self):
        return self._val

    def set_events(self, events):
        self.events = events

    def __eq__(self, other):
        return isinstance(other, State) and self._node == other._node and self._val == other._val

    def __hash__(self):
        return hash(self._val)

    def __repr__(self):
        return str(self._val)


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
    def set_current_state(self, n, e):
        raise NotImplementedError

    @abstractmethod
    def set_old_clocks(self, o, t):
        raise NotImplementedError

    @abstractmethod
    def set_new_clocks(self, o, e):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def parse_state(state):
        raise NotImplementedError


class GsmpWrapper(SimulationObject):
    """
    Proxy class to add simulation function to user defined GSMP
    """
    __gsmp__ = None
    _states = None
    _events = None
    current_state = None
    old_events = None
    new_events = None
    active_events = None

    def __init__(self, obj, infinite=False, adjacent_states=None):
        # TODO: add a save feature after pre-processing

        if not isinstance(obj, Gsmp):
            raise TypeError('wrapped object must be of type %s' % Gsmp)
        self.__gsmp__ = obj

        self._infinite = infinite

        # Construct state and event objects
        self._events = [Event(self, name) for name in self.__gsmp__.events()]

        if adjacent_states is not None:
            self._get_adj_states = lambda s: (
                State(self.__gsmp__, val) for val in adjacent_states(*GsmpWrapper.get_names(s))
            )
        else:
            # The default get_adjacent_state function has to iterate over the entire state space and event set
            self._get_adj_states = lambda s: \
                {_s if self._p(_s, s, e) != 0 else None for _s in self.get_states() for e in self._e(s)} - {None}

        # Create event bitmap - the precondition here is that the event set is finite and < bit count
        self.bitmap = BitMap(self._events)

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
        initial_state = np.random.choice(
            states,
            p=probabilities
        )
        del states, probabilities

        if self._infinite:
            self._states = [initial_state]
        else:
            # Define states
            self._states = [State(self.__gsmp__, val) for val in self.__gsmp__.states()]

            # Replace initial state with the original instance
            i = self._states.index(initial_state)
            self._states.pop(i)
            self._states.insert(i, initial_state)

            # dfs prune states
            visited = {s: False for s in self._states}

            def visit(s):
                if not visited[s]:
                    visited[s] = True
                    for _s in self.get_adj_states(s):
                        visit(_s)

            visit(initial_state)
            for state in self._states:
                if not visited[state]:
                    self._states.remove(state)
                    del state
            del visited

        # bitmap config
        for state in self._states:
            state.set_events(self.bitmap.format(self._e(state)))

        # set trackers
        self.current_state = initial_state
        self.old_events = 0
        self.new_events = self.current_state.events
        self.active_events = self.new_events
        # set initial clocks
        for event in self.get_new_events():
            event.set_clock(self._f_0(event, initial_state))

    # Define wrapper functions
    @staticmethod
    def get_names(*args):
        def get_name(param):
            return param.get_name()

        return tuple(map(get_name, args))

    def _e(self, *args):
        return [Event(self, name) for name in self.__gsmp__.e(*Gsmp.get_names(*args))]

    def _p(self, *args):
        return self.__gsmp__.p(*GsmpWrapper.get_names(*args))

    def _f(self, *args):
        return self.__gsmp__.f(*GsmpWrapper.get_names(*args))

    def _r(self, *args):
        return self.__gsmp__.r(*GsmpWrapper.get_names(*args))

    def _s_0(self, *args):
        return self.__gsmp__.s_0(*GsmpWrapper.get_names(*args))

    def _f_0(self, *args):
        return self.__gsmp__.f_0(*GsmpWrapper.get_names(*args))

    def get_states(self):
        return self._states if self._states else (State(self.__gsmp__, val) for val in self.__gsmp__.states())

    @staticmethod
    def parse_state(state):
        return state.get_name()

    @cache
    def get_adj_states(self, state):
        def clean(s):
            try:
                return self._states[self._states.index(s)]  # Get the original pointer to the state object
            except ValueError:  # State hasn't been created yet
                self._states.append(s)
                s.set_events(self.bitmap.format(self._e(s)))
                return s

        # Want to 'clean' the output from the internal _get_adj_states function
        return list(map(clean, self._get_adj_states(state)))

    def get_events(self):
        return self._events

    def get_current_state(self):
        return self.current_state

    def get_active_events(self):
        return list(self.bitmap.get(self.active_events))

    def get_old_events(self):
        return self.bitmap.get(self.old_events)

    def get_new_events(self):
        return self.bitmap.get(self.new_events)

    def get_new_state(self, o, e):
        adj_states = self.get_adj_states(self.get_current_state())
        return np.random.choice(
            adj_states,
            p=[self._p(_s, o, e) for _s in adj_states]
        )

    def choose_winning_event(self, o, es):
        tmp = []
        for e in es:
            try:
                tmp.append(e.get_clock() / self._r(o, e))
            except ZeroDivisionError:
                pass
            except ValueError:
                pass
        t = np.amin(tmp)
        event_index = np.where(tmp == t)[0]
        # Usually there is only 1 winning event, but in the case of a tie, randomly select a winner
        winning_events = [es[i] for i in event_index]
        return np.random.choice(winning_events), t

    def set_current_state(self, new_state, winning_event):
        e = self.current_state.events - self.bitmap.positions[winning_event]
        e_prime = new_state.events  # e_prime = self._e(new_state)
        old_events = e & e_prime
        new_events = e_prime - old_events
        self.old_events = old_events
        self.new_events = new_events
        self.active_events = old_events | new_events
        self.current_state = new_state

    def set_old_clocks(self, s, t):
        for _e in self.get_old_events():
            _e.tick_down(t * self._r(s, _e))

    def set_new_clocks(self, s, e):
        _s = self.current_state  # _s is the state we are going into
        # it has already been set as the current state
        for _e in self.get_new_events():
            _e.set_clock(self._f(_s, _e, s, e))


class Gsmp(GsmpWrapper, metaclass=ABCMeta):

    def __init__(self, infinite=False, adjacent_states=None):
        # Infinite state spaces must be specified
        if infinite and not adjacent_states:
            raise NotImplementedError('Need to define adjacent states function')
        super().__init__(self, infinite=infinite, adjacent_states=adjacent_states)

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
    def p(self, _s, s, e):
        """
        Returns the probability of the next state being _s when event e occurs in state s
        :param _s: next state identifier
        :param s: current state identifier
        :param e: event identifier
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
    def f_0(self, e, s):
        """
        Returns the distribution function to set the clock of event e in initial state s
        :param e: event identifier
        :param s: state identifier
        :return: float
        """
        raise NotImplementedError


class Compose(SimulationObject):
    """
    Normal event handling can be deferred to sub-process with relevant vector components
    However synchronised events may have more complicated setting functions,
    therefore the user has the choice of yielding setter functions for probability, clock distributions and rates.
    """

    _p = None
    _f = None
    _r = None
    shared_events = None
    find_gsmp = None

    def __init__(self, *args, synchros=None, p=None, f=None, r=None):
        self.nodes = args

        # Wrap the stochastic functions.
        def func(state_vector):
            return [state.get_name() for state in state_vector]
        if p:
            self._p = lambda _s, e, s: p(func(_s), e.get_name(), func(s))
        if f:
            self._f = lambda _s, _e, s, e: f(func(_s), _e.get_name(), func(s), e.get_name())
        if r:
            self._r = lambda s, e: r(func(s), e.get_name())

        self._synchros = synchros

    def reset(self):
        for node in self.nodes:
            node.reset()

        def get_event_from_tuple(name, node):
            return {event.get_name(): event for event in node.get_events()}[name]

        # Configure the synchronised events to have hashing equality
        for synchro in self._synchros:
            name, node = synchro[0]
            e = get_event_from_tuple(name, node)
            e.shared = True
            for _name, _node in synchro[1:]:
                _e = get_event_from_tuple(_name, _node)
                _e.shared = True
                _e.suspend_clock()
                # The bitmap info needs to be updated because the hash of _e has changed
                v = _node.bitmap.positions.pop(_e)
                _e.set_equal_event(e)
                _node.bitmap.positions[_e] = v

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
        # return list(chain.from_iterable(node.get_states() for node in self.nodes))

    @staticmethod
    def parse_state(states):
        return tuple(state.get_name() for state in states)

    def get_current_state(self):
        """
        Return current state vector
        """
        return tuple(n.get_current_state() for n in self.nodes)

    def _get_events(self, events):
        from itertools import chain
        total_events = list(chain.from_iterable(events))
        from collections import Counter
        event_counter = Counter(total_events)

        def determine(e):
            # if e.shared:
            #     return len(self.shared_events[e]) == event_counter[e]
            # return True
            return not e.shared or len(self.shared_events[e]) == event_counter[e]

        total_events[:] = filter(determine, total_events)  # Filter out inactive shared events
        del event_counter
        return list(set(total_events))  # Remove duplicates

    def get_active_events(self):
        """
        :return: list of all active events
        """
        return self._get_events(node.get_active_events() for node in self.nodes)

    def get_old_events(self):
        return self._get_events(node.get_old_events() for node in self.nodes)

    def get_new_events(self):
        return self._get_events(node.get_new_events() for node in self.nodes)

    @cache
    def get_adj_states(self, state):
        from itertools import product
        return list(product(*(
            self.nodes[i].get_adj_states(s_i) for i, s_i in enumerate(state)
        )))

    def get_new_state(self, o, e):
        """
        Given a list of old states and a singular event,
        go to the active nodes and call new state
        Return a list of just the updated states
        """
        def compose_new_state_vector(arg):
            index, event = arg
            if e.shared and self._p is not None:
                # compute all possible adjacent vector states
                adj_states = self.get_adj_states(o)
                return np.random.choice(
                    adj_states,
                    p=[self._p(_s, o, e) for _s in adj_states]
                )
            return self.nodes[index].get_new_state(o[index], event)

        return list(map(compose_new_state_vector, self.find_gsmp[e]))

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
        return np.random.choice(winning_events), winning_time  # Typically only one event wins.

    def set_current_state(self, s, e):
        """
            Given a list of updated states, we now need to enumerate to update nodes.
        """
        for i, (index, event) in enumerate(self.find_gsmp[e]):
            self.nodes[index].set_current_state(s[i], event)

    def set_old_clocks(self, s, t):
        """
        iterate over all the old events in each node, and update clocks
        """
        for event in self.get_old_events():
            if event.shared and self._r is not None:
                time_delta = t * self._r(s, event)
            else:
                i, _event = self.find_gsmp[event][0]
                time_delta = t * self.nodes[i]._r(s[i], _event)
            event.tick_down(time_delta)

    def set_new_clocks(self, s, e):
        """
        iterate over all the new events in each node, and set new clocks
        """
        new_state = self.get_current_state()
        for i, _e in self.find_gsmp[e]:
            node = self.nodes[i]
            for new_event in node.get_new_events():
                if new_event.shared and self._f is not None:
                    new_event.set_clock(self._f(
                        new_state, new_event,
                        s, e
                    ))
                else:
                    new_event.set_clock(node._f(
                        new_state[i], new_event,
                        s[i], _e
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
            old_state, new_state, trigger_event, time_delta = self._generate_path()

            # Increment timing metrics
            if old_state not in state_holding_times:
                state_holding_times[old_state] = 0
            state_holding_times[old_state] += time_delta
            total_time += time_delta

            # print("event {0} fired at time {1} ------- new state {2}".format(
            # winning_event, self.total_time, self.g.get_current_state()))

            epochs -= 1
        return (
            list(map(self.g.parse_state, state_holding_times)),     # observed states
            list(state_holding_times.values()),                     # holding times
            total_time                                              # total simulation time
        )

    def _generate_path(self):
        # Get old state and active events
        old_state = self.g.get_current_state()
        active_events = self.g.get_active_events()
        # Select trigger event and find time from last event
        trigger_event, time_delta = self.g.choose_winning_event(old_state, active_events)
        # Select new state
        new_state = self.g.get_new_state(old_state, trigger_event)
        # Update state trackers for next iteration
        self.g.set_current_state(new_state, trigger_event)
        # Update clocks for next loop
        self.g.set_old_clocks(old_state, time_delta)
        self.g.set_new_clocks(old_state, trigger_event)

        return old_state, trigger_event, new_state, time_delta
