from abc import ABCMeta, abstractmethod
import numpy as np
from bitmap import BitMap


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
        return str('event ' + self.name)


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


class SimulationObject(metaclass=ABCMeta):

    @classmethod
    def __subclasscheck__(cls, subclass):
        return (
            hasattr(subclass, 'get_current_state') and callable(subclass.get_current_state) and
            hasattr(subclass, 'get_active_events') and callable(subclass.get_active_events) and
            hasattr(subclass, 'choose_winning_event') and callable(subclass.choose_winning_event) and
            hasattr(subclass, 'update_state_time') and callable(subclass.update_state_time) and
            hasattr(subclass, 'get_new_state') and callable(subclass.get_new_state) and
            hasattr(subclass, 'set_current_state') and callable(subclass.set_current_state) and
            hasattr(subclass, 'set_old_clocks') and callable(subclass.set_old_clocks) and
            hasattr(subclass, 'set_new_clocks') and callable(subclass.set_new_clocks) or
            NotImplementedError
        )

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
    def update_state_time():
        raise NotImplementedError


class GsmpWrapper(SimulationObject):

    __gsmp__ = None

    def __init__(self, obj):
        # TODO: add a save feature after pre-processing

        if not isinstance(obj, Gsmp):
            raise TypeError('wrapped object must be of type %s' % Gsmp)
        self.__gsmp__ = obj

        # Construct state and event objects
        self.__states = [State(index, name) for index, name in enumerate(self.__gsmp__.states())]
        self.__events = [Event(self, name) for name in self.__gsmp__.events()]

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
        return np.random.choice(
            self.current_state.adjacent_nodes,
            p=[self._p(_s, o, e) for _s in self.current_state.adjacent_nodes]
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
        e_prime = new_state.events
        old_events = e & e_prime
        new_events = e_prime ^ old_events
        self.old_events = old_events
        self.cancelled_events = e ^ old_events
        self.new_events = new_events
        self.active_events = old_events | new_events
        self.current_state = new_state

    def set_old_clocks(self, s, t):
        for _e in self.bitmap.get(self.old_events):
            _e.tick_down(t * self._r(s, _e))

    def set_new_clocks(self, s, e):
        _s = self.current_state
        for _e in self.bitmap.get(self.new_events):
            _e.set_clock(
                self._f_0(_e, _s) if _s == self.initial_state else
                self._f(_s, _e, s, e)
            )

    @staticmethod
    def update_state_time(s, t):
        s.time_spent += t


class Gsmp(GsmpWrapper, metaclass=ABCMeta):

    def __init__(self):
        super().__init__(self)

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
        """"""
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

    def __init__(self, *args, synchros=None, p=None, f=None, r=None):
        self.nodes = args

        # Support for custom functions not yet added
        self.p, self.f, self.r = p, f, r

        def get_event_from_tuple(name, node):
            return {event.get_name(): event for event in node.get_events()}[name]

        # Configure the synchronised events to have hashing equality
        for synchro in synchros:
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

        self.shared_events = {get_event_from_tuple(*ss[0]): ss for ss in synchros}

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

        # Field used for output
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
        from itertools import chain
        active_events = list(chain.from_iterable(node.get_active_events() for node in self.nodes))
        from collections import Counter
        event_counter = Counter(active_events)

        def determine(e):
            if e.shared:
                return len(self.shared_events[e]) == event_counter[e]
            return True
        active_events[:] = filter(determine, active_events)     # Filter out inactive shared events
        del event_counter
        return set(active_events)   # Remove duplicates

    def get_new_state(self, o, e):
        """
        Given a list of old states and a singular event,
        go to the active nodes and call new state
        Return a list of just the updated states
        """
        def compose_new_state_vector(arg):
            index, event = arg
            if e.shared and self.p:
                # compute all possible adjacent vector states
                from itertools import product
                adjacent_states = product(*(s_i.adjacent_nodes for s_i in o))
                return np.random.choice(
                    adjacent_states,
                    p=[self.p(_s, o, e) for _s in adjacent_states]
                )
            return self.nodes[index].get_new_state(o[index], event)

        return list(map(compose_new_state_vector, self.find_gsmp[e]))

    def choose_winning_event(self, o, es):
        """
        Use the active event data structure to pick the winner
        :param o: list of current states in each node
        :param es: set of active events
        :return: winning, time passed
        """

        # Need to process es to extract index info
        _es = {}
        for e in es:
            i, _ = self.find_gsmp[e][0]
            if i not in _es:
                _es[i] = [e]
            else:
                _es[i].append(e)

        # Event handling for custom rate functions
        # Not yet implemented
        # def handle_shared_events(...):
        #     return

        ttl = dict(self.nodes[index].choose_winning_event(o[index], events) for index, events in _es.items())
        winner = min(ttl, key=ttl.get)
        return winner, ttl[winner]

    def set_current_state(self, s, e):
        """
            Given a list of updated states, we now need to enumerate to update nodes.
        """
        for i, (index, event) in enumerate(self.find_gsmp[e]):
            self.nodes[index].set_current_state(s[i], event)

    def set_old_clocks(self, s, t):
        """
            s is a list of old states, which will be a 1-1 mapping with our gsmp nodes
        """
        for i, node in enumerate(self.nodes):
            node.set_old_clocks(s[i], t)

    def set_new_clocks(self, s, e):
        """
            Given the winning event a set of old states (1-1 map with gsmp nodes)
            go to relevant nodes and update event clocks
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
    def __init__(self, gsmp: SimulationObject):
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
            self.g.set_old_clocks(old_state, time_elapsed)
            self.g.set_new_clocks(old_state, winning_event)

            # print("event {0} fired at time {1} ------- new state {2}".format(winning_event, self.total_time, self.g.get_current_state()))

            epochs -= 1
        # return map(lambda s: s.time_spent / self.total_time, self.g.get_states())
