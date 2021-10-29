from queues import queue1, queue2, queue3

from shared_events import shared_events

from clock_functions import f, r, f_0

tandem_queue = queue1 + queue2 + queue3
tandem_queue.shared_events = shared_events
tandem_queue.f = f
tandem_queue.r = r
tandem_queue.f_0 = f_0

if __name__ == "__main__":
    from core import Simulator
    Simulator(tandem_queue).run(until=1000)