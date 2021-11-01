from core import Compose
from queues import queue1, queue2, queue3

from shared_events import shared_events

from clock_functions import f, r, f_0

tandem_queue = Compose(
    queue1, queue2, queue3,
    shared_events=shared_events,
    f=f, r=r, f_0=f_0
)
