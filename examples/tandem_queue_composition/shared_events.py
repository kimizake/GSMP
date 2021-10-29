from queues import queue1, queue2, queue3

shared_events = [
    [("com", queue1), ("arr", queue2)],
    [("arr", queue3), ("com", queue2)]
]
