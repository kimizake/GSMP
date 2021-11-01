from collections import deque

arrivals = deque()
response = deque()
waiting = deque()
service_times = deque()


def plugin(data):
    t = data["time"]
    e = data["event"]

    if e == "arr":
        arrivals.append(t)
        if data["old state"] == 0:
            service_times.append(t)
    else:
        response.append(t - arrivals.popleft())
        waiting.append(response[-1] - (t - service_times.popleft()))
        if data["new state"] > 0:
            service_times.append(t)
