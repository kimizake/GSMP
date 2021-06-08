from core import Simulator
from collections import deque
from mm1 import MM1

arrival = 1
time = 1000


def get_results(util):
    service = arrival / util

    queue = MM1(
        adjacent_states=lambda s: [1] if s == 0 else [s - 1, s + 1],
        service_rate=service, arrival_rate=arrival
    )

    arrivals = deque()
    response = deque()
    waiting = deque()
    service_t = deque()

    def stream(data):
        t = data['time']
        e = data['event']

        if e == 'arr':
            arrivals.append(t)
            if data['old state'] == 0:
                service_t.append(t)
        else:
            response.append(t - arrivals.popleft())
            waiting.append(response[-1] - (t - service_t.popleft()))
            if data['new state'] > 0:
                service_t.append(t)

    from time import perf_counter

    start = perf_counter()
    Simulator(queue).run(until=time, plugin=stream)
    end = perf_counter()

    print('done')
    return response, waiting, end - start


if __name__ == "__main__":
    import numpy as np
    import scipy.stats as st

    utils = np.linspace(0.1, 0.9, num=9)
    alpha = 0.95

    runtimes = []

    rexact = []
    restimate = []
    rci = []
    ra = []

    wexact = []
    westimate = []
    wci = []
    wa = []

    for util in utils:
        responses, waitings, runtime = get_results(util)
        runtimes.append(runtime)

        mr = np.mean(responses)
        restimate.append(mr)
        mw = np.mean(waitings)
        westimate.append(mw)

        cr = st.norm.interval(alpha=alpha, loc=mr, scale=st.sem(responses))
        rci.append(cr)
        cw = st.norm.interval(alpha=alpha, loc=mw, scale=st.sem(waitings))
        wci.append(cw)

        hr = cr[1] - mr
        hw = cw[1] - mw

        er = 1 / (arrival / util - arrival)
        rexact.append(er)
        ew = util / (arrival / util - arrival)
        wexact.append(ew)

        ar = 100 * hr / mr
        ra.append(ar)
        aw = 100 * hw / mw
        wa.append(aw)

    import pandas as pd
    response_df = pd.DataFrame(data={
        r'$\rho$': utils,
        'exact': rexact,
        'estimate': restimate,
        '{}% confidence interval'.format(100 * alpha): rci,
        'accuracy (%)': ra,
        'runtime': runtimes
    })

    response_df.to_csv('mm1 results/response times MM1 {}.csv'.format(time))

    waiting_df = pd.DataFrame(data={
        r'$\rho$': utils,
        'exact': wexact,
        'estimate': westimate,
        '{}% confidence interval'.format(100 * alpha): wci,
        'accuracy (%)': wa,
        'runtime': runtimes
    })

    waiting_df.to_csv('mm1 results/waiting times MM1 {}.csv'.format(time))

    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    ax1.fill_between(utils, [x[0] for x in rci], [x[1] for x in rci], color='b', alpha=.5, label='actual')
    ax1.plot(utils, rexact, 'r--', label='expected')
    ax1.set_xlabel('utilisation ' + r'$\rho$')
    ax1.set_ylabel('mean response time')
    ax1.set_title('M/M/1 response time with simulation time {}'.format(time))
    ax1.legend()

    ax2.fill_between(utils, [x[0] for x in wci], [x[1] for x in wci], color='b', alpha=.5, label='actual')
    ax2.plot(utils, wexact, 'r--', label='expected')
    ax2.set_xlabel('utilisation ' + r'$\rho$')
    ax2.set_ylabel('mean waiting time')
    ax2.set_title('M/M/1 waiting time with simulation time {}'.format(time))
    ax2.legend()

    fig1.savefig('mm1 results/mm1 mean response time {}.png'.format(time))
    fig2.savefig('mm1 results/mm1 mean waiting time {}.png'.format(time))
    plt.show()
