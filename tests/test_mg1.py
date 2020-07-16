import pytest
import numpy as np
from src.gsmp import GsmpSimulation, GsmpSpec
from src.mg1_example import Mg1


@pytest.fixture(scope="function", params=[
    {'arr': np.random.exponential, 'com': np.random.exponential}
])
def gsmp_spec(request):
    with [5, 10, 20] as n:
        from src.gsmp import Event, State
        es = [Event('arr'), Event('com')]
        ss = [State(i) for i in range(n + 1)]
        yield Mg1(ss, es, request.param)


@pytest.mark.parametrize("arrival_rate", [1])
@pytest.mark.parametrize("service_rate", [2])
@pytest.mark.parametrize("state_transitions", [3000])
def test_mean_queue_length(gsmp_spec: GsmpSpec, arrival_rate: float, service_rate: float, state_transitions: int):
    simulation = GsmpSimulation(gsmp_spec)
    simulation_time = simulation.simulate(state_transitions)

    from functools import reduce
    average_queue_length = reduce(
        lambda x, y: x + y,
        [state.label for state in gsmp_spec.states]
    ) / simulation_time

    expected_queue_length = pk_mean_queue_length(arrival_rate, service_rate, gsmp_spec.distribution)

    assert average_queue_length == pytest.approx(expected_queue_length, abs=1e-1)


def pk_mean_queue_length(arrival_rate: float, service_rate: float, distribution: dict) -> float:
    utilization = arrival_rate * service_rate
    service_time_distribution = distribution['com']
    return utilization + (utilization ** 2 + arrival_rate ** 2 * var(service_time_distribution)) / 2 / (1 - utilization)