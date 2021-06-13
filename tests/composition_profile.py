from core import Simulator
from examples.mm1 import MM1
from functools import reduce
from operator import add

if __name__ == "__main__":
    components = 100

    queues = [MM1() for i in range(components)]

    q = reduce(add, queues)
    Simulator(q).run(until=1000)
    # Simulator(MM1()).run(until=1000)
