from core import Simulator
from mm1k import MM1k
from mm1_plugin import plugin, response, waiting

if __name__ == "__main__":
    res = Simulator(MM1k()).run(
        until=1000,
        estimate_probability=True,
        plugin=plugin
    )
    print(res, response, waiting)
