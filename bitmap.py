class BitMap:
    def __init__(self, objects):
        self.size = len(objects)
        self.items = objects

    def get(self, binary):
        string = bin(binary)[2:].zfill(self.size)
        assert len(string) == self.size
        return [item for item in self.items if string[self.items.index(item)] == '1']

    def format(self, objects):
        assert set(objects).issubset(self.items)
        return int("".join(['1' if i in objects else '0' for i in self.items]), 2)

    def all(self):
        return (1 << self.size) - 1
