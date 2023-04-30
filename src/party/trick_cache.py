class TrickCache:
    """
    A cache for trick functions
    Values are indexed by the params passed to the function
    The cache automatically removes old values after a configurable number of cached values

    cache.key: set the current params to index the cache
    cache.new: check if the current params are new
    cache.new(value): set the current params to the given value
    cache.get: get the current params value
    """

    def __init__(self, max_size=10):
        self.max_size = max_size
        self.cache = {}
        self.list = []

    def key(self, *args):
        self.current_key = args

    def new(self, value=None):
        if value is None:
            # Check if the value exists
            return self.current_key not in self.cache
        else:
            # Assign the value
            self.cache[self.current_key] = value
            self.list.append(self.current_key)

            while len(self.cache) > self.max_size:
                del self.cache[self.list.pop(0)]

            return value

    def get(self):
        return self.cache[self.current_key]
