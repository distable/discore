from munch import Munch


def Munchr(**kwargs):
    """
    Munchr allows recursive dot notation assignment
    (all parents automatically created)
    e.g.:
    m = Munchr()
    m.a.b.c = 1
    print(m.a.b.c) # 1
    """

    def __getattr__(self, item):
        # Create missing
        if item not in self:
            self[item] = Munchr()
        return self[item]

    ret = Munch(**kwargs)
    ret.__getattr__ = __getattr__

    return ret
