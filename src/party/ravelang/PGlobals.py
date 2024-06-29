from src.party.ravelang.PNode import PNode

class PGlobals(PNode):
    def __init__(self, prefix: str = "", suffix: str = "", **kwargs: object):
        super(PGlobals, self).__init__([x for x in kwargs.values()], prefix=prefix, suffix=suffix)

        for k, v in kwargs.items():
            globals()[k] = v
        pass
