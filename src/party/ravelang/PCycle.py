from src.party.ravelang.PList import PList

class PCycle(PList):
    """
    A node which evals to a single child node, based on the time interval (cycling)
    """

    def __init__(self, children=None, interval: float = 1, **kwargs):
        super(PCycle, self).__init__(children, **kwargs)
        self.interval = interval

    def eval_text(self, t):
        loop = int(t / self.interval)
        return self.children[loop % len(self.children)].eval_text(t)
