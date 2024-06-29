from typing_extensions import override

from src.lib.loglib import trace_decorator
from src.party.maths import lerp, rng, val_or_range
from src.party.ravelang.PList import PList
from src.renderer import rv


class PSeq(PList):
    def __init__(self, children, width=10, lerp=0.25, scale=1, add=0, prefix='', suffix=''):
        super(PSeq, self).__init__(children, scale=scale, add=add, prefix=prefix, suffix=suffix)
        self.children = children

        self.width = width
        self.lerp = lerp

        if len(prefix) > 0: prefix = prefix + ' '
        if len(suffix) > 0: suffix = ' ' + suffix

    def can_promote(self):
        return False

    def get_debug_string(self,
                         t_length=10,
                         t_timestep=0.5):
        return f"PSeq(children={len(self.children)})"

    @override
    @trace_decorator
    def bake(self):
        super().bake()

        # CREATE THE KEYFRAMES BY STATE -------------------------------------------

        period_f = self.width  # Current period_f
        index = -1  # Current prompt index
        end = 0  # Current prompt end

        for i in range(rv.n):
            # Advance the interpolations
            for j, node in enumerate(self.children):
                if i == end and j == index:  # End of the current scene, tween out
                    node.min = i
                    node.max = i + int(period_f * val_or_range(self.lerp))
                    node.a = node.w
                    node.b = 0
                    node.k = rng(0.4, 0.6)

                # If we're still interpolating
                if i <= node.max and node.max > 0:
                    t = (i - node.min) / (node.max - node.min)
                    w = lerp(node.a, node.b, t)

                    node.w = w
                    node.timeline.append(w)
                else:
                    node.timeline.append(node.w)  # Copy last w

            # Advance the scene, tween in
            if i == end:
                period_f = int(val_or_range(self.width) * rv.fps)

                index = (index + 1) % len(self.children)
                end += period_f

                node = self.children[index]
                node.min = i
                node.max = i + int(period_f * val_or_range(self.lerp))
                node.a = node.w
                node.b = 1
                node.k = rng(0.4, 0.6)

    def is_sorted_print(self):
        return True
