from random import shuffle
from typing import List

import numpy as np
from typing_extensions import override

from src.lib.loglib import trace
from src.party.maths import ilerp, lerp, max, val_or_range
from src.party.ravelang.Keyframe import Keyframe
from src.party.ravelang.PList import PList
from src.renderer import rv


class PProp(PList):
    def __init__(self,
                 children,
                 width: float | tuple[float, float] = None,
                 p: float | tuple[float, float] = None,
                 drift: float | tuple[float, float] = None,
                 lerp: float | tuple[float, float] = 0.95,
                 curve=None,
                 prefix: str = "",
                 suffix: str = ""):
        super(PProp, self).__init__(children, prefix=prefix, suffix=suffix)

        if drift is None: drift = [0, 0.25]
        if p is None: p = [0.75, 0.95]
        if width is None: width = [15, 20]

        self.width = width
        self.p = p
        self.drift = drift
        self.lerp = 20
        self.curve = curve

    @override
    def update_state(self):
        for i, n in enumerate(self.children):
            n.bake_index = i  # No idea why it's resetting to -1.

            if self.is_past_last_keyframe(rv.t):
                self.append_keyframe_column(self.keyframes, rv.t)

            kf = self.get_current_keyframe(n, rv.t)
            if n.last_target_kf != kf:
                n.movement_start = n.w

            w = n.w
            w1 = n.movement_start
            w2 = kf.w
            range = abs(w2 - w1)
            dist = w2 - w
            progress = 1 - (abs(dist) / range)
            if self.curve is not None:
                progress = self.curve(progress)

            are_equal = abs(dist) < 0.05
            if not are_equal:
                # Move towards target taking into account deltaTime (rv.dt) and progress along curve
                w += dist * rv.dt * (1 - progress) * self.lerp

                # Handle overshoot
                if dist > 0 and w > w2:
                    w = w2
                elif dist < 0 and w < w2:
                    w = w2

            n.w = w
            n.last_target_w = w2
            n.last_target_kf = kf

        super().update_state()

    @override
    # @trace_decorator
    def bake(self):
        super().bake()

        keyframes = self.get_keyframes()
        len_keyframes = len(keyframes)

        with trace(f"PProp.bake.bake(blending {len_keyframes} states for frames {self.get_timeline_sector_start()}:{self.get_timeline_sector_end()} sectored by [{'self' if self.get_timeline_sector_node() == self else self.get_timeline_sector_node()}])"):
            for child in self.children:
                for kf_index in range(len_keyframes):
                    index = child.bake_index
                    timeline = child.timeline

                    if kf_index == len_keyframes - 1:
                        # Last state, copy the last value for the rest of the timeline
                        kf_index = keyframes[kf_index][index]
                        timeline[kf_index.t:] = kf_index.w
                        break

                    kf0: Keyframe = keyframes[kf_index][index]
                    kf1: Keyframe = keyframes[kf_index + 1][index]

                    segment_duration = kf1.t - kf0.t

                    # Interpolation frames (100% interpolation otherwise with just np.time_f)
                    time_range = np.arange(
                        np.floor(kf0.t * rv.fps),
                        np.ceil(kf1.t * rv.fps),
                        1,
                        dtype=int)

                    # Rasterize the time range
                    time_range = np.clip(time_range, kf0.t, kf1.t - 1)

                    ts = ilerp(
                        kf1.t - segment_duration * val_or_range(self.lerp),
                        kf1.t,
                        time_range)
                    ws = (self.curve or lerp)(kf0.w, kf1.w, ts)

                    timeline[kf0.t:kf1.t] = ws

    @override
    def append_keyframe_column(self, keyframes, target_time=0):
        width = val_or_range(self.width)

        # Shuffle children
        children = list()
        for i, node in enumerate(self.children):
            node.bake_index = i
            children.append(node)
        shuffle(children)

        # Create keyframes
        t = target_time + width
        column: List[Keyframe] = [None] * len(children)
        p = val_or_range(self.p)
        w = 1

        print(f"PProp.append_keyframe_column at {t} for {len(children)} children")
        if p >= 1:
            print("WARNING: bake_proportion is >= 1")

        for node in children:
            kf = Keyframe(node)
            kf.w = w
            kf.t = t + width * val_or_range(self.drift)
            kf.t = min(kf.t, rv.n - 1)
            kf.prop = p
            kf.lerp = val_or_range(self.lerp)

            w *= p
            column[node.bake_index] = kf

        keyframes.append_column(t, column)

    def is_sorted_print(self):
        return True
