import uuid

from src.lib import corelib


class Keyframe:
    def __init__(self, node):
        self.node = node
        self.t = 0
        self.w = 0
        self.prop = 0
        self.lerp = 0

    def __str__(self):
        return f"Keyframe({self.t:.02f}, {self.w:.02f}, {self.prop:.02f}, {self.lerp:.02f})"

    def __repr__(self):
        return self.__str__()

class Keyframes:
    def __init__(self):
        self.uuid = corelib.make_short_guid()
        self.times = []
        self.columns = []

    def get_next_t(self, default_t=0):
        return self.times[-1] if self.times else default_t

    def append_column(self, t, w):
        self.times.append(t)
        self.columns.append(w)

    def get_keyframe(self, t, child_index):
        return self.columns[t][child_index]

    def __len__(self):
        return len(self.times)

    def __getitem__(self, index):
        return self.columns[index]

    def __iter__(self):
        return zip(self.times, self.columns)

    def is_empty(self):
        return len(self.times) == 0

    def __str__(self):
        return f"Keyframes({self.uuid}, {len(self.times)} frames, {len(self.columns)} columns)"
