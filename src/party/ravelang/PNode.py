import random
from abc import abstractmethod

from src.lib import corelib
from src.lib.loglib import trace_decorator
from src.party.maths import clamp, lerp
from src.party.ravelang import Keyframe
from src.party.ravelang.Keyframe import Keyframes
from src.renderer import rv

wc_regex = r'(([\w\d\s\-;%^&\./\\\'"]+)(:([\w|\d\/,]*))?(:([\w|\d\/,]+))?)'

# noinspection PyUnresolvedReferences

class PNode:
    """
    The `PNode` class represents a node in a prompt tree structure.
    Each node contains either text or children, forming a hierarchy with parent-child relationships and depth levels.
    It supports dynamic text evaluation based on time, considering joining characters and the number of
    children to join.
    The class employs a weight system and keyframes for time-based operations, allowing for timeline-based
    calculations.
    Keyframe management is included for animation purposes.
    Child sorting based on weights at a given time is supported, with the option to enable
    or disable sorted printing.
    The class also implements container-like behavior for child management, allowing for
    easy indexing and iteration over children.

    Key functionalities include:

    - Text evaluation
    - Weight and keyframe management
    - Timeline-based calculations
    - Child sorting
    - Hierarchical structure manipulation
    - Container-like child access

    """

    def __init__(self,
                 text_or_children=None,
                 mask=None,
                 prefix: str = '',
                 suffix: str = '',
                 join_char: str = '',
                 join_num: int = 1):
        """
        Initialize a PNode instance.

        :param text_or_children: Either a string (text) or a list of child nodes
        :param mask: (Not used in current implementation)
        :param prefix: String to prepend to the node's text
        :param suffix: String to append to the node's text
        :param join_char: Character used to join child nodes' text
        :param join_num: Number of child nodes to join
        """
        self.uuid = self._generate_uuid()
        self.children = []
        self.parent = None
        self.depth = 0

        self.text = text_or_children if isinstance(text_or_children, str) else None
        self.mask = mask
        self.join_char = join_char
        self.join_num = join_num

        # State & Baking attributes
        self.bake_index = -1
        self.keyframes = None
        self.w = 1  # Weight
        self.timeline_start_f = None
        self.timeline_end_f = None
        self.timeline = None
        self.min = 0
        self.max = 0
        self.a = 0
        self.b = 0
        self.movement_start = 0

        self.last_target_kf = None
        self.last_target_w = 0

        if not isinstance(text_or_children, str) and text_or_children is not None:
            self.add_children(text_or_children)

    def _generate_uuid(self):
        """Generate a unique identifier for the node."""
        return corelib.make_short_guid()

    def on_copied(self):
        """Called when the node is copied. Regenerates UUID and creates a shallow copy of children."""
        self.uuid = self._generate_uuid()
        self.children = self.children.copy()

    def init_bake(self):
        """Initialize attributes for the baking process."""
        self.min = 0
        self.max = 0
        self.keyframes = Keyframes()
        self.w = random.random()
        self.a = 0
        self.b = 0

    def get_current_keyframe(self, child, t) -> Keyframe:
        """
        Get the current keyframe for a child node at time t.

        :param child: Child node
        :param t: Current time
        :return: Current Keyframe
        """
        keyframes = self.get_keyframes()
        for kf_t, column in keyframes:
            kf = column[child.bake_index]
            if kf.t > t:
                return kf

        print('PNode.get_current_keyframe: No keyframe found for child. Clamping to last keyframe.')
        return kf

    def is_past_last_keyframe(self, t):
        """Check if the current time is past the last keyframe."""
        return self.get_keyframes().get_next_t() < t

    def append_keyframe_columns(self, t_min, t_max):
        """
        Append keyframe columns between t_min and t_max.

        :param t_min: Minimum time
        :param t_max: Maximum time
        """
        keyframes = self.keyframes
        while keyframes.get_next_t() < t_max:
            self.append_keyframe_column(keyframes, t_min)

    def get_keyframes(self):
        """Get the keyframes for this node."""
        return self.keyframes

    @abstractmethod
    def append_keyframe_column(self, keyframes, t_min):
        """
        Append a keyframe column. (Placeholder for actual implementation)

        :param keyframes: Keyframes object
        :param t_min: Minimum time
        """
        pass

    @trace_decorator
    @abstractmethod
    def bake(self):
        """Bake all child nodes. (Placeholder for actual implementation)"""
        for child in self.children:
            child.bake()

    @abstractmethod
    def can_promote(self):
        """Check if this node can be promoted. (Always returns True)"""
        return True

    @abstractmethod
    def is_sorted_print(self):
        """Check if children should be sorted when printing. (Always returns False)"""
        return False

    @abstractmethod
    def update_state(self):
        """Update the state of all child nodes."""
        for child in self.children:
            child.update_state()

    def add_child(self, node):
        """
        Add a child node to this node.

        :param node: Child node to add
        """
        self.children.append(node)
        node.parent = self
        node.depth = self.depth + 1

    def add_children(self, add):
        """
        Add multiple children to this node.

        :param add: List, tuple, or PNode containing children to add
        """
        if isinstance(add, (list, tuple)):
            for child in add:
                self.add_child(child)
        elif isinstance(add, PNode):
            for child in add.children:
                self.add_child(child)
        else:
            raise RuntimeError(f"PNode.add_children: Invalid input children nodes type {type(add)}")

    @trace_decorator
    def eval_text(self, t):
        """
        Evaluate the text of this node at time t.

        :param t: Current time
        :return: Evaluated text
        """
        if self.children:
            ordered_children = self.get_top_children(t) if self.is_sorted_print() else self.children
            return self.join_text_children(t, ordered_children)

        if isinstance(self.text, str):
            return self.text

        raise RuntimeError(f"Invalid PNode text: {self.text}")

    def get_top_children(self, t):
        """
        Get top children sorted by weight at time t.

        :param t: Current time
        :return: Sorted list of children
        """
        if not self.children:
            return [], None

        return sorted(self.children, key=lambda n: n.get_weight_at(t), reverse=True)

    def join_text_children(self, t, top_children):
        """
        Join the text of top children.

        :param t: Current time
        :param top_children: List of top children
        :return: Joined text
        """
        if self.join_num <= 1:
            return top_children[0].eval_text(t)

        num = min(self.join_num, len(top_children))
        join_char = ', ' if self.join_char == ',' else self.join_char
        return join_char.join(child.eval_text(t) for child in top_children[:num])

    def get_timeline_sector_start(self):
        """Get the start of the timeline sector."""
        if self.timeline_start_f is None:
            return self.parent.get_timeline_sector_start() if self.parent else 0
        return self.timeline_start_f

    def get_timeline_sector_end(self):
        """Get the end of the timeline sector."""
        if self.timeline_end_f is None:
            return self.parent.get_timeline_sector_end() if self.parent else rv.n
        return self.timeline_end_f

    def get_timeline_sector_node(self):
        """Get the topmost node of the timeline sector."""
        return self.parent.get_timeline_sector_node() if self.parent else self

    def get_weight_at(self, t):
        """
        Get the weight of this node at time t.

        :param t: Current time
        :return: Weight at time t
        """
        return self.get_bake_at(t) if self.timeline else self.w

    @trace_decorator
    def get_bake_at(self, t):
        """
        Get the baked value at time t.

        :param t: Current time
        :return: Baked value at time t
        """
        if self.timeline is None:
            raise RuntimeError(f"Cannot evaluate a PNode without timeline ({self.eval_text(t)})")

        idx = int(clamp(t * rv.fps, 0, rv.duration * rv.fps))
        last = self.timeline[idx]
        next = self.timeline[idx + 1]

        frame_start = idx / rv.fps
        interframe = (t - frame_start) / (1 / rv.fps)

        return lerp(last, next, interframe)

    def print(self, depth=0):
        """
        Print the node and its children.

        :param depth: Current depth in the tree
        """
        indent = "   " * depth
        print(f"{indent}{self}")

        ordered_children = self.get_top_children(rv.t) if self.is_sorted_print() else self.children

        # Limit to 5 children past first depth
        if depth > 0 and len(ordered_children) > 5:
            for child in ordered_children[:5]:
                child.print(depth + 1)
            print(f"{indent}...")
        else:
            for child in ordered_children:
                child.print(depth + 1)

    def __str__(self):
        """String representation of the node."""
        t_length, t_timestep = 10, 0.5
        type_str = type(self).__name__
        text_str = self.text or 'none-str'

        if self.timeline is not None:
            t_values = [self.get_bake_at(i * t_timestep) for i in range(t_length)]
            str_timeline = [f"{v:.1f}" for v in t_values]
            return f"{type_str}({self.uuid}, {self.text}, {self.w:.02f}: {', '.join(str_timeline)})"
        else:
            return f"{type_str}({self.uuid}, {text_str}, {self.w:.02f}, {self.last_target_kf})"

    # Implement container-like behavior
    def __getitem__(self, k):
        return self.children[k]

    def __setitem__(self, k, v):
        self.children[k] = v

    def __delitem__(self, k):
        del self.children[k]

    def __contains__(self, obj):
        return obj in self.children

    def __iter__(self):
        return iter(self.children)

    def __next__(self):
        return next(self.children)
