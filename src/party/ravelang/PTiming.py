from src.classes.paths import parse_time_to_seconds
from src.party.ravelang.PNode import PNode
from src.renderer import rv


class PTiming(PNode):
    # A node with a list of child, eval_text returns the text of the child for the current time
    # Example usage
    #
    # # Change the artist every second, then hang on the last one
    # TimingNode("""
    # 00:00 Salvador Dali
    # 00:01 Van Gogh
    # 00:02 Picasso
    # """)

    def __init__(self, children, **kwargs):
        super(PTiming, self).__init__(**kwargs)
        # Check every line, get the time for each line and text, and then create a prompt node and add to children with add_child
        self.times = {}
        time_f_last = 0
        for line in children.split('\n'):
            if len(line) == 0: continue
            t, text = line.split(' ', 1)

            time_sec = parse_time_to_seconds(t)
            time_f = rv.to_frame(time_sec)

            node = PNode(text, t)
            node.timeline_start_f = time_f_last
            node.timeline_end_f = time_f
            self.times[node] = time_sec
            self.add_child(node)

            time_f_last = time_f

    def can_promote(self):
        return False

    def get_debug_string(self,
                         t_length=10,
                         t_timestep=0.5):
        return f"PTiming(children={len(self.children)})"

    def eval_text(self, t):
        # Find the node with the closest time
        closest_node = None
        closest_time = 999

        # print(f"PTiming.eval_text: t={t}")

        for node in self.children:
            node_t = self.times[node]
            # print(f"PTiming.eval_text: {node_t}, {node.eval_text(t)}, {closest_time}, {closest_time}")
            if abs(node_t - t) < closest_time and node_t <= t:
                closest_node = node
                closest_time = abs(node_t - t)

        if closest_node is None:
            return ''

        return closest_node.eval_text(t)
