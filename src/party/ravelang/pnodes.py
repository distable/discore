import copy
from pathlib import Path

from src.classes import paths
from src.lib.loglib import trace_decorator
from src.party import maths
from src.party.ravelang.PCycle import PCycle
from src.party.ravelang.PList import PList
from src.party.ravelang.PNode import PNode, wc_regex

font = None


def dfs(node):
    stack = list()
    stack.append(node)

    while len(stack) > 0:
        n = stack.pop()

        yield n

        for child in n.children:
            stack.append(child)


def clone_node_recursive(template):
    new_node = copy.copy(template)
    new_node.on_copied()

    # Iterate through children and clone them
    # Another manual DFS since we need to swap the children one by one
    stack = list()
    stack.append(new_node)

    while len(stack) > 0:
        n = stack.pop()

        for i in range(len(n.children)):
            n.children[i] = copy.copy(n.children[i])
            n.children[i].on_copied()
            n.children[i].parent = n
            stack.append(n.children[i])


    return new_node


def transform(self, wcconfs, globals):
    # Promote
    # Wrap string children in PList
    for i in range(len(self.children)):
        if isinstance(self.children[i], str):
            self.children[i] = PList(self.children[i])

    if len(self.children) == 1 and self.children[0].can_promote():
        self.children = self.children[0].children

    import re

    if not self.text:
        return False

    text_or_children = self.text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text_or_children = re.sub(r'\s+', ' ', text_or_children)  # Collapse multiple spaces to 1, caused by multiline strings
    text_or_children = text_or_children.strip()

    # Match map \<\w+\>
    children = []
    i = 0
    last_i = 0

    def append_wc():
        nonlocal last_i
        s = text_or_children[last_i:i + 1]
        if not s:
            return

        s = s[1:-1]
        if paths.is_valid_url(s):
            pass

        if Path(s).exists() and paths.is_image(s):
            pass
        else:
            parts = re.findall(wc_regex, s)[0]
            txt_children = parts[1]
            txt_template = parts[3]
            txt_joinconf = parts[5]

            if ';' in txt_children or ',' in txt_children:
                tmpl_children = PList(txt_children).children
            else:
                env_template = globals.get(txt_children)
                if env_template is None:
                    env_template = wcconfs.get(txt_children)
                tmpl_children = env_template.children
            if tmpl_children is None:
                raise Exception(f"Couldn't get set: {txt_children}")

            tmpl_node = globals.get(txt_template) or wcconfs.get(txt_children) or wcconfs.get('*')
            if isinstance(tmpl_node, list):
                tmpl_node = maths.choose(tmpl_node)
            if tmpl_node is None:
                tmpl_node = PCycle(interval=1)

            tmpl_node.children = tmpl_children
            replacement_node = clone_node_recursive(tmpl_node)
            replacement_node.text = parts[0]
            replacement_node.join_num = 1
            if txt_joinconf:
                replacement_node.join_char = txt_joinconf[-1]
                replacement_node.join_num = int(int(txt_joinconf[:-1]))
            children.append(replacement_node)
            last_i = i + 1

    def append_text():
        nonlocal last_i
        s = text_or_children[last_i:i]
        if not s: return
        node = PNode(s)
        children.append(node)
        last_i = i + 1

    charlist = list(text_or_children)
    while i < len(charlist):
        c = charlist[i]
        if c == '<':
            append_text()
            last_i = i
            i = text_or_children.find('>', i)
            append_wc()

        i += 1

    if children:
        append_text()
        self.add_children(children)

        return True

    return False


@trace_decorator
def bake(root):
    ret = []

    # root.bake()
    for v in dfs(root):
        v.init_bake()

    for v in dfs(root):
        v.bake()
        ret.append(v)

    return ret


# bake_prompt(f"{scene} with black <a_penumbra> distance fog. Everything is ultra detailed, has 3D overlapping depth effect, into the center, painted with neon reflective/metallic/glowing ink, covered in <a_gem> <a_gem2> gemstone / <n_gem> ornaments, <a_light> light diffraction, (vibrant and colorful colors), {style}, painted with (acrylic)", settypes, setmap, locals())
def print_prompts(root, min, max, step=1):
    for t in range(min, max, step):
        print(f"prompt(t={t}) ", eval_prompt(root, t))


@trace_decorator
def bake_prompt(prompt: str, wcconfs, globals):
    root = PNode(prompt)
    root.join_num = 9999999

    root.print()

    transform(root, wcconfs, globals)

    for v in dfs(root):
        v.init_bake()
    bake(root)

    num = 0
    for _ in dfs(root):
        num += 1

    print(f"pnodes.bake_prompt: Baked {num} nodes")
    if num < 50000:
        root.print()
    else:
        print(f"pnodes.bake_prompt: Too many nodes to print tree ({num})")
    # print("")
    # print_prompts(root, 1, 30)

    return root


@trace_decorator
def eval_prompt(root, t, require_bake=False):
    # print('eval_prompt', t)
    has_bake = True
    if require_bake:
        for node in dfs(root):
            # print("DFS", node)
            # print(node, node.timeline)
            if node.timeline is None:
                has_bake = False
                break

            try:
                node.get_bake_at(t)
            except:
                has_bake = False
                break

    return root.eval_text(t)
