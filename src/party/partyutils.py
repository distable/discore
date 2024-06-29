import re
import time
from colorsys import *

class Timer:
    def __enter__(self):
        self.start = time.process_time()
        return self

    def __exit__(self, *args):
        self.end = time.process_time()
        self.interval = self.end - self.start


def kprint(*kargs):
    s = ""
    for v in kargs:
        s += f"{v}  "

    print(s)


def kprint(*kargs):
    s = ""
    for v in kargs:
        s += f"{v}  "

    print(s)


def printkw(**kwargs):
    s = ""
    for k, v in kwargs.items():
        # idk why we have to check for bools or if its even required, not gonna question it I have better things to do like actually getting stuff done
        if isinstance(v, (float, complex)) and not isinstance(v, bool):
            s += f"{k}={v:.2f}  "
        elif isinstance(v, int) and not isinstance(v, bool):
            s += f"{k}={v}  "
        else:
            s += f"{k}={v}  "

    print(s)


def flat(pool):
    res = []
    for v in pool:
        if isinstance(v, list):
            res += flat(v)
        else:
            if isinstance(v, int):
                res.append(v)
    return res


def mod2dic(module):
    return {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}


rgb_to_hex = lambda tuple: f"#{int(tuple[0] * 255):02x}{int(tuple[1] * 255):02x}{int(tuple[2] * 255):02x}"
hex_to_rgb = lambda hx: (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))


def generate_colors(n):
    golden_ratio_conjugate = 0.618033988749895
    h = 0
    ret = []
    for i in range(n):
        h += golden_ratio_conjugate
        ret.append(rgb_to_hex(hsv_to_rgb(h, 0.825, 0.915)))

    return ret


def split_and_keep(seperator, s):
    return re.split(';', re.sub(seperator, lambda match: match.group() + ';', s))
