import math
import types
from colorsys import hsv_to_rgb

import numpy as np

from src import renderer
from src.party import maths
from src.renderer import rv

rgb_to_hex = lambda tuple: f"#{int(tuple[0] * 255):02x}{int(tuple[1] * 255):02x}{int(tuple[2] * 255):02x}"


def generate_colors(n):
    golden_ratio_conjugate = 0.618033988749895
    h = 0
    ret = []
    for i in range(n):
        h += golden_ratio_conjugate
        ret.append(rgb_to_hex(hsv_to_rgb(h, 0.825, 0.915)))

    return ret


def mod2dic(module):
    if isinstance(module, dict):
        return module
    elif isinstance(module, types.ModuleType):
        return {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}
    elif isinstance(module, list):
        dics = [mod2dic(m) for m in module]
        return {k: v for d in dics for k, v in d.items()}
    else:
        return module.__dict__


def get_evalenv():
    envdic = {}

    def add_object(obj):
        if obj is None:
            return
        envdic.update(obj.__dict__)

    def add_module(mod):
        envdic.update(mod2dic(mod))

    def add_dict(dic):
        envdic.update(dic)

    add_module(math)
    add_module(np)
    add_module(maths)
    add_object(renderer.get_script())
    add_object(rv)
    add_dict(rv._signals)
    envdic['rv'] = rv

    return envdic
