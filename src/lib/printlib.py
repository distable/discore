import os

from src.classes import paths
import torch

_print = print

import sys
import time
import traceback
from contextlib import contextmanager
from time import perf_counter

import numpy as np

import jargs

print_timing = False
print_trace = jargs.args.trace
print_gputrace = jargs.args.trace

last_time = time.time()

# Set default decimal precision for printing
np.set_printoptions(precision=2, suppress=True)


# import torch
# torch.set_printoptions(precision=2)

# stdout = sys.stdout
# sys.stdout = None

# _print = print

# def print(*kargs, **kwargs):
#     _print(*kargs, file=stdout, **kwargs)

def run(code, task):
    try:
        code()
    except Exception as e:
        print(f"{task}: {type(e).__name__}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


def kprint(*kargs):
    s = ""
    for v in kargs:
        s += f"{value_to_print_str(v)}  "

    print(s)


def printkw(chalk=None, **kwargs):
    s = ""
    for k, v in kwargs.items():
        s += f"{value_to_print_str(k)}={value_to_print_str(v)}"
        s += "  "

    if chalk:
        print(chalk(s))
    else:
        print(s)


def print(*args, **kwargs):
    from beeprint import pp
    from munch import Munch
    if isinstance(args[0], dict) or isinstance(args[0], Munch):
        pp(*args, **kwargs)
    else:
        _print(*args, **kwargs)


def print_bp(msg, *args, **kwargs):
    print(f' - {msg}', *args, **kwargs)


def printerr(msg, *args, **kwargs):
    print(msg, *args, **kwargs)


def printerr_bp(msg, *args, **kwargs):
    print(f' - {msg}', *args, **kwargs)


def print_cmd(cmd):
    if jargs.args.print:
        print(f'> {cmd}')


def pct(f: float):
    """
    Get a constant percentage string, e.g. 23% instead of 0.23, 04% instead of 0.04
    """
    if np.isnan(f):
        return 'nan%'

    return f'{int(f * 100):02d}%'


def make_print(module_name):
    def ret(msg, *args, **kwargs):
        # Wrap all args[1:] in chalk.grey
        # if len(args) > 1:
        #     args = [args[0]] + [chalk.grey(str(a)) for a in args[1:]]
        # if len(kwargs) > 1:
        #     # key in red
        #     kwargs = {chalk.white(k): chalk.dim(str(v)) for k, v in kwargs.items()}

        if not print_timing:
            print(f"[{module_name}] {msg}", *args, **kwargs)
        else:
            # Print the elapsed time since the last call to this function
            import time
            global last_time
            if last_time and print_timing:
                print(f"[{module_name}] ({time.time() - last_time:.2f}s) {msg}",
                      *args, **kwargs)
            else:
                print(f"[{module_name}] {msg}", *args, **kwargs)
            last_time = time.time()

    return ret


def make_printerr(module_name):
    def ret(msg, *args, **kwargs):
        printerr(f"[{module_name}] {msg}", *args, **kwargs)

    return ret


indent = 0


@contextmanager
def trace(name) -> float:
    global indent

    start = perf_counter()

    indent += 1
    try:
        yield lambda: perf_counter() - start
    except Exception as e:
        raise e
    finally:
        indent -= 1

    seconds = perf_counter() - start
    if (seconds * 1000) < 2:
        return
    if seconds >= 1:
        s = f'({seconds:.3f}s) {name}'
    else:
        s = f'({int(seconds * 1000)}ms) {name}'
    from yachalk import chalk
    if print_trace:
        s = '..' * indent + s
        print(chalk.grey(s))


@contextmanager
def gputrace(name, vram_dt=False) -> float:
    import torch
    vram = 0
    if vram_dt:
        vram = torch.cuda.memory_allocated()
    start = perf_counter()
    yield lambda: perf_counter() - start
    s = f'{name}: {perf_counter() - start:.3f}s'
    if vram_dt:
        vram = (torch.cuda.memory_allocated() - vram) / 1024 / 1024 / 1024
        s += f' {vram:.3f}GB / {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.3f}GB'
    from yachalk import chalk
    if print_gputrace:
        print(chalk.grey(s))


@contextmanager
def cputrace(name, enable=True, enable_trace=False) -> float:
    if not enable:
        if enable_trace:
            trace(name)
        yield None
        return

    import yappi
    yappi.clear_stats()
    yappi.set_clock_type('cpu')
    yappi.start()

    yield None

    columns = {
        0: ("name", 80),
        1: ("ncall", 5),
        2: ("tsub", 8),
        3: ("ttot", 8),
        4: ("tavg", 8)
    }
    # Print and limit the number of functions to 100
    yappi.get_func_stats().print_all(columns=columns)
    yappi.stop()
    input("Press Enter to continue...")


def _trace_wrapper(func, print_args, *args, **kwargs):
    s_args = ''
    if print_args:
        s_kargs = []
        for a in args:
            s_kargs.append(value_to_print_str(a))

        s_kwargs = []
        for k, v in kwargs.items():
            s_k = value_to_print_str(k)
            s_v = value_to_print_str(v)
            s_kwargs.append(f'{s_k}={s_v}')

        s_args = ', '.join(s_kargs + s_kwargs)

    with trace(f'{func.__name__}({s_args})'):
        return func(*args, **kwargs)


def trace_decorator(func):
    def trace_wrapper(*args, **kwargs):
        return _trace_wrapper(func, True, *args, **kwargs)

    return trace_wrapper


def trace_decorator_noarg(func):
    def trace_wrapper(*args, **kwargs):
        return _trace_wrapper(func, False, *args, **kwargs)

    return trace_wrapper


def value_to_print_str(v):
    from PIL import Image

    s_v = ""
    if isinstance(v, np.ndarray):
        s_v = f"ndarray({v.shape})"
    elif isinstance(v, torch.Tensor):
        s_v = f"Tensor({v.shape})"
    # Simplified PIL image
    elif isinstance(v, Image.Image):
        s_v = f"PIL({v.width}x{v.height}, {v.mode})"
    # idk why we have to check for bools or if its even required, not gonna question it I have better things to do like actually getting stuff done
    elif isinstance(v, (float, complex)) and not isinstance(v, bool):
        s_v = f"{v:.2f}"
    elif isinstance(v, int) and not isinstance(v, bool):
        s_v = f"{v}"
    # tuple of floats
    elif isinstance(v, (tuple, list)) and len(v) > 0 and isinstance(v[0],
                                                                    float):
        # convert each float to a string with 2 decimal places
        v = tuple(f"{x:.2f}" for x in v)
        s_v = f"{v}"
    # Limit floats to 2 decimals
    elif isinstance(v, float):
        s_v = f"{v:.2f}"
    else:
        s_v = f"{v}"

    return s_v


def print_possible_scripts():
    from yachalk import chalk

    for file in paths.get_script_paths():
        # Load the file (which is python) and get the docstring at the top
        # The docstring can span multipline lines
        with open(file, "r") as f:
            fulldocstring = ''
            docstring = f.readline().strip()

            if docstring.startswith('"""'):
                if len(docstring) > 3 and docstring.endswith('"""'):
                    fulldocstring = docstring[3:-3]
                elif docstring == '"""':
                    fulldocstring = f.readline().strip()
                    if fulldocstring.endswith('"""'):
                        fulldocstring = fulldocstring[:-3]

        print(
            f"  {os.path.relpath(file, paths.scripts)[:-3]}  {chalk.dim(fulldocstring)}")


def print_existing_sessions():
    from yachalk import chalk
    from src.classes.Session import Session
    files = list(paths.sessions.iterdir())
    files.sort(key=lambda x: x.stat().st_mtime, reverse=False)

    for file in files:
        # Filters
        if not file.is_dir(): continue
        if file.name.startswith('__pycache__'): continue

        # Print the session
        s = Session(file, log=False)
        s_name = chalk.bold(str.ljust(s.name, 25))

        print(
            f"  {s_name}\t{s.w}x{s.h}\t\t{s.f} frames\t{chalk.dim(s.dirpath)}")


progress_print_out = sys.stdout
