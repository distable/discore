import colorsys
import os
from datetime import timedelta
from typing import Callable, Any
import sys
import time
import traceback
from contextlib import contextmanager
from time import perf_counter
import numpy as np
import jargs
from yachalk import chalk

_print = print

print_timing = False
print_trace = jargs.args.trace
print_gputrace = jargs.args.trace

last_time = time.time()
trace_indent = 0

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


def printbold(*kargs):
    s = ""
    for v in kargs:
        s += f"{value_to_print_str(v)}  "

    from yachalk import chalk
    print(chalk.bold(s))


def kprint(*kargs):
    s = ""
    for v in kargs:
        s += f"{value_to_print_str(v)}  "

    print(s)


def printkw(chalk=None, print_func=None, **kwargs):
    print_func = print_func or print

    s = ""
    for k, v in kwargs.items():
        s += f"{value_to_print_str(k)}={value_to_print_str(v)}"
        s += "  "

    if chalk:
        print_func(chalk(s))
    else:
        print_func(s)


def print(*args, print_func=None, **kwargs):
    print_func = print_func or _print
    from beeprint import pp
    from munch import Munch
    if isinstance(args[0], dict) or isinstance(args[0], Munch):
        pp(*args, **kwargs)
    else:
        print_func(*args, **kwargs)


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


def make_log(module_name) -> callable:
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


def make_logerr(module_name):
    def ret(msg, *args, **kwargs):
        printerr(f"[{module_name}] {msg}", *args, **kwargs)

    return ret


def format_time(seconds: float) -> str:
    delta = timedelta(seconds=seconds)
    if delta.total_seconds() < 1:
        return f"{delta.microseconds // 1000}ms"
    return str(delta).split('.')[0]  # Remove microseconds for readability


def trace_function(frame, event, arg):
    # This function will be called for each line in the trace function
    # We return None to indicate that we don't want to trace this function
    return None


# @contextmanager
# def trace(name: str) -> Callable[[], float]:
#     start_time = perf_counter()
#     old_trace = sys.gettrace()
#     # sys.settrace(lambda *args: None)  # Disable tracing within this context
#
#     try:
#         yield lambda: perf_counter() - start_time
#     finally:
#         # sys.settrace(old_trace)  # Restore original trace function
#         elapsed_time = perf_counter() - start_time
#         if elapsed_time >= 0.002:  # Only print if time is 2ms or more
#             color_func = get_color_for_time(elapsed_time)
#             print(color_func(f"({elapsed_time:.3f}s) {name}"))

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
        print(chalk.dim(s))


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


# @contextmanager
# def trace(name: str) -> Callable[[], float]:
#     start_time = perf_counter()
#     old_trace = sys.gettrace()
#     # sys.settrace(lambda *args: None)  # Disable tracing within this context
#
#     try:
#         yield lambda: perf_counter() - start_time
#     finally:
#         # sys.settrace(old_trace)  # Restore original trace function
#         elapsed_time = perf_counter() - start_time
#         if elapsed_time >= 0.002:  # Only print if time is 2ms or more
#             print(f"({elapsed_time:.3f}s) {name}")


@contextmanager
def trace(name: str) -> Callable[[], float]:
    global trace_indent
    start_time = perf_counter()
    old_trace = sys.gettrace()
    # sys.settrace(lambda *args: None)  # Disable tracing within this context

    trace_indent += 1
    try:
        yield lambda: perf_counter() - start_time
    finally:
        trace_indent -= 1
        # sys.settrace(old_trace)  # Restore original trace function
        elapsed_time = perf_counter() - start_time
        if elapsed_time >= 0.002 and print_trace:  # Only print if time is 2ms or more
            color_func = get_color_for_time(elapsed_time)
            indentation = ".." * trace_indent
            print(color_func(f"{indentation}({elapsed_time:.3f}s) {name}"))


def _trace_wrapper(func: Callable, print_args: bool, *args: Any, **kwargs: Any) -> Any:
    global trace_indent
    s_args = ''
    if print_args:
        old_trace = sys.gettrace()
        # sys.settrace(lambda *args: None)  # Disable tracing for argument processing
        try:
            s_kargs = [value_to_print_str(a) for a in args]
            s_kwargs = [f'{value_to_print_str(k)}={value_to_print_str(v)}' for k, v in kwargs.items()]
            s_args = ', '.join(s_kargs + s_kwargs)
        finally:
            # sys.settrace(old_trace)  # Restore original trace function
            pass

    with trace(f'{func.__name__}({s_args})'):
        return func(*args, **kwargs)


def trace_decorator(func: Callable) -> Callable:
    def trace_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _trace_wrapper(func, True, *args, **kwargs)

    return trace_wrapper


def trace_decorator_noargs(func: Callable) -> Callable:
    def trace_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _trace_wrapper(func, False, *args, **kwargs)

    return trace_wrapper


def value_to_print_str(v):
    from PIL import Image
    import torch

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
    elif isinstance(v, (tuple, list)) and len(v) > 0 and isinstance(v[0], float):
        # convert each float to a string with 2 decimal places
        s_v += '('
        v = list(f"{x:.2f}" for x in v)
        s_v += '  '.join(v)
        s_v += ')'
    # Limit floats to 2 decimals
    elif isinstance(v, float):
        s_v = f"{v:.2f}"
    else:
        s_v = f"{v}"

    return s_v


def print_possible_scripts():
    from yachalk import chalk
    from src.classes import paths

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
    from src.classes import paths
    for file in paths.iter_session_paths():
        # Filters
        if not file.is_dir(): continue
        if file.name.startswith('__pycache__'): continue

        # Print the session
        s = Session(file, log=False)
        s_name = chalk.bold(str.ljust(s.name, 25))

        print(
            f"  {s_name}\t{s.w}x{s.h}\t\t{s.f} frames\t{chalk.dim(s.dirpath)}")


progress_print_out = sys.stdout


def get_color_for_time(seconds: float) -> Callable[[str], str]:
    # Define our gradient keypoints (time in seconds, hue)
    keypoints = [
        (0, 120),  # Green (very fast)
        (0.5, 80),  # Yellow-Green
        (1, 60),  # Yellow
        (1.5, 30),  # Orange
        (2, 0)  # Red (very slow)
    ]

    # Find the appropriate segment of the gradient
    for i, (time, hue) in enumerate(keypoints):
        if seconds <= time:
            if i == 0:
                r, g, b = colorsys.hsv_to_rgb(hue / 360, 1, 1)
                return chalk.rgb(int(r * 255), int(g * 255), int(b * 255))

            # Interpolate between this keypoint and the previous one
            prev_time, prev_hue = keypoints[i - 1]
            t = (seconds - prev_time) / (time - prev_time)
            interpolated_hue = prev_hue + t * (hue - prev_hue)

            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(interpolated_hue / 360, 1, 1)
            return chalk.rgb(int(r * 255), int(g * 255), int(b * 255))

    # If we're past the last keypoint, return the color for the last keypoint
    r, g, b = colorsys.hsv_to_rgb(keypoints[-1][1] / 360, 1, 1)
    return chalk.rgb(int(r * 255), int(g * 255), int(b * 255))
