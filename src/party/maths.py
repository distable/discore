# Very useful maths for disco partying,
# along with a parametric eval and global math env to use for evaluation

import random
import sys
import time
from math import *

import numpy as np
import torch
from numpy import ndarray
from numpy.lib.stride_tricks import sliding_window_view
from opensimplex import OpenSimplex
from scipy.signal import find_peaks

from src import renderer
from src.lib.printlib import trace_decorator

# import pytti
simplex = OpenSimplex(random.randint(-999999, 9999999))
math_env = {}

f = 0
rv = renderer.rv

epsilon = 0.0001


def prepare_math_env(*args):
    math_env.clear()
    for a in args:
        for k, v in a.items():
            math_env[k] = v


def update_math_env(k, v):
    math_env[k] = v
    globals()[k] = v
    # print(math_env[k])
    # print(globals()[k])
    # print(globals())
    # math_env = {'abs': abs, 'max': max, 'min': min, 'pow': pow, 'round': round, '__builtins__': None}
    # math_env.update({key: getattr(math, key) for key in dir(math) if '_' not in key})


SEED_MAX = 2 ** 32 - 1

counters = dict(perlin01=0)

perlin_cache = []
max_perlin_cache = 10
enable_perlin_cache = False

def reset():
    for k in counters.keys():
        counters[k] = 0

    perlin_cache.clear()

def set_seed(seed=None, with_torch=True):
    if seed is None:
        seed = int(time.time())
    elif isinstance(seed, str):
        seed = str_to_seed(seed)

    seed = seed % SEED_MAX
    seed = int(seed)

    global simplex
    global current_seed

    current_seed = seed

    random.seed(seed)
    np.random.seed(seed)

    if with_torch:
        torch.manual_seed(seed)
    pass


def str_to_seed(seed):
    import hashlib
    seed = hashlib.sha1(seed.encode('utf-8'))
    seed = int(seed.hexdigest(), 16)
    seed = seed % SEED_MAX
    return seed


def parametric_eval(string, **kwargs):
    if isinstance(string, str):
        try:
            # print(len(math_env))
            # print(math_env['cosb'])
            output = eval(string)
        except SyntaxError as e:
            raise RuntimeError(f'Error in parametric value: {string}')

        return output
    elif isinstance(string, list):
        return val_or_range(string)
    elif isinstance(string, tuple):
        return val_or_range(string)
    else:
        return string


def choose_or(l: list, default, p=None):
    if len(l) == 1:
        return l[0]
    elif len(l) == 0:
        return default
    else:
        ret = random.choices(l, p)
        if isinstance(ret, list):
            return ret[0]
        return ret


def choose(l: list, w=None, exclude=None):
    if exclude is not None:
        if w is None:
            w = [1] * len(l)
        w[exclude] = 0

    ret = random.choices(l, weights=w)
    if isinstance(ret, list):
        return ret[0]
    return ret

def choices(l: list, n=1, p=None):
    # Choose n items from l
    return random.choices(l, k=n, weights=p)


def choose_or(l: list, default, p=None):
    if len(l) == 1:
        return l[0]
    elif len(l) == 0:
        return default
    else:
        ret = random.choices(l, p)
        if isinstance(ret, list):
            return ret[0]
        return ret


def wchoose(l: list, weights):
    return random.choices(l, weights)

def rng(min=None, max=None):
    if not min and not max:
        return random.random()
    elif not min and max:
        return random.uniform(0, max)
    else:
        return random.uniform(min, max)

def rngi(min=None, max=None):
    if not min and not max:
        return random.random(-sys.maxsize, sys.maxsize)
    elif not min and max:
        return random.randint(0, max)
    else:
        return random.randint(min, max)


def rngn(factor):
    if factor > 0:
        return rng(1, 1 + factor)
    else:
        return rng(1 - factor, 1)


def val_or_range(v, max=None):
    if isinstance(v, list) or isinstance(v, tuple):
        return random.uniform(v[0], v[1])
    elif isinstance(v, float) or isinstance(v, int):
        return float(v)
    else:
        raise ValueError(f"maths.val_or_range: Bad argument={v}, type={type(v)}")


@trace_decorator
def get_rand_sum(n, lo, hi):
    ret = []
    i = 0
    while not lo < sum(ret) < hi:
        ret = [rng(lo / n, hi / n) for _ in range(n)]
        i += 1
        if i > 100:
            print(f"Couldn't find a valid sum for get_rand_sum after {i} tries (n={n}, lo={lo}, hi={hi}, sum={sum(ret)}, ret={ret}, lo/n={lo / n}, hi/n={hi / n})")
            break

    return tuple(ret)


def smooth_damp(current, target, current_velocity, smooth_time, max_speed, delta_time):
    smooth_time = max(0.0001, smooth_time)
    inverse_smooth_time = 2 / smooth_time
    delta_time_scaled = inverse_smooth_time * delta_time
    damping_coefficient = 1 / (1 + delta_time_scaled + 0.48 * delta_time_scaled * delta_time_scaled + 0.235 * delta_time_scaled * delta_time_scaled * delta_time_scaled)
    delta_current_target = current - target
    target_temp = target
    max_speed_smooth_time = max_speed * smooth_time
    delta_current_target = min(max(delta_current_target, -max_speed_smooth_time), max_speed_smooth_time)
    target = current - delta_current_target
    velocity_scaled_delta = (current_velocity + inverse_smooth_time * delta_current_target) * delta_time
    current_velocity = (current_velocity - inverse_smooth_time * velocity_scaled_delta) * damping_coefficient
    current_temp = target + (delta_current_target + velocity_scaled_delta) * damping_coefficient
    if (target_temp - current > 0) == (current_temp > target_temp):
        current_temp = target_temp
        current_velocity = (current_temp - target_temp) / delta_time
    return current_temp, current_velocity


def lerp(a, b, t):
    t = np.nan_to_num(t)
    return a + (b - a) * clamp01(t)


def jlerp(a, b, t, k=0.3):
    return lerp(a, b, jcurve(t, k))


def rlerp(a, b, t, k=0.3):
    return lerp(a, b, rcurve(t, k))


def slerp(a, b, t, k=0.3):
    return lerp(a, b, scurve(t, k))


def ilerp(lo, hi, v):
    ret = clamp01((v - lo) / (hi - lo))
    return np.nan_to_num(ret)


def remap(v, a, b, x, y):
    t = ilerp(a, b, v)
    return lerp(x, y, t)


def sign(v):
    return np.sign(v)


def stsin(t, a, p, w):
    return abs(np.sin(t / p * pi) ** w * a)  # * sign(np.sin(t / p * pi))


def stcos(t, a, p, w):
    return sign(cos(t / p)) * cos(t / p) ** w * a

def sin(t):
    return np.sin(t * tau)

def cos(t):
    return np.cos(t * tau)

def sin01(p=1, w=1, size=None):
    if size is None:
        size = rv.n
    t = np.linspace(0, tau * size / rv.fps / p, size)
    return (0.5 + 0.5 * np.sin(t)) ** w

def sin11(p=1, w=1, size=None):
    return sin01(p, w, size) * 2 - 1

def cos11(p=1, w=1, size=None):
    return cos01(p, w, size) * 2 - 1

def cos01(p=1, w=1, size=None):
    if size is None:
        size = rv.n
    t = np.linspace(0, tau * size / rv.fps / p, size)
    return (0.5 + 0.5 * np.cos(t)) ** w

def swave(t):
    return 1


def cwave(t, p1, p2, a1, a2, o1):
    return 1


def sinb(t, a=1, p=1, o=0):
    return sin(((t / p) - o) * pi) * a


def cosb(t, a=1, p=1, o=0):
    return cos(((t / p) - o) * pi) * a


def sin1(t, a=1, p=1, o=0):
    return (sin((t / p) - o) * .5 + .5) * a


def cos1(t, a=1, p=1, o=0):
    return (cos((t / p) - o) * .5 + .5) * a

def range1(start, finish):
    """
    Return ones from start to finish
    """
    ret = np.zeros(rv.n)

    if isinstance(start, str):
        from src.classes import paths
        start_s = paths.parse_time_to_seconds(start)
        start = int(start_s * rv.fps)

    if isinstance(finish, str):
        from src.classes import paths
        finish_s = paths.parse_time_to_seconds(finish)
        finish = int(finish_s * rv.fps)

    ret[start:finish] = 1
    return ret

def range0(start, finish):
    """
    Return zeros from star to finish
    """
    ret = np.ones(rv.n)
    ret[start:finish] = 0
    return ret

def tsigmoid(x, k=0.3, norm_window=None):
    lo, hi, span = 0, 0, 0

    if isinstance(x, ndarray) and norm_window is not None:
        x, lo, hi = exnorm(x, norm_window)

    x = (x - k * x) / (k - 2 * k * np.abs(x) + 1)

    if isinstance(x, ndarray) and norm_window is not None:
        x = x * (hi - lo) + lo

    return x


def scurve(x, k=0.3, norm_window=None):
    return (1 + tsigmoid(2 * x - 1, -k, norm_window)) / 2


def jcurve(x, k=0.3, norm_window=None):
    return tsigmoid(clamp01(x), k, norm_window)


def rcurve(x, k=0.3, norm_window=None):
    return tsigmoid(clamp01(x), -k, norm_window)


def srcurve(v, a=0.3, b=0.3, norm=None):
    return scurve(rcurve(v, a, norm), b)


def jrcurve(v, a=0.3, b=0.3, norm=None):
    return jcurve(rcurve(v, a, norm), b)

def nprng(lo=0, hi=1) -> ndarray:
    return np.random.uniform(lo, hi, rv.n)

def rng01(lo=0, hi=1) -> ndarray:
    return np.random.uniform(lo, hi, rv.n)

def rng11(lo=-1, hi=1) -> ndarray:
    return np.random.uniform(lo, hi, rv.n)

def perlin01(freq=1.0, lo=0, hi=1) -> ndarray:
    counters['perlin01'] += 1

    if enable_perlin_cache and len(perlin_cache) > max_perlin_cache:
        return perlin_cache[counters['perlin01'] % len(perlin_cache)]

    t = np01(rv.n)
    t *= freq * 100
    t += counters['perlin01'] * 333.33
    nperlin = simplex.noise2array(t, np.zeros(1))[0]
    nperlin = (nperlin + 1) * 0.5
    nperlin = nperlin * (hi - lo) + lo

    if enable_perlin_cache:
        perlin_cache.append(nperlin)

    return nperlin


@trace_decorator
def perlin11(freq=1.0, lo=-1, hi=1) -> ndarray:
    return perlin01(freq, lo, hi)

def perlin(t, freq=1.0) -> ndarray | float:
    if isinstance(t, ndarray):
        return npperlin(t, freq)
    else:
        return simplex.noise2(t * freq, 0)

def max(a, b):
    return np.maximum(a, b)

def schedule(*schedule, size=None) -> ndarray:
    if size is None:
        size = rv.n
    return np.resize(schedule, size)

def np01(size=None) -> ndarray:
    if size is None:
        size = rv.n
    return np.linspace(0, 1, size)

def np0(hi, size=None) -> ndarray:
    if size is None:
        size = rv.n
    return np.linspace(0, hi, size)

def npperlin(count, freq=1.0, off=0) -> ndarray:
    if isinstance(count, ndarray):
        count = count.shape[0]

    ret = simplex.noise2array(np.arange(count) * freq + off, np.zeros(1))[0]
    # ret = np.zeros(count)
    # for i in range(count):
    #     ret[i] = perlin(i + off, freq)

    return ret


def npperlin_like(ar, freq=1, off=0) -> ndarray:
    return npperlin(ar.shape[0], freq, off)

# def noisemix(*ndarrays):
#     # Mix between ndarrays using perlin noise (spread out evenly across 0-1)
#     noise = perlin01()
#     ret = np.zeros(rv.n)
#     for i in range(len(ndarrays)):
#         ret += ndarrays[i] * (noise > i / len(ndarrays))

def iszero(arr):
    if arr is None:
        return True
    if isinstance(arr, ndarray):
        return np.count_nonzero(arr) == 0
    return arr == 0

def clamp(v, lo, hi):
    return np.clip(v, lo, hi)


def clamp01(v):
    return np.clip(v, 0, 1)


def euler_to_quat(pitch, yaw, roll):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return (qw, qx, qy, qz)


def noisy(noise_typ, image):
    """
    Add perlin to an image
    https://stackoverflow.com/a/30609854/1319727
    """
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


def schedule_spk(v, t):
    sum = 0
    l = len(t) if isinstance(t, list) else len(v) if isinstance(v, list) else 0

    for i in range(int(l)):
        tt = (t[i] if isinstance(t, list) else t)
        vv = (v[i] if isinstance(v, list) else v)
        if f % tt == 0:
            sum += vv
    return sum


# def schedule(*args):
#     return args[f % len(args)]


def pdiff(x):
    v = np.diff(x)
    v = np.append(v, v[-1])
    return v


def absdiff(x):
    return abs(pdiff(x))


def pconvolve(x, kernel):
    v = np.convolve(x, kernel)
    return v[:-(len(kernel) - 1)]


def symnorm(x, window=None):
    return norm(x, window, True)


def norm(x, window=None, symmetric=False):
    if window:
        window *= rv.fps
    x, lo, hi = exnorm(x, window, symmetric)
    return x


def exnorm(x, window=None, symmetric=False):
    if window is not None:
        window = int(window)
        window = min(window, len(x))

        lo = np.min(sliding_window_view(x, window_shape=window), axis=1)
        hi = np.max(sliding_window_view(x, window_shape=window), axis=1)

        lo = np.pad(lo, (0, window - 1), 'edge')
        hi = np.pad(hi, (0, window - 1), 'edge')
    else:
        lo = np.min(x)
        hi = np.max(x)

    if symmetric:
        b = np.hi(np.abs(lo), np.abs(hi))
        lo = -b
        hi = b

    x_ret = (x - lo) / (hi - lo)
    return np.nan_to_num(x_ret), lo, hi


def stretch(array: np.ndarray, new_len: int) -> np.ndarray:
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)


def wavg(signal, window=0.08):
    from scipy.ndimage import uniform_filter1d
    size = int(signal.shape[0] * window)
    return uniform_filter1d(signal, size=size)


def convolve(arr, n_iter=20, kernel=None, mask=None):
    if kernel is None:
        kernel = [1.0, 1.0, 1.0]
    if mask is None:
        mask = np.ones_like(arr)

    original = arr

    lo = arr.min()
    hi = arr.max()

    arr = norm(arr)

    kernel = np.array(kernel)
    for i in range(n_iter):
        arr = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, arr)

    arr = norm(arr)
    arr = arr * (hi - lo) + lo

    ret = lerp(original, arr, mask)
    ret = np.nan_to_num(ret)
    return ret

def blur(arr, n_iter=20, mask=None):
    return convolve(arr, n_iter=n_iter, kernel=[1.0, 1.0, 1.0], mask=mask)

def sharpen(arr, n_iter=20, mask=None):
    return convolve(arr, n_iter=n_iter, kernel=[-1.0, 2.0, -1.0], mask=mask)

def get_peaks(s_chapter, prominence=0.01):
    peak_indices = find_peaks(s_chapter, prominence=prominence)[0]
    peaks = np.zeros_like(s_chapter)
    peaks[peak_indices] = 1
    return peaks

def get_valleys(s_chapter, prominence=0.01):
    valley_indices = find_peaks(-s_chapter, prominence=prominence)[0]
    valleys = np.zeros_like(s_chapter)
    valleys[valley_indices] = 1
    return valleys

def get_prominence(s_chapter, prominence=0.01):
    peak_indices = find_peaks(s_chapter, prominence=prominence)[0]
    valley_indices = find_peaks(-s_chapter, prominence=prominence)[0]
    prominence = np.zeros_like(s_chapter)
    prominence[peak_indices] = 1
    prominence[valley_indices] = -1
    return prominence

def smooth_1euro(x, min_cutoff=0.004, beta=0.7):
    end = x.shape[0]

    x_noisy = x
    x_hat = np.zeros_like(x_noisy)
    x_hat[0] = x_noisy[0]
    t = np.linspace(0, end, len(x))
    one_euro_filter = OneEuroFilter(t[0], x_noisy[0], min_cutoff=min_cutoff, beta=beta)

    for i in range(1, len(t)):
        x_hat[i] = one_euro_filter(t[i], x_noisy[i])

    return x_hat


def smoothing_factor(t_e, cutoff):
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev

def zero():
    return np.zeros(rv.n)

def one(v=1.0):
    return np.ones(rv.n) * v

def thresh(v, t=0.5, s=1):
    return clamp01(v * s - t)

def rebase(v, t=0.5, s=1):
    return norm(clamp01(v * s - t))

def arr_slerp(v1, v2, t, DOT_THRESHOLD=0.9995):
    """
    Spherical interpolation between two arrays v1 v2
    """
    inputs_are_torch = False
    if not isinstance(v1, np.ndarray):
        inputs_are_torch = True
        input_device = v1.device
        v1 = v1.cpu().numpy()
        v2 = v2.cpu().numpy()

    dot = np.sum(v1 * v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v1 + t * v2
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v1 + s1 * v2

    if inputs_are_torch:
        import torch
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

class TimeRect:
    def __init__(self):
        pass

    def cutup(self, binary_arr):
        pass

    def lowpass_extract(self, binary_arr):
        pass


# pytti.parametric_eval = parametric_eval

set_seed()

# Aliases & shortcuts
jc = jcurve
rc = rcurve
sr = srcurve
jr = jrcurve