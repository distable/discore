import random
from dataclasses import dataclass, field
from math import sqrt
from typing import Dict

import numpy as np
from numpy import ndarray

import jargs
import userconf
from src.classes import convert
from src.lib import loglib
from src.lib.loglib import trace_decorator

log = loglib.make_log('rvars')
logerr = loglib.make_logerr('rvars')

# These are values that cannot be used as signal names
# Because they are either special or cannot change during rendering
protected_names = [
    'img',
    'session', 'signals',
    'w', 'h', 'w2', 'h2',
    't', 'f', 'dt', 'ref', 'tr', 'len',
    'n', 'scalar', 'draft',
    'is_array_mode'
]


class Signal:
    def __init__(self, data: np.ndarray, name: str):
        self.data = data
        self.name = name

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __len__(self):
        return len(self.data)


class SignalGroup(dict):
    def __getitem__(self, key):
        if key not in self:
            raise KeyError(f"Signal '{key}' not found")
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if not isinstance(value, Signal):
            value = Signal(np.array(value), key)
        super().__setitem__(key, value)


class RenderVars:
    """
    Manipulate render data & state with large ndarrays called signals.
    Various
    """
    def __init__(self):
        self.is_array_mode = False
        self.n = 0
        self.prompt = ""
        self.promptneg = ""
        self.nprompt = None
        self.sampler = 'dpmpp'
        self.nextseed = 0
        self.w = userconf.default_width
        self.h = userconf.default_height
        self.t = 0
        self.f = 0
        self.fps = 24
        self.draft = 1
        self.dry = False
        self.img = None
        self.init_img = None

        self._signals = SignalGroup()
        self._gsignals: Dict[str, SignalGroup] = {}
        self._selected_gsignal = None

    def get(self, key, return_zero_on_missing_float_value=False):
        if key in self.__dict__:
            return self.__dict__[key]

        if key in self._signals:
            signal = self._signals[key]
            return signal.data if self.is_array_mode else signal[self.f]
        else:
            # Return zero and create a new signal if it doesn't exist (in array mode)
            if self.is_array_mode:
                self._signals[key] = Signal(np.zeros(self.n), key)
                return self._signals[key].data
            else:
                if return_zero_on_missing_float_value:
                    return 0.0

                raise AttributeError(f"RenderVars has no attribute '{key}'")  # This has caused too many issues

    def __getattr__(self, key):
        """
        Access signals as object attributes.
        """
        return self.get(key)

    def __getitem__(self, key):
        """
        Dict-style access to signals.
        """
        return self.get(key, True)

    def __setattr__(self, key, value):
        self.set_signal(key, value)

    def __contains__(self, name):
        return self.has_signal(name)

    @property
    def w2(self):
        return self.w // 2

    @property
    def h2(self):
        return self.h // 2

    def to_seconds(self, f):
        return np.clip(f, 0, self.n - 1) / self.fps

    def to_frame(self, t):
        return int(np.clip(t * self.fps, 0, self.n - 1))

    @property
    def speed(self):
        # magnitude of x and y
        return sqrt(self.x ** 2 + self.y ** 2)

    # noinspection PyAttributeOutsideInit
    @speed.setter
    def speed(self, value):
        # magnitude of x and y
        speed = self.speed
        if speed > 0:
            self.x = abs(self.x) / speed * value
            self.y = abs(self.y) / speed * value

    @property
    def duration(self):
        return self.n / self.fps

    @property
    def size(self):
        return self.w, self.h

    @property
    def t(self):
        return self.f / self.fps

    @property
    def dt(self):
        return 1 / self.fps

    def zeros(self) -> ndarray:
        return np.zeros(self.n)

    def ones(self, v=1.0) -> ndarray:
        return np.ones(self.n) * v

    def set_fps(self, fps):
        self.fps = fps
        self.session.fps = fps

    def set_n(self, n):
        self.set_frames(n)

    def set_frames(self, n_frames):
        if isinstance(n_frames, int):
            self.n = n_frames
        if isinstance(n_frames, np.ndarray):
            self.n = len(n_frames)

    def set_duration(self, duration):
        self.n = int(duration * self.fps)
        self.resize_signals_to_n()

    def set_n(self, n):
        self.n = n
        self.resize_signals_to_n()

    def resize_signals_to_n(self):
        # Resize all signals (paddding with zeros)
        for s in self._signals.items():
            self._signals[s[0]] = np.pad(s[1], (0, self.n - len(s[1])))

    def resize(self, crop=False):
        """
        Resize the current frame.
        """
        if self.img is None:
            return

        self.img = convert.resize(self.img, self.w, self.h, crop)

    def set_size(self, w=None, h=None, *, frac=64, remote=None, resize=True, crop=False):
        """
        Set the configured target render size,
        and resize the current frame.
        """
        # 1920x1080
        # 1280x720
        # 1024x576
        # 768x512
        # 640x360

        w = w or self.w
        h = h or self.h

        w = int(w)
        h = int(h)

        draft = 1
        draft += jargs.args.draft

        if jargs.args.remote and remote:
            w, h = remote

        self.w = w // self.draft
        self.h = h // self.draft
        self.w = self.w // frac * frac
        self.h = self.h // frac * frac

        if resize and self.img is not None:
            self.resize(crop)
        if resize and self.session.img is not None:
            self.session.resize(self.w, self.h, crop)

    def init_frame(self, f, scalar=1, cleanse_nulls=True):
        """
        Prime the render state for rendering a frame.
        Signals will be loaded and the frame will be resized.
        Null frame images will be replaced with black images.
        """
        rv = self

        if rv.w is None: rv.w = rv.ses.width
        if rv.h is None: rv.h = rv.ses.height

        rv.dry = False
        rv.nextseed = random.randint(0, 2 ** 32 - 1)
        rv.seed = rv.nextseed
        rv.f = int(f)
        rv.ref = 1 / 12 * rv.fps
        rv.tr = rv.t * rv.ref
        rv.session.f = f
        rv.session.load_f()
        rv.session.resize(rv.w, rv.h, crop=False)

        self.resize_signals_to_n()
        rv.is_array_mode = False
        rv.load_signal_values()
        rv.scalar = scalar
        rv.img = rv.session.img
        rv.prev_img = rv.session.res_frame_cv2(rv.f - 1, default=None)
        rv.img_f = rv.img
        if cleanse_nulls:
            if rv.img is None:
                rv.img = np.zeros((rv.h, rv.w, 3), dtype=np.uint8)
            if rv.prev_img is None:
                rv.prev_img = np.zeros((rv.h, rv.w, 3), dtype=np.uint8)

    def get_constants(self):
        """
        Get the constants for the current render configuration.
        """
        from src.party.maths import np0, np01
        n = self.n
        t = np01(n)
        indices = np0(self.n - 1, self.n)

        return n, t, indices

    def clear_signals(self):
        self.unset_signals()
        self._signals.clear()
        self._gsignals.clear()
        self._selected_gsignal = None

    def unset_signals(self):
        for k, v in self._signals.items():
            self.__dict__.pop(k, None)
            self.__dict__.pop(f'{k}s', None)

    def has_signal(self, name):
        return name in self._signals or name in self.__dict__ and not isinstance(self.__dict__[name], int)


    def load_signal_values(self):
        """
        Load the current frame values for each signal into __dict__.
        """
        for name, value in self._signals.items():
            signal = self._signals[name]
            try:
                # log("fetch", name, self.f, len(signal))

                f = self.f
                self.__dict__[f'{name}s'] = signal
                if self.f > len(signal) - 1:
                    self.__dict__[name] = 0
                else:
                    self.__dict__[name] = signal[f]

            except IndexError:
                log(f'rv.set_frame_signals(IndexError): {name} {self.f} {len(signal)}')

    def load_signal_arrays(self):
        """
        Load the current frame values for each signal into __dict__.
        """
        for name, value in self._signals.items():
            self.__dict__[name] = value
            self.__dict__[f'{name}s'] = value


    def select_gsignal(self, name):
        """
        Select the current signal group.
        """
        if self._selected_gsignal is None:
            # We haven't started using signal groups yet, so just set the name for now
            # We will save the set later when we switch to a new set, or finish the render callback
            self._selected_gsignal = name
        else:
            # We are already using signal groups, so save the current group to the group dictionary
            self._gsignals[self._selected_gsignal] = dict(self._signals)
            self._signals.clear()
            self._selected_gsignal = name

            # Ok now we can load the new set
            self._signals.clear()
            if name in self._gsignals:
                # Already exists, load that shit
                self._signals = self._gsignals[name]
            else:
                # Get them in sync
                self._gsignals[name] = self._signals
                pass


    def copy_gframe(self, src_name, dst_name, i):
        dst, is_current, src = self.get_set_src_dst(dst_name, src_name)

        for k, v in src.items():
            if not k in dst:
                from src.party.maths import zero
                dst[k] = zero()
            dst[k][i] = src[k][i]

    @trace_decorator
    def copy_gframes(self, src_name, dst_name, i_start, i_end):
        dst, is_current, src = self.get_set_src_dst(dst_name, src_name)

        for k, v in src.items():
            if k not in dst:
                from src.party.maths import zero
                dst[k] = zero()
            dst[k][i_start:i_end] = src[k][i_start:i_end]

        # if is_current:
        #     self.load_signal_arrays()

    def get_set_src_dst(self, dst_name, src_name):
        if src_name == self._selected_gsignal:
            src = self._signals
        else:
            src = self._gsignals[src_name]

        is_current = dst_name is None or dst_name == self._selected_gsignal
        if is_current:
            dst = self._signals

        elif dst_name not in self._gsignals:
            self._gsignals[dst_name] = dict()
            dst = dst = self._gsignals[dst_name]
        else:
            dst = self._gsignals[dst_name]

        return dst, is_current, src

    def copy_gsignal(self, src_name, dst_name=None):
        """
        Copy the current frame of the source signal to the current frame of the destination signal.
        """
        dst, is_current, src = self.get_set_src_dst(dst_name, src_name)

        for k, v in src.items():
            dst[k] = src[k].copy()

    def set_signal(self, key, value):
        is_signal_assignment = isinstance(value, ndarray) and len(value.shape) == 1
        is_number_assignment = isinstance(value, (float, int)) and not isinstance(value, bool)
        is_array_mode = self.__dict__.get('is_array_mode', False)

        if is_signal_assignment:
            signal = value
        elif is_number_assignment and is_array_mode:
            if key in protected_names:
                self.__dict__[key] = value
                return
            signal = np.ones(self.n) * value
        else:
            self.__dict__[key] = value
            return

        if key in protected_names:
            log(f"set_signal: {key} is protected and cannot be set as a signal. Skipping...")
            # self.__dict__[key] = value
            return

        if self.n > signal.shape[0]:
            log(f"set_signal: {key} signal is too short. Padding with last value...")
            signal = np.pad(signal, (0, self.n - signal.shape[0]), 'edge')
        elif self.n < signal.shape[0]:
            log(f"set_signal: {key} signal is longer than n, extending RenderVars.n to {signal.shape[0]}...")
            self.set_n(signal.shape[0])

        self._signals[key] = signal
        self._signals[f'{key}s'] = signal

    def load_cv2(self, img):
        return convert.load_cv2(img if img is not None else self.session.img)

    def load_pil(self, img):
        return convert.load_pil(img if img is not None else self.session.img)

    def load_pilarr(self, img):
        return convert.load_pilarr(img if img is not None else self.session.img)


    def get_timestamp_string(self, signal_name):
        import datetime
        ret = '''
chapters = PTiming("""
'''
        indices = np.where(self._signals[signal_name] > 0.5)[0]
        for v in indices:
            from src.gui.ryucalc import to_seconds
            time_s = to_seconds(v)
            delta = datetime.timedelta(seconds=time_s)
            s = str(delta)
            if '.' not in s:
                s += '.000000'

            ret += s + "\n"

        ret += '"""'
        return ret

    # region Deprecated
    def set_signal_set(self, name):
        self.select_gsignal(name)

    def copy_set_frame(self, src_name, dst_name, i):
        self.copy_gframe(self, src_name, dst_name, i)

    def copy_set_frames(self, src_name, dst_name, i_start, i_end):
        self.copy_gframes(src_name, dst_name, i_start, i_end)

    def copy_set(self, src_name, dst_name=None):
        self.copy_gsignal(src_name, dst_name)
    # endregion
