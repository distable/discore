import random
import numpy as np
import jargs
import userconf
from numpy import ndarray
# from dataclasses import dataclass
from math import sqrt
from src.classes import convert
from src.classes.convert import cv2pil, pil2cv
from src.lib.printlib import trace_decorator

# These are values that cannot be used as signal names
# Because they are either special or cannot change during rendering
protected_names = [
    'img',
    'session', 'signals',
    'w', 'h', 'w2', 'h2',
    't', 'f', 'dt', 'ref', 'tr', 'len',
    'n', 'scalar', 'draft']

# @dataclass
class RenderVars:
    """
    Render variables supported by the renderer
    This provides a common interface for our libraries to use.
    """

    def __init__(self):
        self.is_array_mode = False
        self.n = 0
        self.prompt = ""
        self.promptneg = ""
        self.nprompt = None
        self.w = 640
        self.h = 448
        self.scalar = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.r = 0
        self.rx = 0
        self.ry = 0
        self.rz = 0
        self.smear = 0
        self.hue = 0
        self.sat = 0
        self.val = 0
        self.steps = 20
        self.d = 1.1
        self.chg = 1
        self.cfg = 16.5
        self.seed = 0
        self.sampler = 'euler-a'
        self.nguide = 0
        self.nsp = 0
        self.nextseed = 0
        self.t = 0
        self.f = 0
        self.w2 = 0
        self.h2 = 0
        self.dt = 1 / 24
        self.ref = 1 / 12 * 24
        self.draft = 1
        self.dry = False
        self.signals = {}
        self.signal_sets = {}
        self.current_signal_set = None
        self.img = None
        self.init_img = None
        self.init()
        self.trace = "__init__"

    @property
    def speed(self):
        # magnitude of x and y
        return sqrt(self.x ** 2 + self.y ** 2)

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

    def zeros(self) -> ndarray:
        return np.zeros(self.n)

    def ones(self, v=1.0) -> ndarray:
        return np.ones(self.n) * v


    def init(self):
        def zero() -> ndarray:
            return np.zeros(self.n)

        def one(v=1.0) -> ndarray:
            return np.ones(self.n) * v

        for s in self.signals.items():
            self.signals[s[0]] = zero()

        # x, y, z = zero(), zero(), zero()
        # r, rx, ry, rz = zero(), zero(), zero(), zero()
        # d, chg, cfg, seed, sampler = one(), one(), one(16.5), zero(), 'euler-a'
        # nguide, nsp = zero(), zero()
        # smear = zero()
        # hue, sat, val = zero(), zero(), zero()
        # brightness, saturation, contrast = zero(), zero(), zero()
        # ripple_period, ripple_amplitude, ripple_speed = zero(), zero(), zero()
        # smear = zero()
        # seed_atten_ccg1, seed_atten_ccg2, seed_atten_ccg3 = zero(), zero(), zero()
        # seed_atten_time = one()

        # Apply these changes
        dic = dict(locals())
        dic.pop('self')
        self.save_signals(**dic)
        self.load_signal_arrays()

        self.w = userconf.default_width
        self.h = userconf.default_height

    def set_fps(self, fps):
        self.fps = fps
        self.dt = 1 / fps
        self.ref = 1 / 12 * fps
        self.session.fps = fps

    def set_frames(self, n_frames):
        if isinstance(n_frames, int):
            self.n = n_frames
        if isinstance(n_frames, np.ndarray):
            self.n = len(n_frames)

    def set_duration(self, duration):
        self.n = int(duration * self.fps)

        # Resize all signals (paddding with zeros)
        for s in self.signals.items():
            self.signals[s[0]] = np.pad(s[1], (0, self.n - len(s[1])))
        self.load_signal_arrays()

    def set_n(self, n):
        self.n = n

        # Resize all signals (paddding with zeros)
        for s in self.signals.items():
            self.signals[s[0]] = np.pad(s[1], (0, self.n - len(s[1])))
        self.load_signal_arrays()

    def resize(self, crop=False):
        if self.img is None:
            return

        if crop:
            # center anchored crop
            im = cv2pil(self.img)
            im = im.crop((self.w // 2 - self.w // 2, self.h // 2 - self.h // 2, self.w // 2 + self.w // 2, self.h // 2 + self.h // 2))
            self.img = pil2cv(im)
        else:
            im = cv2pil(self.img)
            im = im.resize((self.w, self.h))
            self.img = pil2cv(im)


    def set_size(self, w=None, h=None, *, frac=64, remote=None, resize=True, crop=False):
        # 1920x1080
        # 1280x720
        # 1024x576
        # 768x512
        # 640x360

        w = w or self.w
        h = h or self.h

        draft = 1
        draft += jargs.args.draft

        if jargs.args.remote and remote:
            w, h = remote

        self.w = w // self.draft
        self.h = h // self.draft
        self.w = self.w // frac * frac
        self.h = self.h // frac * frac

        if resize:
            self.resize(crop)

    def start_frame(self, f, scalar=1):
        rv = self

        if rv.w is None: rv.w = rv.ses.width
        if rv.h is None: rv.h = rv.ses.height

        rv.dry = False
        rv.nextseed = random.randint(0, 2 ** 32 - 1)
        rv.f = int(f)
        rv.t = f / rv.fps
        if rv.w: rv.w2 = rv.w / 2
        if rv.h: rv.h2 = rv.h / 2
        rv.dt = 1 / rv.fps
        rv.ref = 1 / 12 * rv.fps
        rv.tr = rv.t * rv.ref

        rv.load_signal_values()
        rv.scalar = scalar
        rv.img = rv.session.img
        if rv.img is None:
            rv.img = np.zeros((rv.h, rv.w, 3), dtype=np.uint8)
        if rv.img.shape[0] != rv.h or rv.img.shape[1] != rv.w:
            impil = cv2pil(rv.img)
            impil = impil.resize((rv.w, rv.h))
            rv.img = pil2cv(impil)
        rv.img_f = rv.img

    def get_constants(self):
        from src.party.maths import np0, np01
        n = self.n
        t = np01(n)
        indices = np0(self.n - 1, self.n)

        return n, t, indices

    def reset_signals(self):
        # Set all signals to zero
        for k, v in self.signals.items():
            self.signals[k] = np.zeros(self.n)

        self.load_signal_arrays()
        self.signals.clear()
        self.signal_sets.clear()
        self.current_signal_set = None

    def has_signal(self, name):
        return name in self.signals

    def save_signals(self, **kwargs):
        if len(kwargs) == 0:
            kwargs = dict(
                    x=self.x, y=self.y, z=self.z, r=self.r,
                    hue=self.hue, sat=self.sat, val=self.val,
                    smear=self.smear,
                    d=self.d, chg=self.chg, cfg=self.cfg,
                    seed=self.seed, sampler=self.sampler,
                    nguide=self.nguide, nsp=self.nsp,
                    brightness=self.brightness, saturation=self.saturation, contrast=self.contrast,
                    ripple_period=self.ripple_period, ripple_amplitude=self.ripple_amplitude, ripple_speed=self.ripple_speed,
                    music=self.music, drum=self.drum, bass=self.bass, piano=self.piano, vocal=self.vocal, voice=self.voice
            )

        for name, signal in kwargs.items():
            self.__dict__[name] = signal
            if isinstance(signal, ndarray):
                if name in protected_names:
                    print(f"set_frame_signals: {name} is protected and cannot be set as a signal. Skipping...")
                    continue

                if self.n > signal.shape[0]:
                    print(f"set_frame_signals: {name} signal is too short. Padding with last value...")
                    signal = np.pad(signal, (0, self.n - signal.shape[0]), 'edge')
                elif self.n < signal.shape[0]:
                    print(f"set_frame_signals: {name} signal is longer than n, extending RenderVars.n to {signal.shape[0]}...")
                    self.set_n(signal.shape[0])

                # print(f"SET {name} {v}")
                self.signals[name] = signal
                self.__dict__[name] = signal
                self.__dict__[f'{name}s'] = signal

        # TODO update len
        # self.n = self.n
        # for name, v in self.signals.items():
        #     self.n = max(self.n, v.shape[0])


    def load_signal_values(self):
        dic = self.__dict__.copy()
        for name, value in dic.items():
            # if isinstance(value, ndarray):
            #     self.signals[name] = value

            if name in self.signals:
                signal = self.signals[name]
                try:
                    # print("fetch", name, self.f, len(signal))

                    f = self.f
                    self.__dict__[f'{name}s'] = signal
                    if self.f > len(signal) - 1:
                        self.__dict__[name] = 0
                    else:
                        self.__dict__[name] = signal[f]

                except IndexError:
                    print(f'rv.set_frame_signals(IndexError): {name} {self.f} {len(signal)}')

    # Same function as above but sets the whole signal
    def load_signal_arrays(self):
        for name, value in self.signals.items():
            self.__dict__[name] = value
            self.__dict__[f'{name}s'] = value

    def set_signal_set(self, name):
        if self.current_signal_set is None:
            # We haven't started using signal sets yet, so just set the name for now
            # We will save the set later when we switch to a new set, or finish the render callback
            self.current_signal_set = name
        else:
            # We are already using signal sets, so save the current set to the set dictionary
            self.signal_sets[self.current_signal_set] = dict(self.signals)
            self.signals.clear()
            self.current_signal_set = name

            # Ok now we can load the new set
            self.signals.clear()
            if name in self.signal_sets:
                # Already exists, load that shit
                self.signals = self.signal_sets[name]
            else:
                # Get them in sync
                self.signal_sets[name] = self.signals
                pass

    def copy_set_frame(self, src_name, dst_name, i):
        dst, is_current, src = self.get_set_src_dst(dst_name, src_name)

        for k, v in src.items():
            if not k in dst:
                from src.party.maths import zero
                dst[k] = zero()
            dst[k][i] = src[k][i]

        if is_current:
            self.load_signal_arrays()

    @trace_decorator
    def copy_set_frames(self, src_name, dst_name, i_start, i_end):
        dst, is_current, src = self.get_set_src_dst(dst_name, src_name)

        for k, v in src.items():
            if not k in dst:
                from src.party.maths import zero
                dst[k] = zero()
            dst[k][i_start:i_end] = src[k][i_start:i_end]

        if is_current:
            self.load_signal_arrays()

    def get_set_src_dst(self, dst_name, src_name):
        if src_name == self.current_signal_set: src = self.signals
        else: src = self.signal_sets[src_name]
        is_current = dst_name is None or dst_name == self.current_signal_set
        if is_current: dst = self.signals
        elif dst_name not in self.signal_sets:
            self.signal_sets[dst_name] = dict()
            dst = dst = self.signal_sets[dst_name]
        else: dst = self.signal_sets[dst_name]
        return dst, is_current, src

    def copy_set(self, src_name, dst_name=None):
        dst, is_current, src = self.get_set_src_dst(dst_name, src_name)

        for k, v in src.items():
            dst[k] = src[k].copy()

        if is_current:
            self.load_signal_arrays()


    def __getattr__(self, key):
        if key not in self.__dict__:
            if not self.__dict__['is_array_mode']:
                print(f"RenderVars: Assigning 0 to {key}")
                self.__dict__[key] = 0
            else:
                print(f"RenderVars: Assigning [0] to {key}")
                self.__dict__[key] = np.zeros(self.n)
        return self.__dict__[key]
        # self.__dict__[key] = 0

    def __setattr__(self, key, value):
        self.set_signal(key, value)

    def set_signal(self, key, value):
        if self.__dict__.get('is_array_mode', False) and \
                isinstance(value, (float, int)) and \
                not isinstance(value, bool) and \
                key not in protected_names:
            value = np.ones(self.n) * value
        if isinstance(value, ndarray) and len(value.shape) == 1:
            # self.__dict__[key] = value
            self.save_signals(**{key: value})
        else:
            self.__dict__[key] = value


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
        indices = np.where(self.signals[signal_name] > 0.5)[0]
        for v in indices:
            from src.gui.ryusig import to_seconds
            time_s = to_seconds(v)
            delta = datetime.timedelta(seconds=time_s)
            s = str(delta)
            if '.' not in s:
                s += '.000000'

            ret += s + "\n"

        ret += '"""'
        return ret

    def to_seconds(self, frame):
        return np.clip(frame, 0, self.n - 1) / self.fps

    def to_frame(self, t):
        return int(np.clip(t * self.fps, 0, self.n - 1))
