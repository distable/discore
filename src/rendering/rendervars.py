import random
import numpy as np
import jargs
import userconf
from numpy import ndarray
from dataclasses import dataclass
from math import sqrt
from src.classes import convert
from src.classes.convert import cv2pil

# These are values that cannot be used as signal names
# Because they are either special or cannot change during rendering
protected_names = [
    'session', 'signals',
    'w', 'h', 'w2', 'h2',
    't', 'f', 'dt', 'ref', 'tr', 'len',
    'n', 'scalar', 'draft' ]

@dataclass
class RenderVars:
    """
    Render variables supported by the renderer
    This provides a common interface for our libraries to use.
    """

    def __init__(self):
        self.is_array_mode = False
        self.n = 1000
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
        self.init()
        self.trace = "__init__"

    def __getattr__(self, key):
        if key not in self.__dict__:
            if self.__dict__['is_array_mode']:
                self.__dict__[key] = 0
            else:
                return 0
        return self.__dict__[key]
        # self.__dict__[key] = 0

    def __setattr__(self, key, value):
        if self.__dict__.get('is_array_mode', False) and \
                isinstance(value, (float, int)) and \
                not isinstance(value, bool) and \
                key not in protected_names:
            value = np.ones(self.n) * value

        if isinstance(value, ndarray) and len(value.shape) == 1:
            self.save_signals(**{key: value})
            self.__dict__[key] = value
        else:
            self.__dict__[key] = value


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

    def init(self):
        def zero() -> ndarray:
            return np.zeros(self.n)

        def one(v=1.0) -> ndarray:
            return np.ones(self.n) * v

        for s in self.signals.items():
            self.signals[s[0]] = zero()

        x, y, z = zero(), zero(), zero()
        r, rx, ry, rz = zero(), zero(), zero(), zero()
        d, chg, cfg, seed, sampler = one(), one(), one(16.5), zero(), 'euler-a'
        nguide, nsp = zero(), zero()
        smear = zero()
        hue, sat, val = zero(), zero(), zero()
        brightness, saturation, contrast = zero(), zero(), zero()
        ripple_period, ripple_amplitude, ripple_speed = zero(), zero(), zero()
        smear = zero()
        seed_atten_ccg1, seed_atten_ccg2, seed_atten_ccg3 = zero(), zero(), zero()
        seed_atten_time = one()

        # Apply these changes
        dic = dict(locals())
        dic.pop('self')
        self.save_signals(**dic)
        self.load_signals()

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

    def set_n(self, n):
        self.n = n

        # Resize all signals (paddding with zeros)
        for s in self.signals.items():
            self.signals[s[0]] = np.pad(s[1], (0, self.n - len(s[1])))

    def set_size(self, w, h, *, frac=64, remote=(768, 512), resize=True, crop=False):
        draft = 1
        draft += jargs.args.draft

        jargs.args.remote = True
        if jargs.args.remote and remote:
            w, h = remote

        self.w = w // self.draft
        self.h = h // self.draft
        self.w = self.w // frac * frac
        self.h = self.h // frac * frac

        # image resize (pillow)
        if resize and self.session.img.shape[0] != self.h and self.session.img.shape[1] != self.w:
            impil = cv2pil(self.session.img)
            impil = impil.resize((self.w, self.h))
            self.session.img = impil
        if crop:
            # center anchored crop
            self.session.img = self.session.img.crop((self.w // 2 - self.w // 2, self.h // 2 - self.h // 2, self.w // 2 + self.w // 2, self.h // 2 + self.h // 2))

    def start_frame(self, f, scalar=1):
        rv = self

        if rv.w is None: rv.w = rv.ses.width
        if rv.h is None: rv.h = rv.ses.height

        rv.img = rv.session.img
        rv.dry = False
        rv.nextseed = random.randint(0, 2 ** 32 - 1)
        rv.f = int(f)
        rv.t = f / rv.fps
        if rv.w: rv.w2 = rv.w / 2
        if rv.h: rv.h2 = rv.h / 2
        rv.dt = 1 / rv.fps
        rv.ref = 1 / 12 * rv.fps
        rv.tr = rv.t * rv.ref

        rv.load_values()
        rv.scalar = scalar
        if rv.img is None:
            rv.img = np.zeros((rv.h, rv.w, 3), dtype=np.uint8)

    def get_constants(self):
        from src.party.maths import np0, np01
        n = self.n
        t = np01(n)
        indices = np0(self.n - 1, self.n)

        return n, t, indices

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

        for name, v in kwargs.items():
            self.__dict__[name] = v
            if isinstance(v, ndarray):
                if name in protected_names:
                    print(f"set_frame_signals: {name} is protected and cannot be set as a signal. Skipping...")
                    continue

                if self.n > v.shape[0]:
                    print(f"set_frame_signals: {name} signal is too short. Padding with last value...")
                    v = np.pad(v, (0, self.n - v.shape[0]), 'edge')
                elif self.n < v.shape[0]:
                    print(f"set_frame_signals: {name} signal is longer than n, extending RenderVars.n to {v.shape[0]}...")
                    self.set_n(v.shape[0])

                # print(f"SET {name} {v}")
                self.signals[name] = v
                self.__dict__[f'{name}s'] = v

        # TODO update len
        self.n = self.n
        for name, v in self.signals.items():
            self.n = max(self.n, v.shape[0])


    def load_values(self):
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
    def load_signals(self):
        for name, value in self.signals.items():
            self.__dict__[name] = value
            self.__dict__[f'{name}s'] = value

    def load_cv2(self, img):
        return convert.load_cv2(img if img is not None else self.session.img)

    def load_pil(self, img):
        return convert.load_pil(img if img is not None else self.session.img)

    def load_pilarr(self, img):
        return convert.load_pilarr(img if img is not None else self.session.img)
