"""
effects.py is a simple library for sequencing 'effects'.
"""

from src import renderer
from src.party.easing import QuadOut
from src.party.maths import *

# # All the supported effects
# supported_effects = ['rotl', 'rotr', 'shake', 'in', 'black', 'white', 'hue']  # 'd', 'u', 'l', 'r']
#
# # CONFIGURATION
# # ----------------------------------------
# effect_pool = supported_effects
# # effect_pool = ['none']
# effect_period = 2
# effect_p = 3
# effects = []  # Current effects
#
#
# # code = sorted(effect_pool, key=lambda x: rng())
#
# def set_random_effect():
#     effects.append(choose(supported_effects))
#
#
# def is_effect_frame(v):
#     i = get_effect_cycle(v)
#     return v.f % i == 0
#
#
# def get_effect_cycle(v):
#     return int(v.fps * effect_period / 2)
#
#
# def get_effect_amp(v, p=10):
#     return abs(sin(v.t / effect_period * tau)) ** p
#
#
# def encode(letter):
#     """
#     Returns the command for the letter
#     """
#     hexval = hex(ord(letter))  # Get letter hex value
#     decval = int(hexval, 16)  # convert to base 10 (hexval='0x41' -> 65)
#
#     return code[decval % len(code)]
#
#
# def callback(v, name):
#     if name == 'frame':
#         if is_effect_frame(v) == 0:
#             set_seed(v.nextseed)
#             set_random_effect()
#
#     if name == 'camera':
#         for fx in effects:
#             fx.callback(v, name)
#
#         cycle = get_effect_cycle(v)
#         effect_spike = get_effect_amp(v, effect_p)
#
#         hud(schedule=f'{v.f % cycle + 1}/{cycle}', spike=effect_spike, on=effect, triggered=effect_triggered)
#
#
# class FX:
#     def __init__(self, v, name):
#         self.v = v
#         self.name = name
#         self.triggered = False
#         self.just_triggered = False
#
#     def callback(self, v, name):
#         self.just_triggered = False
#         if not self.triggered and self.get_amp(v) >= 1 - 0.05:
#             self.triggered = True
#             self.just_triggered = True
#
#         if self.triggered and self.get_amp(v) <= 0.05:
#             effects.remove(self)
#
#     def get_amp(self, v):
#         return get_effect_amp(v, effect_p)
#
#     def add_movement_smear(self, v, amp):
#         v.smear *= 1.5 * amp
#         v.cfg *= 1 + 0.1 * amp
#
#
# class RotLFX(FX):
#     def callback(self, v, name):
#         super().callback(v, name)
#         v.r += 10 * self.get_amp(v)
#
#
# class RotRFX(FX):
#     def callback(self, v, name):
#         super().callback(v, name)
#         v.r -= 10 * self.get_amp(v)
#
#
# class ShakeFX(FX):
#     def callback(self, v, name):
#         super().callback(v, name)
#         camera_shake(v, 37.5 * self.get_amp(v))
#         self.add_movement_smear(v, self.get_amp(v))
#
#
# class InFX(FX):
#     def callback(self, v, name):
#         super().callback(v, name)
#         v.z += 0.5 * self.get_amp(v)
#
#
# class BlackFX(FX):
#     def callback(self, v, name):
#         super().callback(v, name)
#         if self.just_triggered:
#             v.save(Image.new('RGBA', v.image.size, (0, 0, 0, 255)))
#
#
# class WhiteFX(FX):
#     def callback(self, v, name):
#         super().callback(v, name)
#         if self.just_triggered:
#             v.save(Image.new('RGBA', v.image.size, (255, 255, 255, 255)))
#
#
# class HueFX(FX):
#     def callback(self, v, name):
#         super().callback(v, name)
#         v.hue += 20 * self.get_amp()
#
#
# class DownFX(FX):
#     def callback(self, v, name):
#         super().callback(v, name)
#         v.y += 45 * self.get_amp()
#         self.add_movement_smear(v, self.get_amp(v))
#
#
# class UpFX(FX):
#     def callback(self, v, name):
#         super().callback(v, name)
#         v.y -= 45 * self.get_amp()
#         self.add_movement_smear(v, self.get_amp(v))
#
#
# class LeftFX(FX):
#     def callback(self, v, name):
#         super().callback(v, name)
#         v.x -= 45 * self.get_amp()
#         self.add_movement_smear(v, self.get_amp(v))
#
#
# class RightFX(FX):
#     def callback(self, v, name):
#         super().callback(v, name)
#         v.x += 45 * self.get_amp()
#         self.add_movement_smear(v, self.get_amp(v))

class FX:
    def __init__(self):
        self.signal = None
        self.activation = None

current_fx = 0
all_fxs = []

rv = renderer.rv

def clear():
    global all_fxs
    all_fxs = []

def interval_1(seconds=1, start=0):
    interval = seconds * rv.fps
    start = start * rv.fps
    n = rv.n
    ret = np.zeros(n)
    ret[start::interval] = 1
    return ret

def prepare_fxs(n_fx):
    global all_fxs
    all_fxs = []
    for i in range(n_fx):
        fx = FX()
        fx.signal = np.zeros(rv.n)
        fx.activation = np.zeros(rv.n)
        all_fxs.append(fx)

def random(n_fx, activation_signal, threshold=0.75):
    prepare_fxs(n_fx)

    above_thresh_indices = np.flatnonzero(activation_signal >= threshold)
    for i, f in enumerate(above_thresh_indices):
        all_fxs[rngi(n_fx)].activation[f] = 1

def cycle(n_fx, activation_signal, threshold=0.75):
    prepare_fxs(n_fx)

    above_thresh_indices = np.flatnonzero(activation_signal >= threshold)
    fx_index = 0
    for i, f in enumerate(above_thresh_indices):
        all_fxs[fx_index].activation[f] = 1
        fx_index += 1
        if fx_index >= n_fx:
            fx_index = 0

def prepare_fx(ones, ease, mask):
    global current_fx
    fx = all_fxs[current_fx]
    current_fx += 1
    ones = np.ones(rv.n) * ones
    if mask is None:
        mask = np.ones(rv.n)
    if ease is None:
        ease = QuadOut()

    f_elapsed = sys.maxsize
    for f in range(rv.n):
        if fx.activation[f] == 1:
            fx.signal[f] = 1
            f_elapsed = 0
        else:
            f_elapsed += 1

        t = f_elapsed / rv.fps
        t = clamp(t, 0, ease.duration)
        fx.signal[f] = ease.ease(t, reverse=True)


    return fx, ones, ease, mask

def pulse_r(rz=30, ease=None, mask=None):
    fx, rz, ease, mask = prepare_fx(rz, ease, mask)
    rv.rz += fx.signal * rz * mask
    rv.r += fx.signal * rz * mask

def pulse_z(z=50, ease=None, mask=None):
    fx, z, ease, mask = prepare_fx(z, ease, mask)
    rv.z += fx.signal * z * mask

def shake():
    pass

def shake(s_total):
    pass

def flash():
    pass

def restore(offset=-1):
    pass
