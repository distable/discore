"""
This is your toolbox
This is your playground
Go work
Go wild
Make art
Make love
"""

import math
import os
from colorsys import hsv_to_rgb
from pathlib import Path

import cv2
import PIL
from PIL import Image
from skimage import exposure
from tqdm import tqdm, trange

import jargs
from src import plugins
from src.classes import convert, paths
from src.classes.convert import load_pil, pil2cv, save_png, save_jpg
from src.lib import devices
from src.lib.printlib import trace
from src.party import maths
from src.party.maths import *
from src.party.partyutils import hex_to_rgb
from src.party.trick_cache import TrickCache
from src.plugins import plugfun, plugfun_img
from src.rendering import hud
from src.rendering.rendervars import RenderVars

# CONFIG
# if you're feeling courageous
# ----------------------------
entropy_val = 4
entropy_sat = 0.5

# STATE
# do not touch these unless
# you are ready to go to jail
# ----------------------------
last_frame_prompt = None
cchg = 0
px, py = 0, 0

# Models --------------------
midas = None
midas_transforms = None

border_cache = TrickCache()
left_mask_cache = TrickCache()
right_mask_cache = TrickCache()
top_mask_cache = TrickCache()
down_mask_cache = TrickCache()
perp_mask_cache = TrickCache()
grain_cache = TrickCache()
sp_cache = TrickCache()
vignette_cache = TrickCache()

rv = renderer.rv
session = renderer.session

cv2.ocl.useOpenCL()

rgb_black = np.array([0, 0, 0], dtype='uint8')
rgb_white = np.array([255, 255, 255], dtype='uint8')
rgba_black = np.array([0, 0, 0, 255], dtype='uint8')
rgba_white = np.array([255, 255, 255, 255], dtype='uint8')


ENABLE_FLOW_PRECOMPUTE = False
DEFAULT_FLOW_METHOD = 'Farneback'


# if jargs.args.remote:
#     DEFAULT_FLOW_METHOD = 'unimatch'

def rv_set(original_param, im):
	if isinstance(original_param, RenderVars):
		rv.img = im
	return im


@trace_decorator
def get_jutter_img(name, rand):
	face = session.res_cv2(name, mode='fit')
	face = mat2d(face, x=rng(-5, 5) * rand,
	             y=rng(-5, 5) * rand,
	             z=rng(-0.05, 0.05) * rand,
	             r=rng(-25, 25) * rand)
	return face


def load_cv2(img=None, size=None):
	if isinstance(img, RenderVars) or img is None:
		return rv.img
	if isinstance(img, str) and not Path(img).is_absolute():
		img = rv.session.res(img)

	return convert.load_cv2(img, size)


def set_euler_schedule():
	return schedule_v1v2('euler-a', 'euler', 0.8, hudname='sampler')


last_seed = None


@trace_decorator
def get_interval_seed(interval=1.0):
	interval *= rv.fps
	interval = int(interval)

	global last_seed
	if last_seed is None or rv.f % interval == 0:
		last_seed = rngi(0, 1000000)

	return last_seed


def add_entropy_shake(pct=(0.046, 0.078)):
	# # this used to be *8 instead of a pct
	# v.x += v.x * pct * (1 + v.d) * perlin(v.t + 100, 100)
	# v.y += v.y * pct * (1 + v.d) * perlin(v.t + 1000, 80)
	f = rv.f
	pct = val_or_range(pct)

	# entropy shake
	rv.x += copysign(rv.w * pct / rv.dt, rng() - 0.5)
	rv.y += copysign(rv.h * pct / rv.dt, rng() - 0.5)


# print(v.h * pct / v.dt)


def add_entropy_cc(scale=1):
	rv.hue += lerp(0.0, 20 * scale, jcurve(abs(perlin(rv.t, 0.35)), 0.3))
	rv.sat += lerp(-entropy_sat * scale, entropy_sat * scale,
	               perlin(rv.t + 00, 32))
	rv.val += lerp(-entropy_val * scale, entropy_val * scale,
	               perlin(rv.t + 30, 32))


def add_rotor_preset():
	rv.cfg *= schedule(1, 1, 1.15, 1, 1, 2, 1, 1, 1.15)
	rv.chg *= schedule(1, 1, 1, 1, 1, 1, 1, 1.1, 1, 1, 1, 1, 1.175)
	rv.nsp *= schedule(1, 1, 1, 0.5, 1, 1, 1, 2, 1)
	rv.sampler = set_euler_schedule(rv)
	rv.steps *= schedule(0.5, 1, 1.5)
	rv.smear = 0.75
	rv.val = abs(rv.val)


def set_d_calc():
	"""
	Calculate the amount of camera change
	"""
	dt = 1 / rv.fps
	rv.x /= dt
	rv.y /= dt
	rv.z /= dt
	rv.r /= dt

	rv.d *= sum((clamp01(rcurve(ilerp(0, 0.525, rv.z), 0.275)),
	             scurve(ilerp(0, 15, abs(rv.r)), 0.11),
	             abs(rv.x / rv.w),
	             abs(rv.y / rv.h),
	             ))
	rv.d *= rv.scalar
	rv.d = clamp(rv.d, 0.05, 1)

	rv.x *= dt
	rv.y *= dt
	rv.z *= dt
	rv.r *= dt


def set_d_chg():
	"""
	The good stuff
	"""
	rv.nguide *= lerp(0.08, 0.2, rv.d)
	rv.nsp *= lerp(0, 0.006, rcurve(rv.d, 0.4))
	rv.chg *= rcurve(clamp(rv.d, 0.0, 0.85), 0.5)
	rv.chg *= clamp(rv.chg, 0.11, 0.485)


def add_chg_schedule():
	if rv.f % 2 == 0: rv.chg *= 0.85
	if rv.f % 4 == 0: rv.chg *= 0.85
	if rv.f % 6 == 0: rv.chg *= 0.85
	if rv.f % 8 == 0: rv.chg *= 0.85


def set_seed_smear():
	"""
	smear=1 is every
	frame
	smear=0.5 is every .5 second
	smear=0.0 disables smear
	smearing corresponds to fixed seed
	"""
	v1 = rv.nextseed
	v2 = str_to_seed(rv.session.name)

	sv = schedule_v1v2(v1, v2, rv.smear, 'smear')
	smeared = sv == v2

	rv.seed = v1
	if smeared:
		rv.seed = v2


def get_color_img(name_or_hex):
	"""
	Return a cv2 image of a solid color
	args:
		name_or_hex: name of color or hex code in #RRGGBB format
	"""
	color = (0, 0, 0)
	if name_or_hex.startswith('#'):
		color = hex_to_rgb(name_or_hex)
	else:
		if name_or_hex in colors:
			color = colors[name_or_hex]
		else:
			print(
				f'get_color_img: Color {name_or_hex} not found, defaulting to black')

	ret = np.zeros((rv.h, rv.w, 3), np.uint8)
	ret[:] = color
	return ret


def get_black_img():
	return get_color_img('#000000')


def get_white_img():
	return get_color_img('#FFFFFF')


def schedule_01(t, frames=None, hudname=None):
	return schedule_v1v2(0, 1, t, hudname)


def schedule_v1v2(v1, v2, t, hudname=None):
	"""
	Schedule v1 or v2 based on t (0-1) and f (frame number)
	0 = all v1
	1 = all v2
	0.5 = v1 on even frames, v2 on odd frames
	0.25 = v1 on even frames, v2 on odd frames, but v1 is 2x as long
	0.75 = v1 on even frames, v2 on odd frames, but v2 is 2x as long
	etc.
	"""
	t = clamp01(t)

	every, v = get_v1v2_schedule_stride(t, v1, v2)

	if hudname is not None:
		hud.hud(**{
			hudname: t,
			'schedule': f'{(f % every)} / {every}',
			'on': v == v2
		})

	return v


def get_v1v2_schedule_stride(t, v1=0, v2=1):
	f = rv.f
	span = rv.fps / 2
	every1 = span - int((1 - abs(t * 2 - 1)) * span)
	every2 = span - int((1 - abs((1 - t) * 2 - 1)) * span)
	every1 = clamp(every1, 2, span)
	every2 = clamp(every2, 2, span)
	every = 256
	v = v1
	if t == 0:
		v = v1
	elif t == 1:
		v = v2
	elif t <= 0.5:
		every = every1
		spike = f % every1 == 0
		v = v1
		if spike:
			v = v2
	elif t > 0.5:
		every = every2
		spike = f % every2 == 0
		v = v2
		if spike:
			v = v1
	return every, v


def add_chg_cheap_inertia(v):
	global cchg
	if v.chg > cchg:
		cchg = v.chg
	else:
		v.chg = cchg
		cchg *= 1 - v.dt * 3


def move_towards_depth(v):
	"""
	Move towards the deepest area in the camera (with depth estimation)
	"""

	# TODO it would be great if we could get the depth directly from elsewhere
	if v.depth is not None:
		depth_pil = v.session.run('depth')
		depth = np.array(depth_pil)

		# Take only the R channel (is all the same, it's a grayscale image)
		depth = depth[:, :, 0]

		# Get the position of max depth
		max_depth = np.unravel_index(np.argmax(depth), depth.shape)
		x = max_depth[1]
		y = max_depth[0]

		# The goal offset from center
		goal_offset = (v.max_depth_goal[0] - v.w2, v.max_depth_goal[1] - v.h2)
	else:
		x = v.w2
		y = v.h2
		goal_offset = (sin(v.t) * v.w2, 0)
		print(
			"move_towards_depth: no depth estimation available, default to sin/cos")

	if not v.max_depth_pos:
		v.max_depth_pos = (x, y)
		v.max_depth_goal = goal_offset
	else:
		v.max_depth_pos = v.max_depth_pos
		v.max_depth_goal = goal_offset

	# Move towards the max depth
	smooth_time = 0.1
	max_time = 1.0
	v.x, v.xvel = maths.smooth_damp(v.xvel, v.max_depth_goal[0], v.x_vel,
	                                smooth_time, max_time, v.dt)
	v.y, v.yvel = maths.smooth_damp(v.yvel, v.max_depth_goal[0], v.y_vel,
	                                smooth_time, max_time, v.dt)


def draw_circles(img, x, y, opacity, size, area, count, falloff=0.0,
                 hsv=(0, 0, 1), mask=None):
	"""
	Draw circles onto the image to guide the image generator.
	"""
	if rv.dry: return
	area = val_or_range(area)

	_opacity = val_or_range(opacity)

	# Apply falloff (fade with distance more the higher the falloff)
	# if falloff > 0:
	#     _size = lerp(_size, _size * distance, falloff)
	#     _opacity = lerp(_opacity, _opacity * distance, falloff)

	# if isinstance(mask, np.ndarray):
	#     _opacity *= mask[_y, _x]

	orig = load_cv2(img)
	im = load_cv2(img)
	for i in range(int(count)):
		# perfect circle!
		theta = rng(0, 2 * math.pi)
		cx = math.cos(theta)
		cy = math.sin(theta)
		distance = math.sqrt(cx ** 2 + cy ** 2)

		_x = int(area * cx + x)
		_y = int(area * cy + y)
		_size = val_or_range(size)
		# Range 0-1
		_hue = val_or_range(hsv[0])
		_sat = val_or_range(hsv[1])
		_val = val_or_range(hsv[2])
		_r, _g, _b = hsv_to_rgb(_hue, _sat, _val)

		cv2.circle(im,
		           (_x, _y),
		           int(_size),
		           (int(_r * 255), int(_g * 255), int(_b * 255)),
		           -1)

	im = cv2.addWeighted(orig, 1 - _opacity, im, _opacity, 0)

	return rv_set(img, im)


@trace_decorator
def draw_image(v, img_guide, opacity=0.1):
	"""
	Blend an image onto the current frame to inprint its shape or as a detail generator.
	"""
	if v.dry: return

	if isinstance(img_guide, str) or isinstance(img_guide, Path):
		img_guide = cv2.imread(str(img_guide))

	# print(img_guide)

	im = v.img
	im_guide = np.array(img_guide)
	im = cv2.addWeighted(im, 1 - opacity, im_guide, opacity, 0)
	# print(f'draw_image({img_guide}, {opacity})')

	v.img = im


def draw_consistency_iter(v, history=3, fn_curve=None):
	"""
	draw_consistency_dots_n(v, 3)
	draw_consistency_dots_n(v, 6)
	draw_consistency_dots_n(v, [3, 6, 9])
	draw_consistency_dots_n(v, [2, 4, 8, 12])
	draw_consistency_dots_n(v, [3, 8, 16])
	"""
	if isinstance(history, int):
		history = list(range(1, history + 1))

	history.sort()
	max_dist = max(history)

	for i in reversed(history):  # Furthest to closest
		f = v.session.f - i
		if f < 0: continue

		t = i / max_dist
		if callable(fn_curve):
			t = fn_curve(t)

		yield f, t


def draw_consistency_n(v, img, n=5):
	"""
	Blend the past n frames onto the current frame to increase coherence.
	"""
	if v.dry: return

	for f, t in draw_consistency_iter(v, n - 1):
		im = v.session.res_frame(f)
		if Path(im).exists():
			draw_image(v, im, opacity=lerp(.1, .5, t))


def draw_consistency_dots(v, f_behind=3):
	draw_dot_matrix(v,
	                spacing=13,
	                radius=4,
	                hsv=v.session.res_frame(max(0, v.session.f - f_behind)),
	                opacity=[.2, .8],
	                dropout=0.05,
	                noise=1)


def draw_consistency_dots_n(v, history=3, fn_curve=None):
	"""
	draw_consistency_dots_n(v, 3)
	draw_consistency_dots_n(v, 6)
	draw_consistency_dots_n(v, [3, 6, 9])
	draw_consistency_dots_n(v, [2, 4, 8, 12])
	draw_consistency_dots_n(v, [3, 8, 16])
	"""
	for f, t in draw_consistency_iter(v, history, fn_curve):
		t1 = 1 - t
		draw_dot_matrix(v,
		                spacing=lerp(13, 8, t),
		                radius=lerp(4, 8, t),
		                hsv=v.session.res_frame(f),
		                opacity=[.7 * t1, .9 * t1],
		                dropout=0.05 * t,
		                noise=0)


@trace_decorator
def draw_dot_matrix(v, spacing, radius, hsv, opacity, dropout=0., noise=0,
                    mask=None, op='blend'):
	"""
	Draw a dot matrix onto the image.
	"""
	if v.dry: return
	if hsv is None: return

	w = v.session.img.shape[1]
	h = v.session.img.shape[0]

	# hsv can be either (h,s,v) or a rgb cv2 image that we sample from
	if isinstance(hsv, str) or isinstance(hsv, Path):
		hsv = cv2.imread(Path(hsv).expanduser().resolve().as_posix())
		hsv = cv2.resize(hsv, (w, h), interpolation=cv2.INTER_NEAREST)
		if hsv is None:
			hsv = 0, 0, 0
			print(f'Could not load image {hsv}')
		else:
			hsv = hsv[..., ::-1]  # BGR to RGB

	if isinstance(opacity, str) or isinstance(opacity, Path):
		opacity = cv2.imread(Path(opacity).expanduser().resolve().as_posix())
		opacity = cv2.resize(opacity, (w, h), interpolation=cv2.INTER_NEAREST)
		if opacity is None:
			opacity = 1
			print(f'Could not load image {opacity}')
		else:
			opacity = opacity[..., ::-1]  # BGR to RGB

	spacing = val_or_range(spacing)
	spacing = int(spacing)

	img = v.session.img
	dst = np.zeros_like(img)
	for x in range(0, v.session.width - 1, spacing):
		for y in range(0, v.session.height - 1, spacing):
			if rng() < dropout:
				continue

			x = x + rng(-noise, noise)
			y = y + rng(-noise, noise)
			x = int(x)
			y = int(y)
			x = clamp(x, 0, v.session.width - 1)
			y = clamp(y, 0, v.session.height - 1)

			if isinstance(hsv, np.ndarray):
				_r, _g, _b = hsv[y, x]
				_r = int(_r)
				_g = int(_g)
				_b = int(_b)
			else:
				_hue = val_or_range(hsv[0])
				_sat = val_or_range(hsv[1])
				_val = val_or_range(hsv[2])
				_r, _g, _b = hsv_to_rgb(_hue, _sat, _val)

				_r = int(_r * 255)
				_g = int(_g * 255)
				_b = int(_b * 255)

			_opacity = 1.0
			if isinstance(opacity, np.ndarray):
				_opacity = opacity[y, x]
			else:
				_opacity = val_or_range(opacity)

			if mask is not None:
				if isinstance(mask, np.ndarray):
					_opacity *= mask[y, x]
				elif isinstance(mask, PIL.Image):
					_opacity *= mask.getpixel((x, y))[0] / 255

			_radius = val_or_range(radius)
			_radius = int(_radius)

			if op == 'blend' or op is None:
				copy = img.copy()
				cv2.circle(copy, (x, y), _radius, (_r, _g, _b), -1)
				img = cv2.addWeighted(img, 1 - _opacity, copy, _opacity, 0)
			elif op == 'add' or op == 'additive':
				cv2.circle(dst, (x, y), _radius, (_r, _g, _b), -1)
			elif op == 'sub' or op == 'subtract':
				cv2.circle(dst, (x, y), _radius, (_r, _g, _b), -1)
			elif op == 'mul' or op == 'multiply':
				cv2.circle(dst, (x, y), _radius, (_r, _g, _b), -1)

	if op == 'add' or op == 'additive':
		copy = cv2.add(img, dst)
		img = cv2.addWeighted(img, 1 - _opacity, copy, _opacity, 0)
	elif op == 'sub' or op == 'subtract':
		copy = cv2.subtract(img, dst)
		img = cv2.addWeighted(img, 1 - _opacity, copy, _opacity, 0)
	elif op == 'mul' or op == 'multiply':
		copy = cv2.multiply(img, dst)
		img = cv2.addWeighted(img, 1 - _opacity, copy, _opacity, 0)
		pass

	# dw.ellipse((x - radius, y - radius, x + radius, y + radius),
	#            fill=(int(_r * 255), int(_g * 255), int(_b * 255), int(_opacity * 255)))
	v.session.img = img


@trace_decorator
def draw_sun_guide(v, x=None, y=None, scale=1, opacity=1, color=None):
	if x is None: x = v.w2
	if y is None: y = v.h2

	# + 0.55 * (abs(sin(v.t * 0.85 * tau)) ** 5)
	opacity = [0.5 * scale * opacity, 1 * v.scalar * opacity]
	size = [1 * scale, 15 * scale]
	area = 10 * scale
	count = 10 * scale
	falloff = 0.9

	draw_circles(v, x, y, opacity, size, area, count, falloff,
	             hsv=color or (0, 0, 1))


@trace_decorator
def draw_vignette(size=0.25, blur=350, opacity=0.1):
	"""
	Draw a vignette onto the image
	"""
	if rv.dry: return
	import cupy as cp

	vignette_cache.key(rv.w, rv.h, size, blur, opacity)
	if vignette_cache.new():
		vignette = np.zeros((rv.h, rv.w, 3), dtype=np.uint8)

		# Draw a white ellipse in the center of the image, matching the aspect ratio
		cv2.ellipse(vignette,
		            (int(rv.w2), int(rv.h2)),
		            (int(rv.w * (1 - size)), int(rv.h * (1 - size))),
		            0,
		            0,
		            360,
		            (255, 255, 255), -1)

		# Blur the ellipse
		blur = int(blur)
		if blur % 2 == 0:
			blur += 1
		cv2.GaussianBlur(vignette, (blur, blur), 0, dst=vignette)

		# Invert the vignette image
		# cv2.imwrite(rv.session.det_frame_path(rv.f, 'vignette').as_posix(), vignette)

		# Multiply the vignette image with the original image
		vignette = vignette / 255
		vignette = cp.asarray(vignette)
		vignette_cache.new(vignette)

	vignette = vignette_cache.get()

	# convert.save_img((vignette * 255).astype(np.uint8), rv.session.res('vignette'))

	img = rv.img
	img = cp.asarray(img)
	img = img.astype(np.float32)
	img = lerp(img, img * vignette, opacity)
	img = img.astype(np.uint8)
	img = cp.asnumpy(img)
	rv.img = img


@trace_decorator
def get_border_mask(feather_radius=50):
	border_cache.key(rv.w, rv.h, feather_radius)
	if border_cache.new():
		mask = np.ones((rv.h, rv.w), np.float32)
		# mask[feather_radius:-feather_radius, feather_radius:-feather_radius] = 1
		mask[0:feather_radius, :] *= np.linspace(0, 1, feather_radius)[:, None]
		mask[-feather_radius:, :] *= np.linspace(1, 0, feather_radius)[:, None]
		mask[:, 0:feather_radius] *= np.linspace(0, 1, feather_radius)[None, :]
		mask[:, -feather_radius:] *= np.linspace(1, 0, feather_radius)[None, :]

		mask = np.stack([mask, mask, mask], axis=2)
		mask = 1 - mask

		border_cache.new(mask)

	return border_cache.get()


@trace_decorator
def get_right_mask(pct=0.5, blur=30):
	right_mask_cache.key(rv.w, rv.h, pct, blur)
	if right_mask_cache.new():
		pct = 1 - pct
		right_mask = np.zeros((rv.h, rv.w, 3), np.uint8)
		right_mask[:, int(rv.w * pct):] = 1
		right_mask *= 255
		right_mask = cv2.blur(right_mask, (blur, blur))
		right_mask_cache.new(right_mask)

	return right_mask_cache.get()


@trace_decorator
def get_left_mask(pct=0.5, blur=30):
	left_mask_cache.key(rv.w, rv.h, pct, blur)
	if left_mask_cache.new():
		left_mask = np.zeros((rv.h, rv.w, 3), np.uint8)
		left_mask[:, :int(rv.w * pct)] = 1
		left_mask *= 255
		left_mask = cv2.blur(left_mask, (blur, blur))
		left_mask_cache.new(left_mask)

	return left_mask_cache.get()


@trace_decorator
def get_top_mask(pct=0.5, blur=30):
	top_mask_cache.key(rv.w, rv.h, pct, blur)
	if top_mask_cache.new():
		top_mask = np.zeros((rv.h, rv.w, 3), np.uint8)
		top_mask[:int(rv.h * pct), :] = 1
		top_mask *= 255
		top_mask = cv2.blur(top_mask, (blur, blur))
		top_mask_cache.new(top_mask)

	return top_mask_cache.get()


@trace_decorator
def get_down_mask(pct=0.5, blur=30):
	down_mask_cache.key(rv.w, rv.h, pct, blur)
	if down_mask_cache.new():
		pct = 1 - pct
		down_mask = np.zeros((rv.h, rv.w, 3), np.uint8)
		down_mask[int(rv.h * pct):, :] = 1
		down_mask *= 255
		down_mask = cv2.blur(down_mask, (blur, blur))
		down_mask_cache.new(down_mask)

	return down_mask_cache.get()


@trace_decorator
def get_yperp_mask(stretch=None):
	perp_mask_cache.key(rv.w, rv.h)
	if perp_mask_cache.new() or stretch is not None:
		stretch = stretch or 1.0
		mask = np.zeros((rv.h, rv.w, 3), np.uint8)
		w = rv.w
		h = rv.h

		# nobody can say I'm uncultured after this
		# https://youtu.be/GQO3SSlsgJM?t=4122

		for y in range(rv.h):
			yd = (y - h * 0.5) / (h * stretch)
			if yd < 0:
				yd = -yd
			mask[y, :] = yd * 255 * 2

		return perp_mask_cache.new(mask)

	return perp_mask_cache.get()


@trace_decorator
def get_left_axis_mask(p1, p2, blur=200):
	return get_axis_mask(p1, p2, side=-1, blur=blur)


@trace_decorator
def get_right_axis_mask(p1, p2, blur=200):
	return get_axis_mask(p1, p2, side=1, blur=blur)


@trace_decorator
def get_axis_mask(img, p1, p2, side=-1, blur=200):
	w2 = rv.w / 2
	h2 = rv.h / 2

	lspan = max(w2, h2) * 2
	ffspan = lspan * 1.25

	# P1 and P2 are tuples of (x, y)
	midpoint = (w2, h2)

	# Get vector orthogonal to line p1p2
	p1p2 = (p2[0] - p1[0], p2[1] - p1[1])
	p1p2 = (p1p2[0] / np.linalg.norm(p1p2), p1p2[1] / np.linalg.norm(p1p2))
	perp = (-p1p2[1], p1p2[0])
	ffpdir = (p1p2[0], p1p2[1])
	if side == 1:
		ffpdir = (-p1p2[0], -p1p2[1])

	l1 = (midpoint[0] + perp[0] * lspan, midpoint[1] + perp[1] * lspan)
	l2 = (midpoint[0] - perp[0] * lspan, midpoint[1] - perp[1] * lspan)
	ffp = (int(midpoint[0]) - ffpdir[0] * ffspan,
	       int(midpoint[1]) - ffpdir[1] * ffspan)

	# snap points to nearest pixel
	l1 = (clamp(l1[0], 0, rv.w - 1), clamp(l1[1], 0, rv.h - 1))
	l2 = (clamp(l2[0], 0, rv.w - 1), clamp(l2[1], 0, rv.h - 1))
	ffp = (clamp(ffp[0], 0, rv.w - 1), clamp(ffp[1], 0, rv.h - 1))

	l1 = (int(l1[0]), int(l1[1]))
	l2 = (int(l2[0]), int(l2[1]))
	ffp = (int(ffp[0]), int(ffp[1]))

	mask = np.ones((rv.h, rv.w, 3), np.uint8)
	mask = cv2.line(mask, l1, l2, (255, 255, 255), 8)
	# mask = cv2.circle(mask, ffp, 5, (255, 255, 255), -1)
	mask = cv2.floodFill(mask, None, ffp, (255, 255, 255))[1]
	mask = cv2.blur(mask, (blur, blur))

	return mask


@trace_decorator
def get_perlin_mask(img, size=5, exponent=1):
	import opensimplex
	global px, py
	img = load_cv2(img)
	px += img.x
	py += img.y

	# Create perlin for the image (v.w, v.h)
	# Move the perlin noise by (px, py)
	# Z axis is time (v.t)

	noise = opensimplex.noise3array(
		x=np.linspace(0, size, img.w) + px,
		y=np.linspace(0, size, img.h) + py,
		z=np.full(1, img.t),
	)[0]
	noise = (noise + 1) / 2
	noise = noise ** exponent
	return noise


def shade_perlin(img=None, strength=0.1, size=5, exponent=1):
	im = load_cv2(img)

	noise = 1 - clamp01(
		get_perlin_mask(rv, size=size, exponent=exponent) * (exponent / 2))
	# noise = noise ** 1.5
	# noise = clamp(noise, 0.8, 1)

	# noise *= 255
	# noise = noise.astype(np.uint8)
	# rv.img = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)

	im = rv.img.astype(np.float32) / 255
	im += (np.expand_dims(noise, axis=2) - 1) * 2 * strength
	im = clamp(im, 0, 1)
	im *= 255

	im = im.astype(np.uint8)
	return rv_set(img, im)


def shade_depth(img=None, strength=0.5, exponent=1, depth=None):
	if depth is None:
		depth = rv.depth
	if depth is None:
		return img


	img = rv.load_cv2(img)
	im = rv.img.astype(np.float32)
	depthnorm = clamp(depth, 0, 255) / 255
	depthnorm = depthnorm ** exponent
	depthnorm = clamp(depthnorm, 0, 1)
	im_prev = im
	im = clamp((im / 255) * depthnorm, 0, 1)
	im = (im*255).astype(np.uint8)

	im_blended = alpha_blend(im_prev, im, strength)
	return im_blended


@trace_decorator
def mat2d(img=None, x=0, y=0, z=0, r=0, mask=None, origin=(0.5, 0.5), **kwargs):
	"""
	2D transformations
	Args:
		img:
		h:
		w:
		x: X translation in pixels.
		y: Y translation in pixels.
		z: Zoom in percent. (1.0 = 100%, 0.5 = 50%, 2.0 = 200%, etc.)
		r: Rotation in degrees, origin is centered.
		origin:
		mask: Mask the distortion. This is not binary, the full spectrum from 0-255 acts as a weight.
		resolution: The work resolution. Lower resolution is less accurate, but much faster. Since this is a linear operation, the resolution can be set very low without much loss of quality, it will interpolate nearly perfectly.

	Returns:

	"""
	import cupy as cp

	if x == 0 and y == 0 and z == 1 and r == 0:
		return img

	im = load_cv2(img)
	if im is None: return

	w = im.shape[1]
	h = im.shape[0]

	# resolution = lerp(1.0, 0.25, ilerp(6668, 6668 + 25, v.f))
	hud.hud(x=x, y=y, z=z, r=r, origin=origin)

	fast_version = mask is None
	if not fast_version:
		# workaround, it's reversed somehow..
		y = -y
	if fast_version:
		# this is also reversed..........
		z = -z

	with trace("get_mat2d.get_matrix"):
		origin = (w * origin[1], h * origin[0])
		translate = cp.array(
			[[1, 0, -x],
			 [0, 1, -y]]
		)
		rotate = cv2.getRotationMatrix2D(origin, r, 1 - z)
		transform = cp.matmul(cp.vstack([rotate, [0, 0, 1]]),
		                      cp.vstack([translate, [0, 0, 1]]))

	if fast_version:
		# Fast native version. The other way uses remap, which is slow.
		im = cv2.warpPerspective(im, cp.asnumpy(transform), (w, h),
		                         borderMode=cv2.BORDER_REFLECT101)
		return im

	# Create coordinate arrays
	with trace("get_mat2d.create_buffers"):
		y, x = cp.indices((h, w)).astype(cp.float32)
		ones = cp.ones((h, w)).astype(cp.float32)
		coords = cp.stack([x.ravel(), y.ravel(), ones.ravel()], axis=1)

	with trace("get_mat2d.matmul"):
		# Multiply with the transformation matrix
		transformed_coords = cp.matmul(coords, transform.T)

	with trace("get_mat2d.reshape"):
		# Reshape transformed coordinates back into (h, w) arrays
		new_xs = transformed_coords[:, 0].reshape((h, w))
		new_ys = transformed_coords[:, 1].reshape((h, w))

	if mask is not None:
		with trace("get_mat2d.mask_convert"):
			# Convert mask to single-channel float array and normalize
			mask = mask[:, :, 0].astype(cp.float32)
			mask = cp.asarray(mask)

		with trace("get_mat2d.mask_apply"):
			# Scale transformed coordinates by mask
			# cv2.imwrite("new_xs_half.png", new_xs / w)
			# cv2.imwrite("new_ys_half.png", new_ys / h)
			new_xs = (new_xs - x) * mask + x
			new_ys = (new_ys - y) * mask + y

	# with trace("get_mat2d.resize"):
	#     new_xs = cv2.resize(new_xs, (w, h), interpolation=cv2.INTER_LINEAR)
	#     new_ys = cv2.resize(new_ys, (w, h), interpolation=cv2.INTER_LINEAR)

	# these are indices so we need to scale them as well
	# new_xs = new_xs / (new_xs.max()) * (w-1)
	# new_ys = new_ys / (new_ys.max()) * (h-1)
	# new_ys, new_xs = cp.indices((h, w)).astype(cp.float32)
	# cv2.imwrite(f"new_xs_{resolution}.png", new_xs / w * 255)
	# cv2.imwrite(f"new_ys_{resolution}.png", new_ys / h * 255)

	with trace("get_mat2d.remap"):
		# Fill color
		# TODO it does not work with any other color but pink........ wat
		fill_colors = [
			# (255, 0, 0),
			# (0, 255, 0),
			# (0, 0, 255),
			# (255, 255, 0),
			# (0, 255, 255),
			(255, 0, 255),
		]
		fill_color = fill_colors[random.randint(0, len(fill_colors) - 1)]

		new_xs = cp.asnumpy(new_xs)
		new_ys = cp.asnumpy(new_ys)
		new_xs = new_xs.astype(np.float32)
		new_ys = new_ys.astype(np.float32)

		# Convert new_xs and new_ys to UMat
		# new_xs = cv2.UMat(np.float32(new_xs))
		# new_ys = cv2.UMat(np.float32(new_ys))
		# im = cv2.UMat(im)
		warped = cv2.remap(im, new_xs, new_ys, cv2.INTER_LINEAR,
		                   borderMode=cv2.BORDER_REFLECT,
		                   borderValue=fill_color)
	# warped = cv2.UMat.get(warped)

	# with trace("get_mat2d.border_fill"):
	#     # cv2.imwrite(v.session.determine_frame_path(v.f, 'warped').as_posix(), warped)
	#     # if mask is not None:
	#     #     cv2.imwrite(v.session.determine_frame_path(v.f, 'mask').as_posix(), (mask*255).astype(np.uint8))
	#
	#     # Replace pure pink pixels with original im but randomly flipped & rotated (destroys border stretching AND increases entropy, simple but elegant)
	#     # Turn all the pink pixels into a mask
	#     reds = warped[:, :, 2]
	#     greens = warped[:, :, 1]
	#     blues = warped[:, :, 0]
	#     threshold = 5
	#     hi = 255 - threshold
	#     lo = 000 + threshold
	#     fr = fill_color[0]
	#     fg = fill_color[1]
	#     fb = fill_color[2]
	#     fill_indices = (reds > hi if fr > hi else reds < lo) & \
	#                    (greens > hi if fg > hi else greens < lo) & \
	#                    (blues > hi if fb > hi else blues < lo)
	#
	#     fill = im
	#     random_flip_dir = random.randint(-2, 1)
	#     random_rotate_direction = random.randint(-1, 1)
	#     rotation_codes = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
	#     if random_flip_dir > -2:
	#         fill = cv2.flip(im, random_flip_dir)
	#     if random_rotate_direction > -1:
	#         fill = cv2.rotate(fill, rotation_codes[random_rotate_direction])
	#         fill = cv2.resize(fill, (w, h))
	#
	#     warped[fill_indices] = fill[fill_indices]

	return rv_set(img, warped)


@trace_decorator
def do_grid_ripples(v, ripple_amplitude=None, ripple_period=None,
                    ripple_speed=None, mask=None):
	"""
	Distort the screen with sine waves.

	Args:
		v:
		ripple_amplitude: Movement in pixels.
		ripple_period: How many waves.
		ripple_speed:
		mask:

	Returns:
	"""
	import cupy as cp

	if v == rv and ripple_amplitude is None: ripple_amplitude = (
		rv.ripple_amplitude, rv.ripple_amplitude)
	if v == rv and ripple_speed is None: ripple_speed = (
		rv.ripple_speed, rv.ripple_speed)
	if v == rv and ripple_period is None: ripple_period = (
		rv.ripple_period * tau, rv.ripple_period * tau)

	hud.hud(ripple_speed=ripple_speed, ripple_period=ripple_period,
	        ripple_amplitude=ripple_amplitude)
	w = v.w
	h = v.h
	flex_x = cp.zeros((h, w))
	flex_y = cp.zeros((h, w))

	# x = cp.arange(w)
	# y = cp.arange(h)
	# x, y = cp.meshgrid(x, y)
	y, x = cp.indices((h, w)).astype(cp.float32)

	sx = cp.cos((x + v.t * ripple_speed[0]) / ripple_period[0]) * \
	     ripple_amplitude[0]
	sy = cp.sin((y + v.t * ripple_speed[1]) / ripple_period[1]) * \
	     ripple_amplitude[1]
	cx = sy
	cy = sx
	# cy = cp.sin((y + v.t * ripple_speed[1]) / ripple_period[1]) * ripple_amplitude[1]

	if mask is not None:
		mask = cv2.resize(mask, (w, h))
		mask = mask[:, :, 0].astype(cp.float32)
		mask = cp.asarray(mask)
		cx *= mask
		cy *= mask

	cx = cx.astype(np.float32)
	cy = cy.astype(np.float32)

	new_xs = x + cx
	new_ys = y + cy
	new_xs = cp.asnumpy(new_xs)
	new_ys = cp.asnumpy(new_ys)
	new_xs = new_xs.astype(np.float32)
	new_ys = new_ys.astype(np.float32)

	with trace("do_grid_ripples.remap:"):
		v.img = cv2.remap(v.img, new_xs, new_ys, cv2.INTER_LINEAR,
		                  borderMode=cv2.BORDER_REFLECT, borderValue=(0, 0, 0))

	return v.img


def sp_noise(img, coverage):
	im = load_cv2(img)

	if len(im.shape) == 2:
		black = 0
		white = 255
	else:
		colorspace = im.shape[2]
		if colorspace == 3:  # RGB
			white = rgb_white
			black = rgb_black
		else:  # RGBA
			white = rgba_white
			black = rgba_black

	sp = get_spnoise(im, coverage)
	# im = cp.asarray(im)
	im[sp < (coverage / 2)] = black
	im[sp > 1 - (coverage / 2)] = white
	# im = cp.asnumpy(im)

	return rv_set(img, im)


@trace_decorator
def saltpepper(img, coverage=0.02):
	# noise = random_noise(img, mode='s&p', amount=j.coverage)
	# img = np.array(255 * noise, dtype='uint8')

	# if mask is not None:
	#     mask = np.array(mask.convert("RGB"))
	return sp_noise(img, coverage)

	im = load_cv2(img)

	original_dtype = im.dtype

	# Derive the number of intensity levels from the array datatype.
	intensity_levels = 2 ** (im[0, 0].nbytes * 8)
	min_intensity = 0
	max_intensity = intensity_levels - 1

	random_image_arr = np.random.choice(
		[min_intensity, 1, np.nan],
		p=[coverage / 2, 1 - coverage, coverage / 2],
		size=im.shape
	)

	# This results in an image array with the following properties:
	# - With probability 1 - prob: the pixel KEEPS ITS VALUE (it was multiplied by 1)
	# - With probability prob/2: the pixel has value zero (it was multiplied by 0)
	# - With probability prob/2: the pixel has value np.nan (it was multiplied by np.nan)
	# `arr.astype(np.float)` to make sure np.nan is a valid value.
	salt_and_peppered_arr = im.astype(float) * random_image_arr

	# Since we want SALT instead of NaN, we replace it.
	# We cast the array back to its original dtype so we can pass it to PIL.
	salt_and_peppered_arr = np.nan_to_num(
		salt_and_peppered_arr,
		nan=max_intensity
	).astype(original_dtype)

	ret = salt_and_peppered_arr

	# if mask is not None:
	#     zeros = np.zeros_like(im)
	# ret = ret * mask + zeros * (1-mask)

	return rv_set(img, ret)


@trace_decorator
def grain(img=None, strength=0.1, mask=None):
	"""
	High frequency grain at low opacity
	"""

	hud.hud(grain=strength)

	im = load_cv2(img)
	if im is None:
		print("WARNING: grain() called with no image and none in rv.img")
		return None

	noise = get_grain(im)
	if mask:
		noise = noise * mask

	import cupy as cp
	im = cp.asarray(im)
	noise = cp.asarray(noise)

	im = im.astype(np.float32)
	im += noise * strength
	im = np.clip(im, 0, 255)
	im = cp.asnumpy(im)
	im = im.astype(np.uint8)

	# im = cv2.convertScaleAbs(im)

	if isinstance(img, RenderVars):
		rv.img = im

	return rv_set(img, im)


@trace_decorator
def grain_mul(img=None, strength=0.1, mask=None):
	"""
	High frequency grain at low opacity
	"""

	im = load_cv2(img)
	if im is None:
		print("WARNING: grain() called with no image and none in rv.img")
		return None

	noise = get_grain(im)
	noise = noise.astype(np.float32) / 255.0
	if mask:
		noise = noise * mask

	im = im.astype(np.float32)
	im += noise * strength
	return rv_set(img, im)


@trace_decorator
def get_grain(im):
	global grain_cache
	grain_cache_size = 24
	grain_cache.key(*im.shape)
	if grain_cache.new():
		cache = []
		while True:
			g = np.random.randint(-128, 128, im.shape, dtype=np.int8)
			cache.append(g)
			if len(cache) == grain_cache_size:
				break
		grain_cache.new(cache)
	cache = grain_cache.get()
	noise = cache[rv.f % grain_cache_size]
	return noise


def get_spnoise(im, coverage):
	global grain_cache
	sp_cache_size = 8
	sp_cache_increments = 0.001

	coverage = int(coverage / sp_cache_increments) * sp_cache_increments

	sp_cache.key(*im.shape, coverage)
	if sp_cache.new():
		cache = []
		for i in range(sp_cache_size):
			probs = np.random.random(im.shape[:2])

			cache.append(probs)
		sp_cache.new(cache)

	cache = sp_cache.get()
	noise = cache[rv.f % sp_cache_size]
	return noise


def edgecomp(img):
	img, comp = edgedet(img)
	return comp


def edgedet(pil):
	from PIL import Image
	import numpy as np
	import scipy.signal as sg

	if pil is None:
		return [], 0

	img = np.asarray(pil)
	img = img[:, :, 0]
	img = img.astype(np.int16)
	edge = np.array([[-1, -1, -1],
	                 [-1, 8, -1],
	                 [-1, -1, -1]])
	results = sg.convolve(img, edge, mode='same')
	results[results > 127] = 255
	results[results < 0] = 0
	results = results.astype(np.uint8)
	comp = np.mean(results)

	return PIL.Image.fromarray(results), comp


@trace_decorator
def do_canny_edge_reinforce(im, white, black, edge_img=None):
	if edge_img is None:
		edge_img = im
	if isinstance(edge_img, (str, Path)):
		edge_img = cv2.imread(str(edge_img))
		# Resize to fit
		edge_img = cv2.resize(edge_img, (rv.w, rv.h))

	edges = cv2.Canny(cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY), 20, 80)
	edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

	edges_blurred1 = cv2.GaussianBlur(edges, (0, 0), 1)
	edges_blurred2 = cv2.GaussianBlur(edges, (0, 0), 2)

	# Blur main image
	# v.img = cv2.GaussianBlur(v.image_cv2, (0,0), 2)
	xblur = 2
	yblur = 2

	im = cv2.blur(im, (xblur, yblur))  # Alternative with cheap box blur
	im = contrast(im, 1.25)

	im = im
	im = im.astype(np.uint16)
	im = im * (1 + edges * white)
	im = im * (1 - edges_blurred2 * black)

	# im = im * (1-edges_blurred2*v.edges)
	# im = im * (1-edges_blurred2*v.edges)

	# rgb edge fuzz: replace the edges with random colors (random hue, fixed saturation and value)
	# edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
	# edge_hsv = np.zeros((v.h, v.w, 3), dtype=np.uint8)
	# edge_hsv[...,0] = np.random.randint(0, 180, (v.h, v.w))
	# edge_hsv[...,1] = 255
	# edge_hsv[...,2] = 255
	# edge_rgb = cv2.cvtColor(edge_hsv, cv2.COLOR_HSV2BGR)
	# im = np.where(edges[...,None], edge_rgb, im)

	im = np.clip(im, 0, 255)
	im = im.astype(np.uint8)
	# cv2.imwrite('edges.jpg', cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
	# cv2.imwrite('edges.jpg', edges)
	return im


# @plugfun(plugfun_img)
@trace_decorator
def img_to_hed(img):
	return plugins.get("heddet").img_to_hed(img)


def img_to_canny(img, lo=100, hi=200):
	img = load_cv2(img)
	canny = cv2.Canny(img, lo, hi)
	canny = canny[:, :, None]
	canny = np.concatenate([canny, canny, canny], axis=2)

	return canny


def img_to_canny_blend(frame, guide, t, lo, hi):
	frame = load_cv2(frame)
	guide = load_cv2(guide)
	# print(frame, guide)
	if guide is None:
		return frame

	composite = frame
	if t > 0:
		composite = alpha_blend(frame, guide, t)

	canny = img_to_canny(composite, lo, hi)
	hud.snap('canny', canny)
	return pil2cv(canny)


@plugfun()
def img_to_hed_blend(frame, guide, t):
	frame = load_cv2(frame)
	guide = load_cv2(guide)
	if guide is None:
		return frame

	composite = frame
	if t > 0:
		composite = alpha_blend(frame, guide, t)
	composite = load_pil(composite)

	hed = img_to_hed(composite)
	hud.snap('hed', hed)
	return pil2cv(hed)


@trace_decorator
def do_perlin_deform(v):
	# Perlin deform
	# perlin_speed = 0.1
	# perlin_scale = 0.1
	# perlin = np.zeros((h,w),np.float32)
	# for y,x in np.ndindex(perlin.shape):
	#     perlin[y,x] = pnoise2(x*perlin_scale, y*perlin_scale, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=w, repeaty=h, base=0)
	#     flex
	#     perlin = perlin * 10
	#     flex_x += perlin
	#     flex_y += perlin
	pass


@trace_decorator
def do_histogram_stretch(im):
	"""
		Histogram stretching to improve contrast
	Args:
		im:

	Returns:
	"""
	im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	im[:, :, 0] = clahe.apply(im[:, :, 0])
	im = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)
	return im


@trace_decorator
def do_histogram_norm(im):
	"""
		Histogram normalization to improve contrast ensure we have pure white and black
	"""
	im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
	im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
	im = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)
	return im


@trace_decorator
def do_min_brightness(hsv, v):
	brightness = hsv[..., 2].mean()
	min_brightness = 50
	ratio = brightness / min_brightness
	if ratio < 1:
		print('brightness ratio', ratio)
		return cv2.convertScaleAbs(v.img, alpha=1 / ratio, beta=0)
	return v.img


@trace_decorator
def hsv_lerp(hsv, hsv_speed, hsv_target):
	if hsv_target[0] is not None: hsv[:, :, 0] = lerp(hsv[:, :, 0],
	                                                  hsv_target[0], hsv_speed)
	if hsv_target[1] is not None: hsv[:, :, 1] = lerp(hsv[:, :, 1],
	                                                  hsv_target[1], hsv_speed)
	if hsv_target[2] is not None: hsv[:, :, 2] = lerp(hsv[:, :, 2],
	                                                  hsv_target[2], hsv_speed)
	hsv = hsv.astype(np.uint8)
	return hsv


@trace_decorator
def brightness(hsv, brightness, mask=None):
	hsv = hsv.astype(np.float32)
	initial = None
	if mask:
		initial = hsv.copy()

	hsv[..., 2] = hsv[..., 2] * (1 + brightness)
	hsv = np.clip(hsv, 0, 255)

	if mask:
		hsv = lerp(initial, hsv, mask)

	hsv = hsv.astype(np.uint8)
	return hsv


@trace_decorator
def saturation(hsv, saturation, mask=None):
	if saturation is None: return hsv
	hsv = hsv.astype(np.float32)
	initial = None
	if mask:
		initial = hsv.copy()

	hsv[..., 1] = hsv[..., 1] * (1 + saturation)
	hsv = np.clip(hsv, 0, 255)

	if mask:
		hsv = lerp(initial, hsv, mask)
	hsv = hsv.astype(np.uint8)
	return hsv


@trace_decorator
def hue(hsv, hue, mask=None):
	"""
	Rotate the hues by the specified amount in degrees (0 = no change)
	"""
	initial = None
	if mask:
		initial = hsv.copy()

	hsv[..., 0] = (hsv[..., 0] + hue) % 180

	if mask:
		hsv = lerp(initial, hsv, mask)

	return hsv


@trace_decorator
def stretch_colors(image):
	# Convert image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Apply histogram equalization to stretch the color range
	stretched = exposure.equalize_hist(gray)

	# Convert the stretched grayscale image back to BGR
	stretched_bgr = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)

	return stretched_bgr


@trace_decorator
def contrast(img, strength):
	# Contrast adjustment
	im = img
	if im is None:
		im = rv.img

	lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
	l_channel, a, b = cv2.split(lab)
	# Applying CLAHE to L-channel, feel free to try different values for the limit and grid size:
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	cl = clahe.apply(l_channel)
	limg = cv2.merge((cl, a,
	                  b))  # merge the CLAHE enhanced L-channel with the a and b channel
	enhanced_img = cv2.cvtColor(limg,
	                            cv2.COLOR_LAB2BGR)  # Converting image from LAB Color model to BGR color spcae
	# lerp between original and enhanced
	weight = strength - 1
	im = cv2.addWeighted(im, 1 - weight, enhanced_img, weight, 0)
	hud.hud(contrast=strength)
	return rv_set(img, im)


def selective_contrast_enhancement(image, low_threshold, high_threshold):
	# Convert the image to LAB color space
	lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

	# Split the LAB channels
	l_channel, a_channel, b_channel = cv2.split(lab)

	# Apply histogram stretching to the L channel
	l_channel_stretched = apply_histogram_stretching(l_channel, low_threshold,
	                                                 high_threshold)

	# Merge the adjusted L channel with the original A and B channels
	enhanced_lab = cv2.merge([l_channel_stretched, a_channel, b_channel])

	# Convert the enhanced LAB image back to BGR color space
	enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

	return enhanced_image


def apply_histogram_stretching(channel, low_threshold, high_threshold):
	# Normalize the channel values to the range of 0-1
	normalized_channel = channel.astype(np.float32) / 255.0

	# Stretch the histogram by applying the thresholds
	stretched_channel = np.clip(
		(normalized_channel - low_threshold) / (high_threshold - low_threshold),
		0, 1)

	# Scale the stretched channel values back to the range of 0-255
	scaled_channel = (stretched_channel * 255).astype(np.uint8)

	return scaled_channel


@trace_decorator
def alpha_blend_masked(a, b, mask):
	# Convert uint8 to float
	a = a.astype(float)
	b = b.astype(float)

	mask = mask.astype(float) / 255
	a = cv2.multiply(1 - mask, a)
	b = cv2.multiply(mask, b)

	cv2.imwrite('alpha_mask.png', mask * 255)
	cv2.imwrite('alpha_mask_a.png', a)
	cv2.imwrite('alpha_mask_b.png', b)
	out = cv2.add(a, b)

	return out


@trace_decorator
def alpha_blend(a, b, t):
	if a is None and b is None:
		raise Exception("alpha_blend: both a and b are None")
	if a is None:
		print("WARNING: alpha_blend: a is None")
		return b
	if b is None:
		print("WARNING: alpha_blend: b is None")
		return a

	# a/b are cv2 images
	a = load_cv2(a)
	b = load_cv2(b)
	b = convert.crop_or_pad(b, a.shape[1], a.shape[0])

	if t < 0:
		print(f"WARNING: alpha_blend t<0  (t={str(t)})")
		return a
	if t > 1:
		print(f"WARNING: alpha_blend t>1  (t={str(t)})")
		return b

	ret = cv2.addWeighted(a, 1 - t, b, t, 0, dtype=cv2.CV_32F)
	ret = ret.astype(np.uint8)
	return ret


@trace_decorator
def rife_blend(a, b, t):
	apath = paths.tmp / 'a.png'
	bpath = paths.tmp / 'b.png'
	outdir = paths.tmp / 'out_rife'

	paths.mktree(paths.tmp)
	paths.mktree(outdir)

	cv2.imwrite(str(apath), a)
	cv2.imwrite(str(bpath), b)

	# Rife
	cmd = f'python {paths.rife} --model_path {paths.rife_model} --image_path {apath} {bpath} --output_path {outdir} --alpha {t}'


def get_matrix_for_hybrid_motion(frame_idx, dimensions, inputfiles,
                                 hybrid_video_motion):
	matrix = get_translation_matrix_from_images(str(inputfiles[frame_idx]),
	                                            str(inputfiles[frame_idx + 1]),
	                                            dimensions, hybrid_video_motion)
	print(
		f"Calculating {hybrid_video_motion} RANSAC matrix for frames {frame_idx} to {frame_idx + 1}")
	return matrix


def get_flow_for_hybrid_motion(frame_idx, dimensions, inputfiles,
                               hybrid_frame_path, method,
                               save_flow_visualization=False):
	print(f"Calculating optical flow for frames {frame_idx} to {frame_idx + 1}")
	flow = get_flow(str(inputfiles[frame_idx]), str(inputfiles[frame_idx + 1]),
	                dimensions, method)
	if save_flow_visualization:
		flow_img_file = os.path.join(hybrid_frame_path,
		                             f"flow{frame_idx:05}.jpg")
		flow_cv2 = cv2.imread(str(inputfiles[frame_idx]))
		flow_cv2 = cv2.resize(flow_cv2, (dimensions[0], dimensions[1]),
		                      cv2.INTER_AREA)
		flow_cv2 = cv2.cvtColor(flow_cv2, cv2.COLOR_BGR2RGB)
		flow_cv2 = draw_flow_lines_in_grid_in_color(flow_cv2, flow)
		flow_PIL = Image.fromarray(np.uint8(flow_cv2))
		flow_PIL.save(flow_img_file)
		print(f"Saved optical flow visualization: {flow_img_file}")
	return flow


def draw_flow_lines_in_grid_in_color(img, flow, step=8, magnitude_multiplier=1,
                                     min_magnitude=1, max_magnitude=100):
	flow = flow * magnitude_multiplier
	h, w = img.shape[:2]
	y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
	fx, fy = flow[y, x].T
	lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
	lines = np.int32(lines + 0.5)
	vis = img.copy()  # Create a copy of the input image
	mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
	hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
	hsv[..., 0] = ang * 180 / np.pi / 2
	hsv[..., 1] = 255
	hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	# Iterate through the lines
	for (x1, y1), (x2, y2) in lines:
		# Calculate the magnitude of the line
		magnitude = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

		# Only draw the line if it falls within the magnitude range
		if min_magnitude <= magnitude <= max_magnitude:
			b = int(bgr[y1, x1, 0])
			g = int(bgr[y1, x1, 1])
			r = int(bgr[y1, x1, 2])
			color = (b, g, r)
			cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, thickness=1,
			                tipLength=0.2)

	return vis


def flow_to_bgr(flow_map, max_flow=20):
	"""
	Convert a flow map to a BGR image for visualization.

	Args:
		flow_map (ndarray): Flow map of shape (H, W, 2) representing optical flow vectors.
		max_flow (float): Maximum flow value for normalization (default: 20).

	Returns:
		ndarray: BGR image of the flow map.
	"""
	hsv = np.zeros((flow_map.shape[0], flow_map.shape[1], 3), dtype=np.uint8)
	hsv[..., 1] = 255

	magnitude, angle = cv2.cartToPolar(flow_map[..., 0], flow_map[..., 1])

	# Normalize the magnitude for visualization
	magnitude = np.clip(magnitude / max_flow, 0, 1)

	# Set hue according to the angle
	hsv[..., 0] = angle * 180 / np.pi / 2

	# Set value according to the magnitude
	hsv[..., 2] = magnitude * 255

	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	return bgr


def bgr_to_flow(bgr_image, max_flow=20):
	"""
	Convert a BGR image back to a flow map.

	Args:
		bgr_image (ndarray): BGR image representing the flow map.
		max_flow (float): Maximum flow value for denormalization (default: 20).

	Returns:
		ndarray: Flow map of shape (H, W, 2) representing optical flow vectors.
	"""
	hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

	# Extract hue (angle) and value (magnitude)
	angle = hsv[..., 0] * np.pi / 180 * 2
	magnitude = hsv[..., 2] / 255.0

	# Compute x and y components of flow vectors
	flow_map = np.zeros((hsv.shape[0], hsv.shape[1], 2), dtype=np.float32)
	flow_map[..., 0], flow_map[..., 1] = cv2.polarToCart(magnitude, angle)

	# Denormalize the flow values
	flow_map *= max_flow

	return flow_map


@trace_decorator
def get_flow(img1, img2, method=DEFAULT_FLOW_METHOD):
	method = method or DEFAULT_FLOW_METHOD

	dimensions = (rv.w, rv.h)

	i1 = load_cv2(img1)
	i2 = load_cv2(img2)
	i1 = cv2.resize(i1, (dimensions[0], dimensions[1]), cv2.INTER_AREA)
	i2 = cv2.resize(i2, (dimensions[0], dimensions[1]), cv2.INTER_AREA)
	if method == "DenseRLOF":
		r = get_flow_dense_rlof(i1, i2)
	elif method == "SF":
		r = get_flow_SF(i1, i2)
	elif method == "Farneback":
		r = get_flow_farneback(i1, i2)
	elif method == "rife":
		# r = get_flow_rife(i1, i2)
		raise NotImplementedError("RIFE flow estimation is not implemented yet")
	elif method == 'unimatch':
		r = get_flow_unimatch(i1, i2)

	return r


@trace_decorator
def get_flow_farneback(img1, img2, last_flow=None, pyr_scale=0.5, levels=3,
                       winsize=15, iterations=3, poly_n=5, poly_sigma=1.2):
	i1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	i2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	flags = 0  # flags = cv2.OPTFLOW_USE_INITIAL_FLOW
	flow = cv2.calcOpticalFlowFarneback(i1, i2, last_flow, pyr_scale, levels,
	                                    winsize, iterations, poly_n, poly_sigma,
	                                    flags)
	return flow


@trace_decorator
def get_flow_dense_rlof(i1, i2, last_flow=None):
	return cv2.optflow.calcOpticalFlowDenseRLOF(i1, i2, flow=last_flow)


@trace_decorator
def get_flow_SF(i1, i2, last_flow=None):
	layers = 3
	averaging_block_size = 2
	max_flow = 4
	return cv2.optflow.calcOpticalFlowSF(i1, i2, layers, averaging_block_size,
	                                     max_flow)


@trace_decorator
def get_flow_unimatch(i1, i2, last_flow=None):
	unimatch = plugins.get('unimatch', loading=True)
	flow = unimatch.calc_flow(i1, i2)
	return flow


def get_flow_compute_dirpath(resname):
	resdir = session.res_frame_dirpath(resname)
	flowdir = paths.unhide(resdir.parent / f"{resdir.name}_flow/")
	flowdir = paths.hide(flowdir)

	return flowdir


def precompute_flows(resname, method=None):
	if not ENABLE_FLOW_PRECOMPUTE: return

	print("Precomputing flows...")

	framedir = session.extract_frames(resname)
	flowdir = get_flow_compute_dirpath(resname)

	inputfiles = sorted(framedir.glob("*.jpg"))

	if flowdir.exists():
		# listing out the files to count them is too slow
		# this is assuming the user isnt a fuckig idiot & stopped midway LOL
		return flowdir

	# Save zero flow for first frame
	flow_file = flowdir / f"{inputfiles[0].stem}.jpg"
	flow_zero = np.zeros((rv.h, rv.w, 2))
	flow_bgr = flow_to_bgr(flow_zero)
	convert.save_jpg(load_pil(flow_bgr), flow_file, quality=100)

	for i in tqdm(range(1, len(inputfiles) - 1),
	              desc=f"Precomputing {method} for {resname}..."):
		img1 = inputfiles[i]
		img2 = inputfiles[i + 1]

		flow_file = flowdir / f"{img1.stem}.npy"
		if flow_file.exists():
			pass

		flow = get_flow(img1, img2, method)
		paths.mktree(flowdir)

		# print(flow_file, get_flow_max_magnitude(flow))
		# print(flow)
		# print("computed flow ^")

		flow_bgr = flow_to_bgr(flow)
		flow_pil = load_pil(flow_bgr)
		convert.save_jpg(flow_pil, flow_file, quality=100)

	# np.save(flow_file, flow)

	# Copy info.json from src dir to flow dir
	# TODO This is leaky abstraction
	paths.cp(framedir / 'info.json', flowdir / 'info.json')

	return flowdir


def load_precomputed_flow(path):
	bgr = load_cv2(path)
	flow = bgr_to_flow(bgr)
	return flow


@trace_decorator
def flow_to_mask(flow):
	"""
	Converts a flow image to a black and white mask image.
	Still return with 3 channels so we can use it with the color images.
	"""
	# Calculate the magnitude of the flow
	magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

	# Normalize the magnitude
	magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

	# Convert to uint8
	magnitude = magnitude.astype(np.uint8)

	# Convert to 3 channels
	magnitude = np.repeat(magnitude[:, :, np.newaxis], 3, axis=2)

	return magnitude


@trace_decorator
def get_flow_avg_direction(flow):
	"""
	Calculates the average direction of the flow.
	"""
	# Calculate the angle of the flow
	_, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

	# Calculate the average angle
	avg_angle = np.average(angle)

	# Convert to vector
	avg_angle = np.array([np.cos(avg_angle), np.sin(avg_angle)])

	return avg_angle


@trace_decorator
def get_flow_avg_magnitude(flow):
	"""
	Calculates the average magnitude of the flow.
	"""
	# Calculate the magnitude of the flow
	magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

	# Calculate the average magnitude
	avg_magnitude = np.average(magnitude)

	return avg_magnitude


@trace_decorator
def get_flow_max_magnitude(flow):
	"""
	Calculates the maximum magnitude of the flow.
	"""
	# Calculate the magnitude of the flow
	magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

	# Calculate the maximum magnitude
	max_magnitude = np.max(magnitude)

	return max_magnitude


@trace_decorator
def get_flow_strongest_direction(flow, pull=1):
	"""
	Calculates the strongest direction of the flow.

	args:
		pull: The power to raise the magnitude to. Higher values will pull the average more towards the strongest flow.
	"""
	# Calculate the magnitude of the flow
	magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

	# Calculate the strongest direction
	strongest_direction = np.average(magnitude ** pull)

	return strongest_direction


def get_init_motion_stats(resname='init'):
	framedir = session.extract_frames(resname)

	fpath_magnitude = (framedir / 'magnitude.npy').as_posix()
	fpath_avg_x = (framedir / 'avg_x.npy').as_posix()
	fpath_avg_y = (framedir / 'avg_y.npy').as_posix()

	if (framedir / 'magnitude.npy').exists():
		magnitude = np.load(fpath_magnitude)
		avg_x = np.load(fpath_avg_x)
		avg_y = np.load(fpath_avg_y)
	else:
		magnitude = np.zeros(rv.n)
		avg_x = np.zeros(rv.n)
		avg_y = np.zeros(rv.n)
		for i in trange(2, rv.n - 1, desc="Calculating motion stats"):
			im1 = session.res_frame_cv2(resname, i - 1)
			im2 = session.res_frame_cv2(resname, i)
			flow = get_flow(im1, im2)

			magnitude[i] = get_flow_avg_magnitude(flow)
			direction = get_flow_avg_direction(flow)
			avg_x = direction[0]
			avg_y = direction[1]

		magnitude = norm(magnitude)

		# Cache into files
		np.save(fpath_magnitude, magnitude)
		np.save(fpath_avg_x, avg_x)
		np.save(fpath_avg_y, avg_y)

	return magnitude, avg_x, avg_y


def get_blurred(img, blur=0):
	if blur > 0:
		# Define the size of the blur kernel
		blur = ceil(blur)
		kernel_size = (blur, blur)

		# Apply blur to the image
		img = cv2.blur(img, kernel_size)

	return img


@trace_decorator
def flow_warp(img, flow=None, init=None, blur=0, strength=1, exponent=1):
	if init is not None or flow is None:
		if rv.f < 2: return img
		flow = get_flow(
			rv.session.res_frame_cv2(init, rv.f - 1),
			rv.session.res_frame_cv2(init, rv.f)
		)
		if blur > 0:
			# # Extract the flow vectors in the x and y directions
			# flow_x = flow[..., 0]
			# flow_y = flow[..., 1]
			#
			# # Define the size of the blur kernel
			# blur = ceil(blur)
			# kernel_size = (blur, blur)
			#
			# # Apply blur to the flow vectors separately
			# blurred_flow_x = cv2.blur(flow_x, kernel_size)
			# blurred_flow_y = cv2.blur(flow_y, kernel_size)
			#
			# # Combine the blurred flow vectors into a single flow map
			# flow = np.stack((blurred_flow_x, blurred_flow_y), axis=-1)

			# Split the flow map into magnitude and direction channels
			magnitude, direction = cv2.split(flow)

			# Apply Gaussian blur to the magnitude channel
			magnitude_blurred = cv2.GaussianBlur(magnitude, (5, 5), 0)

			# Apply Gaussian blur to the direction channel
			direction_blurred = cv2.GaussianBlur(direction, (5, 5), 0)

			# Merge the blurred magnitude and direction channels back into a flow map
			flow = cv2.merge([magnitude_blurred, direction_blurred])

	img = load_cv2(img)

	# Resize the flow to the image size
	flow = cv2.resize(flow, (img.shape[1], img.shape[0]), cv2.INTER_AREA)

	# Warp the image
	h, w = flow.shape[:2]
	flow *= strength
	flow **= exponent
	flow = -flow
	# print(flow)
	flow[:, :, 0] += np.arange(w)
	flow[:, :, 1] += np.arange(h)[:, np.newaxis]

	res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)

	return res


def create_trippy_effect(base_flow_path, blurred_flow_path, image_path,
                         weight_original=0.7, weight_warped=0.3):
	base_flow = cv2.imread(base_flow_path, 0)
	blurred_flow = cv2.imread(blurred_flow_path, 0)

	diff_flow = cv2.absdiff(blurred_flow, base_flow)
	equalized_diff = cv2.equalizeHist(diff_flow)
	normalized_diff = cv2.normalize(equalized_diff, None, 0, 255,
	                                cv2.NORM_MINMAX)

	original_image = cv2.imread(image_path)
	warped_image = cv2.remap(original_image, base_flow, None, cv2.INTER_LINEAR)

	final_image = cv2.addWeighted(original_image, weight_original, warped_image,
	                              weight_warped, 0)

	return final_image


def get_translation_matrix_from_images(i1, i2, dimensions, hybrid_video_motion,
                                       max_corners=200, quality_level=0.01,
                                       min_distance=30, block_size=3):
	img1 = cv2.imread(i1, 0)
	img2 = cv2.imread(i2, 0)
	img1 = cv2.resize(img1, (dimensions[0], dimensions[1]), cv2.INTER_AREA)
	img2 = cv2.resize(img2, (dimensions[0], dimensions[1]), cv2.INTER_AREA)

	# Detect feature points in previous frame
	prev_pts = cv2.goodFeaturesToTrack(img1,
	                                   maxCorners=max_corners,
	                                   qualityLevel=quality_level,
	                                   minDistance=min_distance,
	                                   blockSize=block_size)

	if prev_pts is None or len(prev_pts) < 8:
		return get_hybrid_video_motion_default_matrix(hybrid_video_motion)

	# Get optical flow
	curr_pts, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, prev_pts, None)

	# Filter only valid points
	idx = np.where(status == 1)[0]
	prev_pts = prev_pts[idx]
	curr_pts = curr_pts[idx]

	if len(prev_pts) < 8 or len(curr_pts) < 8:
		return get_hybrid_video_motion_default_matrix(hybrid_video_motion)

	if hybrid_video_motion == "Perspective":  # Perspective - Find the transformation between points
		transformation_matrix, mask = cv2.findHomography(prev_pts, curr_pts,
		                                                 cv2.RANSAC, 5.0)
		return transformation_matrix
	else:  # Affine - Compute a rigid transformation (without depth, only scale + rotation + translation)
		transformation_rigid_matrix, rigid_mask = cv2.estimateAffinePartial2D(
			prev_pts, curr_pts)
		return transformation_rigid_matrix


def image_transform_ransac(image_cv2, xform, hybrid_video_motion,
                           border_mode=cv2.BORDER_REPLICATE):
	if hybrid_video_motion == "Perspective":
		return image_transform_perspective(image_cv2, xform,
		                                   border_mode=border_mode)
	else:  # Affine
		return image_transform_affine(image_cv2, xform, border_mode=border_mode)


def image_transform_affine(image_cv2, xform, border_mode=cv2.BORDER_REPLICATE):
	return cv2.warpAffine(
		image_cv2,
		xform,
		(image_cv2.shape[1], image_cv2.shape[0]),
		borderMode=border_mode
	)


def image_transform_perspective(image_cv2, xform,
                                border_mode=cv2.BORDER_REPLICATE):
	return cv2.warpPerspective(
		image_cv2,
		xform,
		(image_cv2.shape[1], image_cv2.shape[0]),
		borderMode=border_mode
	)


def autocontrast_grayscale(image, low_cutoff=0, high_cutoff=100):
	# Perform autocontrast on a grayscale np array image.
	# Find the minimum and maximum values in the image
	min_val = np.percentile(image, low_cutoff)
	max_val = np.percentile(image, high_cutoff)

	# Scale the image so that the minimum value is 0 and the maximum value is 255
	image = 255 * (image - min_val) / (max_val - min_val)

	# Clip values that fall outside the range [0, 255]
	image = np.clip(image, 0, 255)

	return image


def get_hybrid_video_motion_default_matrix(hybrid_video_motion):
	if hybrid_video_motion == "Perspective":
		arr = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
	else:
		arr = np.array([[1., 0., 0.], [0., 1., 0.]])
	return arr


@plugfun(default_return=plugfun_img)
def get_depth(img, model_type="DPT_Hybrid"):
	depth = plugins.get('hfmidas3d').get_depth(img, model_type)
	rv.depth = depth
	hud.snap(model_type, depth)
	return depth


def maintain_colors(prev_img, color_match_sample, mode):
	from skimage.exposure import match_histograms
	if mode == 'RGB':
		return match_histograms(prev_img, color_match_sample, multichannel=True)
	elif mode == 'HSV':
		prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
		color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
		matched_hsv = match_histograms(prev_img_hsv, color_match_hsv,
		                               multichannel=True)
		return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
	elif mode == 'LAB':  # Match Frame 0 LAB
		prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
		color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
		matched_lab = match_histograms(prev_img_lab, color_match_lab,
		                               multichannel=True)
		return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)


def palette(v, img, palette_img, mode='LAB'):
	from src.classes.convert import pil2cv
	from src.classes.convert import load_pilarr

	palette_img = load_pilarr(palette_img)

	img = pil2cv(img)
	pal = pil2cv(palette_img)
	# droplerp_np(img, pal, 4, j.speed)
	retcv = maintain_colors(img, pal, mode)

	return retcv


def progrock_brightness(image,
                        enable_adjust_brightness=False,
                        high_brightness_threshold=180,
                        high_brightness_adjust_ratio=0.97,
                        high_brightness_adjust_fix_amount=2,
                        max_brightness_threshold=254,
                        low_brightness_threshold=40,
                        low_brightness_adjust_ratio=1.03,
                        low_brightness_adjust_fix_amount=2,
                        min_brightness_threshold=1):
	# @markdown Automatic Brightness Adjustment ------------------------------------------------------------
	# @markdown Automatically adjust image brightness when its mean value reaches a certain threshold\
	# @markdown Ratio means the vaue by which pixel values are multiplied when the thresjold is reached\
	# @markdown Fix amount is being directly added to\subtracted from pixel values to prevent oversaturation due to multiplications\
	# @markdown Fix amount is also being applied to border values defined by min\max threshold, like 1 and 254 to keep the image from having burnt out\pitch black areas while still being within set high\low thresholds

	# Credit: https://github.com/lowfuel/progrockdiffusion

	from PIL import ImageStat
	from PIL import ImageEnhance
	def get_stats(image):
		stat = ImageStat.Stat(image)
		brightness = sum(stat.mean) / len(stat.mean)
		contrast = sum(stat.stddev) / len(stat.stddev)
		return brightness, contrast

	brightness, contrast = get_stats(image)
	if brightness > high_brightness_threshold:
		filter = ImageEnhance.Brightness(image)
		image = filter.enhance(high_brightness_adjust_ratio)
		image = np.array(image)
		image = np.where(image > high_brightness_threshold,
		                 image - high_brightness_adjust_fix_amount, image).clip(
			0, 255).round().astype('uint8')
		image = PIL.Image.fromarray(image)
	if brightness < low_brightness_threshold:
		filter = ImageEnhance.Brightness(image)
		image = filter.enhance(low_brightness_adjust_ratio)
		image = np.array(image)
		image = np.where(image < low_brightness_threshold,
		                 image + low_brightness_adjust_fix_amount, image).clip(
			0, 255).round().astype('uint8')
		image = PIL.Image.fromarray(image)

	image = np.array(image)
	image = np.where(image > max_brightness_threshold,
	                 image - high_brightness_adjust_fix_amount, image).clip(0,
	                                                                        255).round().astype(
		'uint8')
	image = np.where(image < min_brightness_threshold,
	                 image + low_brightness_adjust_fix_amount, image).clip(0,
	                                                                       255).round().astype(
		'uint8')
	image = PIL.Image.fromarray(image)
	return image


def maxsize(w=None,
            h=None,
            grid=64):
	cw = session.width
	ch = session.height

	if w and h:
		session.width = w
		session.height = h
	elif w:
		# Set the width to the given value, and scale the height accordingly to preserve aspect ratio
		aspect = ch / cw
		session.width = w
		session.height = w * aspect
	elif h:
		# Set the height to the given value, and scale the width accordingly to preserve aspect ratio
		aspect = cw / ch
		session.height = h
		session.width = h * aspect

	session.width = int(session.width // grid * grid)
	session.height = int(session.height // grid * grid)

	return session.image.resize((session.width, session.height),
	                            PIL.Image.BICUBIC)


class VectorField:
	def __init__(self):
		# Grid of pixels, each pixel is a vector+energy
		self.w = 0
		self.h = 0
		self.grid = np.zeros((0, 0, 3), np.float32)

	def set_size(self, w, h):
		self.w = w
		self.h = h
		self.grid = np.zeros((w, h, 3), np.float32)

	def add_border(self, x, y):
		# Add energy to opposite borders in direction of (x,y)
		# TODO we should slerp

		# Left border
		self.grid[0, :, 0] = x
		self.grid[0, :, 1] = y
		self.grid[0, :, 2] += abs(x)

		# Right border
		self.grid[-1, :, 0] = x
		self.grid[-1, :, 1] = y
		self.grid[-1, :, 2] += abs(x)

		# Top border
		self.grid[:, 0, 0] = x
		self.grid[:, 0, 1] = y
		self.grid[:, 0, 2] += abs(y)

		# Bottom border
		self.grid[:, -1, 0] = x
		self.grid[:, -1, 1] = y
		self.grid[:, -1, 2] += abs(y)

	def update(self):
		"""
		Update the grid by moving the energy around to neighbor cells
		"""
		pass  # TODO vector field update


# def do_grid_ripples(v, ripple_amplitude, ripple_period, ripple_speed, mask):
#     w = v.w
#     h = v.h
#     flex_x = np.zeros((h, w), np.float32)
#     flex_y = np.zeros((h, w), np.float32)
#     for y in range(h):
#         for x in range(w):
#             # x += v.t*ripple_speed[0]
#             # y += v.t*ripple_speed[1]
#             cx = math.cos((x + v.t * ripple_speed[0]) / ripple_period[0]) * ripple_amplitude[0]
#             cy = math.sin((y + (v.t + 300) * ripple_speed[1]) / ripple_period[1]) * ripple_amplitude[1]

#             flex_x[y, x] = x + cx
#             flex_y[y, x] = y + cy

#     v.image_cv2 = cv2.remap(v.image_cv2, flex_x, flex_y, cv2.INTER_LINEAR)

#

colors = dict(
	black=(0, 0, 0),
	white=(255, 255, 255),
	red=(255, 0, 0),
	green=(0, 255, 0),
	blue=(0, 0, 255),
	yellow=(255, 255, 0),
	cyan=(0, 255, 255),
	magenta=(255, 0, 255),
	orange=(255, 165, 0),
	purple=(128, 0, 128),
	pink=(255, 192, 203),
	gray=(128, 128, 128),
	grey=(128, 128, 128),
	brown=(165, 42, 42),
	maroon=(128, 0, 0),
	navy=(0, 0, 128),
	teal=(0, 128, 128),
	olive=(128, 128, 0),
	darkgreen=(0, 100, 0),
	darkred=(139, 0, 0),
	darkblue=(0, 0, 139),
	darkgray=(169, 169, 169),
	darkgrey=(169, 169, 169),
)


def res_flow_cv2(resname, f=None, loop=False):
	if f is None:
		f = session.f

	flowdir = get_flow_compute_dirpath(resname)
	if ENABLE_FLOW_PRECOMPUTE and flowdir.exists():
		path = session.res_frame(flowdir, f, loop=loop)
		flow = load_precomputed_flow(path)

		return flow

	prev = session.res_frame_cv2(resname, f - 1, loop=loop)
	curr = session.res_frame_cv2(resname, f, loop=loop)
	flow = get_flow(prev, curr)

	return flow





