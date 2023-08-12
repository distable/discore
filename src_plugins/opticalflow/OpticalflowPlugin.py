import argparse
import multiprocessing
import os
import pathlib
import shlex
import sys
from pathlib import Path

import PIL
import scipy
from python_color_transfer.color_transfer import ColorTransfer, Regrain
from tqdm import tqdm

from src import renderer
from src.classes import paths
from src.classes.Plugin import Plugin
from src.classes.Session import Session
from src.classes.convert import load_pil, load_pilarr, save_npy, save_png, load_cv2, load_npy
from src.lib import devices

import torch
import cv2
from PIL import Image, ImageOps
from torch.nn import functional as F

with_bidir = False
with_occlusion = False


def get_flow_paths(path_initframes):
	dir_flow_fwd = path_initframes / 'flow_fwd'
	dir_flow_bwd = path_initframes / 'flow_bwd'
	dir_flow_occ_fwd = path_initframes / 'flow_occ_fwd'
	dir_flow_occ_bwd = path_initframes / 'flow_occ_bwd'
	return dir_flow_fwd, dir_flow_bwd, dir_flow_occ_fwd, dir_flow_occ_bwd


def load_img(img, size, resize_mode='lanczos'):
	if resize_mode == 'lanczos':
		resize_mode = PIL.Image.LANCZOS

	img = PIL.Image.open(img).convert('RGB')
	if img.size != size:
		img = img.resize(size, resize_mode)

	return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()[None, ...].cuda()


# region Flow Visualization https://github.com/tomrunia/OpticalFlow_Visualization
# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np


def make_colorwheel():
	"""
	Generates a color wheel for optical flow visualization as presented in:
		Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
		URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
	Code follows the original C++ source code of Daniel Scharstein.
	Code follows the the Matlab source code of Deqing Sun.
	Returns:
		np.ndarray: Color wheel
	"""

	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR
	colorwheel = np.zeros((ncols, 3))
	col = 0

	# RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
	col = col + RY
	# YG
	colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
	colorwheel[col:col + YG, 1] = 255
	col = col + YG
	# GC
	colorwheel[col:col + GC, 1] = 255
	colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
	col = col + GC
	# CB
	colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
	colorwheel[col:col + CB, 2] = 255
	col = col + CB
	# BM
	colorwheel[col:col + BM, 2] = 255
	colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
	col = col + BM
	# MR
	colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
	colorwheel[col:col + MR, 0] = 255
	return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
	"""
	Applies the flow color wheel to (possibly clipped) flow components u and v.
	According to the C++ source code of Daniel Scharstein
	According to the Matlab source code of Deqing Sun
	Args:
		u (np.ndarray): Input horizontal flow of shape [H,W]
		v (np.ndarray): Input vertical flow of shape [H,W]
		convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
	Returns:
		np.ndarray: Flow visualization image of shape [H,W,3]
	"""
	flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
	colorwheel = make_colorwheel()  # shape [55x3]
	ncols = colorwheel.shape[0]
	rad = np.sqrt(np.square(u) + np.square(v))
	a = np.arctan2(-v, -u) / np.pi
	fk = (a + 1) / 2 * (ncols - 1)
	k0 = np.floor(fk).astype(np.int32)
	k1 = k0 + 1
	k1[k1 == ncols] = 0
	f = fk - k0
	for i in range(colorwheel.shape[1]):
		tmp = colorwheel[:, i]
		col0 = tmp[k0] / 255.0
		col1 = tmp[k1] / 255.0
		col = (1 - f) * col0 + f * col1
		idx = (rad <= 1)
		col[idx] = 1 - rad[idx] * (1 - col[idx])
		col[~idx] = col[~idx] * 0.75  # out of range
		# Note the 2-i => BGR instead of RGB
		ch_idx = 2 - i if convert_to_bgr else i
		flow_image[:, :, ch_idx] = np.floor(255 * col)
	return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
	"""
	Expects a two dimensional flow image of shape.
	Args:
		flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
		clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
		convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
	Returns:
		np.ndarray: Flow visualization image of shape [H,W,3]
	"""
	assert flow_uv.ndim == 3, 'input flow must have three dimensions'
	assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
	if clip_flow is not None:
		flow_uv = np.clip(flow_uv, 0, clip_flow)
	u = flow_uv[:, :, 0]
	v = flow_uv[:, :, 1]
	rad = np.sqrt(np.square(u) + np.square(v))
	rad_max = np.max(rad)
	epsilon = 1e-5
	u = u / (rad_max + epsilon)
	v = v / (rad_max + epsilon)
	return flow_uv_to_colors(u, v, convert_to_bgr)


# endregion


PT = ColorTransfer()
RG = Regrain()

latent_scale_schedule = [0, 0]  # controls coherency with previous frame in latent space. 0 is a good starting value. 1+ render slower, but may improve image coherency. 100 is a good value if you decide to turn it on.
init_scale_schedule = [0, 0]  # controls coherency with prev frame in pixel space. 0 - off, 1000 - a good starting value if you decide to turn it on.
warp_interp = PIL.Image.LANCZOS

mask_result = False  # imitates inpainting by leaving only inconsistent areas to be diffused
warp_strength = 1  # leave 1 for no change. 1.01 is already a strong value.

# @title Frame correction ------------------------------------------------------------
# @markdown Match frame pixels or latent to other frames to prevent oversaturation and feedback loop artifacts
# @markdown Latent matching
# @markdown Match the range of latent vector towards the 1st frame or a user defined range. Doesn't restrict colors, but may limit contrast.
normalize_latent = 'off'  # @param ['off', 'first_latent', 'user_defined']
# @markdown User defined stats to normalize the latent towards
latent_fixed_mean = 0.  # @param {'type':'raw'}
latent_fixed_std = 0.9  # @param {'type':'raw'}

latent_norm_4d = True  # @param {'type':'boolean'} @markdown Match latent on per-channel basis

# @markdown Video Optical Flow Settings: ------------------------------------------------------------
check_consistency = True  # @param {type: 'boolean'}##@param {type: 'number'} #0 - take next frame, 1 - take prev warped frame

# @title Consistency map mixing
# @markdown You can mix consistency map layers separately\
# @markdown missed_consistency_weight - masks pixels that have missed their expected position in the next frame \
# @markdown overshoot_consistency_weight - masks pixels warped from outside the frame\
# @markdown edges_consistency_weight - masks moving objects' edges\
# @markdown The default values to simulate previous versions' behavior are 1,1,1

missed_consistency_weight = 1  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
overshoot_consistency_weight = 1  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
edges_consistency_weight = 1  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}

# Inpaint occluded areas on top of raw frames. 0 - 0% inpainting opacity (no inpainting), 1 - 100% inpainting opacity. Other values blend between raw and inpainted frames.
inpaint_opacity = 0.5
# 0 - off, other values control effect opacity
match_color_strength = 0  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.1'}

# @markdown ###Color matching --------------------------------------------------------------------------------
# @markdown Color match frame towards stylized or raw init frame. Helps prevent images going deep purple. As a drawback, may lock colors to the selected fixed frame. Select stylized_frame with colormatch_offset = 0 to reproduce previous notebooks.
colormatch_frame = 'init_frame_offset'  # @param ['off', 'stylized_frame', 'init_frame', 'stylized_frame_offset', 'init_frame_offset']
# @markdown Color match strength. 1 mimics legacy behavior
color_match_frame_str = 1  # @param {'type':'number'}
# @markdown in offset mode, specifies the offset back from current frame, and 0 means current frame. In non-offset mode specifies the fixed frame number. 0 means the 1st frame.
colormatch_offset = 0  # @param {'type':'number'}
colormatch_method = 'LAB'  # @param ['LAB', 'PDF', 'mean']
colormatch_method_fn = PT.lab_transfer
if colormatch_method == 'LAB':
	colormatch_method_fn = PT.pdf_transfer
if colormatch_method == 'mean':
	colormatch_method_fn = PT.mean_std_transfer
# @markdown Match source frame's texture
colormatch_regrain = False  # @param {'type':'boolean'}

# @title Video mask settings ------------------------------------------------------------
# @markdown Check to enable background masking during render. Not recommended, better use masking when creating the output video for more control and faster testing.
use_background_mask = True  # @param {'type':'boolean'}
# @markdown Check to invert the mask.
invert_mask = False  # @param {'type':'boolean'}
# @markdown Apply mask right before feeding init image to the model. Unchecking will only mask current raw init frame.
apply_mask_after_warp = True  # @param {'type':'boolean'}
# @markdown Choose background source to paste masked stylized image onto: image, color, init video.
background = "init_video"  # @param ['image', 'color', 'init_video']
# @markdown Specify the init image path or color depending on your background source choice.
background_source = 'red'  # @param {'type':'string'}

# @title Video Masking ------------------------------------------------------------
mask_source = 'init_video'  # @param ['init_video','mask_video'] @markdown Generate background mask from your init video or use a video as a mask
extract_background_mask = True  # @param {'type':'boolean'} @markdown Check to rotoscope the video and create a mask from it. If unchecked, the raw monochrome video will be used as a mask.
mask_video_name = ''  # @param {'type':'string'} @markdown Specify path to a mask video for mask_video mode.

# @title Generate optical flow and consistency maps ------------------------------------------------------------
# @markdown Turbo Mode ------------------------------------------------------------
# @markdown (Starts after frame 1,) skips diffusion steps and just uses flow map to warp images for skipped frames.
# @markdown Speeds up rendering by 2x-4x, and may improve image coherence between frames. frame_blend_mode smooths abrupt texture changes across 2 frames.
# @markdown For different settings tuned for Turbo Mode, refer to the original Disco-Turbo Github: https://github.com/zippy731/disco-diffusion-turbo

turbo_mode = False  # @param {type:"boolean"}
turbo_steps = "3"  # @param ["2","3","4","5","6"] {type:"string"}
turbo_preroll = 1  # frames


class OpticalflowPlugin(Plugin):
	def __init__(self, dirpath: Path = None, id: str = None):
		super().__init__(dirpath, id)
		self.gmflow_model = None

	def title(self):
		return "opticalflow"

	def describe(self):
		return ""

	def init(self):
		sys.path.append(self.repo('RAFT').as_posix())
		sys.path.append(self.repo('RAFT/core').as_posix())

	def precompute(self, resname='init', force=False):
		session = renderer.session

		path_initframes = renderer.session.extract_frames(resname)
		dir_flow_fwd, \
			dir_flow_bwd, \
			dir_flow_occ_bwd, \
			dir_flow_occ_fwd = get_flow_paths(path_initframes)

		paths.mktree(dir_flow_fwd)
		if with_bidir:
			paths.mktree(dir_flow_bwd)
		if with_occlusion:
			paths.mktree(dir_flow_occ_fwd)
			paths.mktree(dir_flow_occ_bwd)

		# CREATE FLOW DATA
		# ------------------------------------------------------------
		flows = list(dir_flow_fwd.glob('*.npy'))

		frame_paths = sorted(path_initframes.glob('*.jpg'))
		if len(frame_paths) < 2:
			print(f'WARNING!\nCannot create flow maps: Found {len(frame_paths)} framepaths extracted from your video input.\nPlease check your video path.')

		if len(flows) == len(frame_paths) - 1:
			print("All flow maps already computed.")
			return

		if len(frame_paths) < 2:
			print("Only 1 frame.")
			return

		parser = argparse.ArgumentParser()
		parser.add_argument('--model', help="restore checkpoint")
		parser.add_argument('--dataset', help="dataset for evaluation")
		parser.add_argument('--small', action='store_true', help='use small model')
		parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
		parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
		args = parser.parse_args(['--mixed_precision'])

		for f in pathlib.Path(f'{dir_flow_bwd}').glob('*.*'):
			f.unlink()

		pairs = list(session.res_frameiter_pairs(path_initframes, load=False))
		pairs = []
		for p in pairs:
			if not p[3].with_suffix('.jpg').exists():
				pairs.append(p)
		if len(pairs) == 0:
			print("All flow maps already computed.")
			return

		num_processes = 3
		num_processes = min(len(pairs) - num_processes, num_processes)
		with tqdm(total=len(pairs),
		          desc=f'Generating optical flow maps with {num_processes} processes',
		          unit='frames') as pbar:
			pool = multiprocessing.Pool(processes=num_processes, initializer=process_create, initargs=[path_initframes])
			jobs = []
			for pair in pairs:
				result = pool.apply_async(process_pair, pair)
				jobs.append(result)
			pool.close()

			for result in jobs:
				try:
					result.get()
				except Exception as e:
					pass
				pbar.update(1)

			pool.join()

		frame_paths = sorted(path_initframes.glob('*.*'))
		if len(frame_paths) == 0:
			sys.exit("ERROR: 0 framepaths found.\nPlease check your video input path and rerun the video settings cell.")

		flows = list(dir_flow_bwd.glob('*.*'))
		if len(flows) == 0:
			sys.exit("ERROR: 0 flow files found.\nPlease rerun the flow generation cell.")

	def flow_setup(self):
		frame_num = renderer.session.f
		session = renderer.session
		img = renderer.rv.img

		if frame_num == 0 and use_background_mask:
			img = apply_mask(session, img, frame_num, background, background_source, invert_mask)

		return img

	def fetch_flow(self, resname):
		session = renderer.session
		rv = renderer.rv

		dir_frames = session.res_frame_dirpath(resname)
		path_flow = Session(dir_frames).res_frame('flow_fwd', rv.f, ext='npy')
		flow = load_npy(path_flow)
		return flow

	def flow(self, resname,
	         padmode='reflect',
	         padpct=0.1,
	         strength=1.0):
		rv = renderer.rv
		animpil = renderer.rv.image
		flowpath = renderer.session.res_frame(f'.{resname}', rv.f, subdir='flow_fwd', ext='npy')
		warped = warp(animpil, flowpath,
		              padmode=padmode,
		              padpct=padpct,
		              multiplier=strength)

		return warped

	def ccblend(self,
	            name: str = None, img: str = None, flow: str = None,
	            t: float = 0.5,
	            ccblur: int = 2,
	            ccstrength: float = 1.0,
	            cccolor: float = 1.0,
	            reverse: bool = False,
	            **kwargs):
		session = renderer.session

		img1 = renderer.rv.img
		img2 = session.res_framepil(img, resize=True)
		ccpath = get_consistency_path(session, flow, reverse=reverse)

		blended = blend(img1, img2, t,
		                ccpath=ccpath, ccstrength=ccstrength, ccblur=ccblur, cccolor=cccolor)

		if session.f == 1:
			return blended

		if mask_result:
			imgprev = session.res_framepil(session.f - 1, resize=True)

			diffuse_inpaint_mask_blur = 15
			diffuse_inpaint_mask_thresh = 220

			consistency_mask = load_cc(ccpath, blur=ccblur)
			consistency_mask = cv2.GaussianBlur(consistency_mask, (diffuse_inpaint_mask_blur, diffuse_inpaint_mask_blur), cv2.BORDER_DEFAULT)
			consistency_mask = np.where(consistency_mask < diffuse_inpaint_mask_thresh / 255., 0, 1.)
			consistency_mask = cv2.GaussianBlur(consistency_mask, (3, 3), cv2.BORDER_DEFAULT)

			# consistency_mask = torchvision.transforms.functional.resize(consistency_mask, image.size)
			print(imgprev.size, consistency_mask.shape, blended.size)
			cc_sz = consistency_mask.shape[1], consistency_mask.shape[0]
			image_masked = np.array(blended) * (1 - consistency_mask) + np.array(imgprev) * consistency_mask

			# image_masked = np.array(image.resize(cc_sz, warp_interp))*(1-consistency_mask) + np.array(init_img_prev.resize(cc_sz, warp_interp))*(consistency_mask)
			image_masked = PIL.Image.fromarray(image_masked.round().astype('uint8'))
			# image = image_masked.resize(image.size, warp_interp)
			image = image_masked

		return blended

	def consistency(self, image, resname):
		session = renderer.session
		f = session.f
		ccpath = session.res_frame(resname, f, 'flowcc21')
		# TODO
		return image


def get_consistency_path(session, resname, reverse=False):
	fwd = session.res_frame(resname, 'flowcc21')
	bwd = session.res_frame(resname, 'flowcc12')

	if reverse:
		return bwd
	else:
		return fwd


def warp(framepil, flowpath,
         padmode='reflect', padpct=0.1,
         multiplier=1.0):
	flow21 = np.load(flowpath)
	framearr = load_pilarr(framepil)

	pad = int(max(flow21.shape) * padpct)
	flow21 = np.pad(flow21, pad_width=((pad, pad), (pad, pad), (0, 0)), mode='constant')
	framearr = np.pad(framearr, pad_width=((pad, pad), (pad, pad), (0, 0)), mode=padmode)

	h, w = flow21.shape[:2]
	flow = flow21.copy()
	flow21[:, :, 0] += np.arange(w)
	flow21[:, :, 1] += np.arange(h)[:, np.newaxis]
	# print('flow stats', flow.max(), flow.min(), flow.mean())
	# print(flow)
	flow21 *= multiplier
	# print('flow stats mul', flow.max(), flow.min(), flow.mean())
	# res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
	res = cv2.remap(framearr, flow21, None, cv2.INTER_LANCZOS4)
	warpedarr21 = res
	warpedarr21 = warpedarr21[pad:warpedarr21.shape[0] - pad, pad:warpedarr21.shape[1] - pad, :]

	return PIL.Image.fromarray(warpedarr21.round().astype('uint8'))


def blend(img1, img2, t=0.5,
          ccpath=None, ccstrength=0.0, ccblur=2, cccolor=0.0,
          padmode='reflect', padpct=0.0) -> Image.Image:
	"""

	Args:
		img1: The start image.
		img2: The goal image.
		t: How much to blend the two images. 0.0 is all img1, 1.0 is all img2.
		ccpath: The path to the consistency mask.
		ccstrength: How much of the consistency mask to use.
		ccblur: Blur radius to soften the consistency mask. (Softens transition between raw video init and stylized frames in occluded areas)
		cccolor: Match color of inconsistent areas to unoccluded ones, after inconsistent areas were replaced with raw init video or inpainted. 0 to disable, 0 to 1 for strength.
		padmode: The padding mode to use.
		padpct: The padding percentage to use.

	Returns: The blended image.

	"""

	img1 = load_pilarr(img1)
	pad = int(max(img1.shape) * padpct)

	img2 = load_pilarr(img2, size=(img1.shape[1] - pad * 2, img1.shape[0] - pad * 2))
	# initarr = np.array(img2.convert('RGB').resize((flow21.shape[1] - pad * 2, flow21.shape[0] - pad * 2), warp_interp))
	t = 1.0

	if ccpath:
		ccweights = load_cc(ccpath, blur=ccblur)
		if cccolor:
			img2 = match_color(img1, img2, blend=cccolor)

		ccweights = ccweights.clip(1 - ccstrength, 1.)
		blended_w = img2 * (1 - t) + t * (img1 * ccweights + img2 * (1 - ccweights))
	else:
		if cccolor:
			img2 = match_color(img1, img2, blend=cccolor)

		blended_w = img2 * (1 - t) + img1 * t

	blended_w = PIL.Image.fromarray(blended_w.round().astype('uint8'))
	return blended_w


def match_color(stylized_img, raw_img, blend=1.0):
	img_arr_ref = cv2.cvtColor(np.array(stylized_img).round().astype('uint8'), cv2.COLOR_RGB2BGR)
	img_arr_in = cv2.cvtColor(np.array(raw_img).round().astype('uint8'), cv2.COLOR_RGB2BGR)
	# img_arr_in = cv2.resize(img_arr_in, (img_arr_ref.shape[1], img_arr_ref.shape[0]), interpolation=cv2.INTER_CUBIC )
	img_arr_col = PT.pdf_transfer(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
	img_arr_reg = RG.regrain(img_arr_in=img_arr_col, img_arr_col=img_arr_ref)

	blended = img_arr_reg * blend + img_arr_in * (1 - blend)
	blended = cv2.cvtColor(blended.round().astype('uint8'), cv2.COLOR_BGR2RGB)
	return blended


def match_color_var(stylized_img, raw_img, opacity=1., f=PT.pdf_transfer, regrain=False):
	img_arr_ref = cv2.cvtColor(np.array(stylized_img).round().astype('uint8'), cv2.COLOR_RGB2BGR)
	img_arr_in = cv2.cvtColor(np.array(raw_img).round().astype('uint8'), cv2.COLOR_RGB2BGR)
	img_arr_ref = cv2.resize(img_arr_ref, (img_arr_in.shape[1], img_arr_in.shape[0]), interpolation=cv2.INTER_CUBIC)
	img_arr_col = f(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
	if regrain: img_arr_col = RG.regrain(img_arr_in=img_arr_col, img_arr_col=img_arr_ref)
	img_arr_col = img_arr_col * opacity + img_arr_in * (1 - opacity)
	img_arr_reg = cv2.cvtColor(img_arr_col.round().astype('uint8'), cv2.COLOR_BGR2RGB)

	return img_arr_reg


def load_cc(path: Image.Image | Path | str, blur=2):
	ccpil = PIL.Image.open(path)
	multilayer_weights = np.array(ccpil) / 255
	weights = np.ones_like(multilayer_weights[..., 0])
	weights *= multilayer_weights[..., 0].clip(1 - missed_consistency_weight, 1)
	weights *= multilayer_weights[..., 1].clip(1 - overshoot_consistency_weight, 1)
	weights *= multilayer_weights[..., 2].clip(1 - edges_consistency_weight, 1)

	if blur > 0: weights = scipy.ndimage.gaussian_filter(weights, [blur, blur])
	weights = np.repeat(weights[..., None], 3, axis=2)

	if DEBUG: print('weight min max mean std', weights.shape, weights.min(), weights.max(), weights.mean(), weights.std())
	return weights


def apply_mask(fg: Path | Image.Image, bg: Path | Image.Image, mask: Path | Image.Image, invert=False):
	# NOTE: this was used for applying background
	# Get the size we're working with
	size = (1, 1)
	if isinstance(fg, Image.Image): size = fg.size
	if isinstance(bg, Image.Image): size = bg.size
	if isinstance(mask, Image.Image): size = mask.size

	# Get the images
	fg = load_pil(fg, size)
	bg = load_pil(bg, size)
	mask = load_pil(mask, size).convert('L')
	if invert:
		mask = PIL.ImageOps.invert(mask)

	# Composite everything
	bg.paste(fg, (0, 0), mask)
	return bg


# implemetation taken from https://github.com/lowfuel/progrockdiffusion


# def run_consistency(image, frame_num):
# 	if mask_result and check_consistency and frame_num > 0:
# 		diffuse_inpaint_mask_blur = 15
# 		diffuse_inpaint_mask_thresh = 220
# 		print('imitating inpaint')
# 		frame1_path = f'{path_init_video_frames}/{frame_num:06}.jpg'
# 		weights_path = f"{flo_fwd_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
# 		consistency_mask = load_cc(weights_path, blur=consistency_blur)
# 		consistency_mask = cv2.GaussianBlur(consistency_mask,
# 		                                    (diffuse_inpaint_mask_blur, diffuse_inpaint_mask_blur), cv2.BORDER_DEFAULT)
# 		consistency_mask = np.where(consistency_mask < diffuse_inpaint_mask_thresh / 255., 0, 1.)
# 		consistency_mask = cv2.GaussianBlur(consistency_mask,
# 		                                    (3, 3), cv2.BORDER_DEFAULT)
#
# 		# consistency_mask = torchvision.transforms.functional.resize(consistency_mask, image.size)
# 		init_img_prev = PIL.Image.open(init_image)
# 		print(init_img_prev.size, consistency_mask.shape, image.size)
# 		cc_sz = consistency_mask.shape[1], consistency_mask.shape[0]
# 		image_masked = np.array(image) * (1 - consistency_mask) + np.array(init_img_prev) * (consistency_mask)
#
# 		# image_masked = np.array(image.resize(cc_sz, warp_interp))*(1-consistency_mask) + np.array(init_img_prev.resize(cc_sz, warp_interp))*(consistency_mask)
# 		image_masked = PIL.Image.fromarray(image_masked.round().astype('uint8'))
# 		# image = image_masked.resize(image.size, warp_interp)
# 		image = image_masked
# 	return image


def chunk_list(lst, chunk_size):
	return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


# def do_run():
# 	# if (args.animation_mode == 'Video Input') and (args.midas_weight > 0.0):
# 	# midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model(args.midas_depth_model)
#
# 	for i in range(args.n_batches):
# 		# gc.collect()
# 		# torch.cuda.empty_cache()
# 		steps = get_scheduled_arg(frame_num, steps_schedule)
# 		style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
# 		skip_steps = int(steps - steps * style_strength)
# 		cur_t = diffusion.num_timesteps - skip_steps - 1
# 		total_steps = cur_t
#
# 		consistency_mask = None
# 		if check_consistency and frame_num > 0:
# 			frame1_path = f'{path_init_video_frames}/{frame_num:06}.jpg'
# 			if reverse_cc_order:
# 				weights_path = f"{flo_fwd_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
# 			else:
# 				weights_path = f"{flo_fwd_folder}/{frame1_path.split('/')[-1]}_12-21_cc.jpg"
#
# 			consistency_mask = load_cc(weights_path, blur=consistency_blur)
#
#
# #         for k, image in enumerate(sample['pred_xstart']):
# #             # tqdm.write(f'Batch {i}, step {j}, output {k}:')
# #             current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
# #             percent = math.ceil(j/total_steps*100)
# #             if args.n_batches > 0:
# #               #if intermediates are saved to the subfolder, don't append a step or percentage to the name
# #               if (cur_t == -1 or cur_t == stop_early-1) and args.intermediates_in_subfolder is True:
# #                 save_num = f'{frame_num:06}' if animation_mode != "None" else i
# #     filename = f'{args.batch_name}({args.batchNum})_{save_num}.png'
# #   else:
# #     #If we're working with percentages, append it
# #     if args.steps_per_checkpoint is not None:
# #       filename = f'{args.batch_name}({args.batchNum})_{i:06}-{percent:02}%.png'
# #     # Or else, iIf we're working with specific steps, append those
# #     else:
# #       filename = f'{args.batch_name}({args.batchNum})_{i:06}-{j:03}.png'
#
# # image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
# # if frame_num > 0:
# #   print('times per image', o); o+=1
# #   image = PIL.Image.fromarray(match_color_var(first_frame, image, f=PT.lab_transfer))
# #   # image.save(f'/content/{frame_num}_{cur_t}_{o}.jpg')
# #   # image = PIL.Image.fromarray(match_color_var(first_frame, image))
#
# # #reapply init image on top of
#
# # if j % args.display_rate == 0 or cur_t == -1 or cur_t == stop_early-1:
# #   image.save('progress.png')
# #   display.clear_output(wait=True)
# #   display.display(display.Image('progress.png'))
# # if args.steps_per_checkpoint is not None:
# #   if j % args.steps_per_checkpoint == 0 and j > 0:
# #     if args.intermediates_in_subfolder is True:
# #       image.save(f'{partialFolder}/{filename}')
# #     else:
# #       image.save(f'{sessionDir}/{filename}')
# # else:
# #   if j in args.intermediate_saves:
# #     if args.intermediates_in_subfolder is True:
# #       image.save(f'{partialFolder}/{filename}')
# #     else:
# #       image.save(f'{sessionDir}/{filename}')
# # if (cur_t == -1) | (cur_t == stop_early-1):
# #   if cur_t == stop_early-1: print('early stopping')
# # if frame_num == 0:
# #   save_settings()
# # if args.animation_mode != "None":
# #   # sys.exit(os.getcwd(), 'cwd')
# #   image.save('prevFrame.png')
# # image.save(f'{sessionDir}/{filename}')
# # if args.animation_mode == 'Video Input':
# #   # If turbo, save a blended image
# #   if turbo_mode and frame_num > 0:
# #     # Mix new image with prevFrameScaled
# #     blend_factor = (1)/int(turbo_steps)
# #     newFrame = cv2.imread('prevFrame.png') # This is already updated..
# #     prev_frame_warped = cv2.imread('prevFrameScaled.png')
# #     blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1-blend_factor), 0.0)
# #     cv2.imwrite(f'{sessionDir}/{filename}',blendedImage)
# #   else:
# #     image.save(f'{sessionDir}/{filename}')
#
# # if frame_num != args.max_frames-1:
# #   display.clear_output()
#

def process_create(_init_frames):
	global model, path
	print("Creating process...")
	model = init_gmflow()
	path = _init_frames
	print("Process created.")


def process_pair(i, frame1, frame2, in_frame1, in_frame2):
	global model, path

	# print(f"{multiprocessing.current_process().name} Processing pair entry {i}: {in_frame1.stem} / {in_frame2.stem}...")

	dir_flow_fwd, \
		dir_flow_bwd, \
		dir_flow_occ_fwd, \
		dir_flow_occ_bwd = get_flow_paths(path)

	out_flow_fwd = dir_flow_fwd / in_frame1.stem
	out_flow_bwd = dir_flow_bwd / in_frame1.stem
	out_occ_fwd = (dir_flow_occ_fwd / in_frame1.stem).with_suffix('.png')
	out_occ_bwd = (dir_flow_occ_bwd / in_frame1.stem).with_suffix('.png')

	if out_flow_fwd.with_suffix('.png').exists():
		return

	frame1 = load_cv2(in_frame1)
	frame2 = load_cv2(in_frame2)

	with torch.no_grad():
		flow_fwd, flow_bwd, fwd_occ, bwd_occ = calc_flow(frame1, frame2, with_bidir, with_occlusion, model)

	save_npy(out_flow_fwd.with_suffix('.npy'), flow_fwd)
	im_flow_fwd = flow_to_image(flow_fwd)
	save_png(im_flow_fwd, out_flow_fwd.with_suffix('.png'), True)

	if with_bidir:
		save_npy(out_flow_bwd.with_suffix('.npy'), flow_bwd)
		im_flow_bwd = flow_to_image(flow_bwd)
	# save_png(im_flow_bwd, out_flow_bwd.with_suffix('.png'), True)

	if with_occlusion:
		im_occ_bwd = Image.fromarray((bwd_occ[0].cpu().numpy() * 255.).astype(np.uint8))
		im_occ_fwd = Image.fromarray((fwd_occ[0].cpu().numpy() * 255.).astype(np.uint8))
		save_png(im_occ_fwd, out_occ_fwd, True)
		save_png(im_occ_bwd, out_occ_bwd, True)


def init_gmflow():
	from plug_repos.opticalflow.gmflow.gmflow.gmflow import GMFlow

	args = get_gmflow_args()
	model = GMFlow(feature_channels=args.feature_channels,
	               num_scales=args.num_scales,
	               upsample_factor=args.upsample_factor,
	               num_head=args.num_head,
	               attention_type=args.attention_type,
	               ffn_dim_expansion=args.ffn_dim_expansion,
	               num_transformer_layers=args.num_transformer_layers,
	               ).to(devices.device)

	if args.resume:
		print('Load checkpoint: %s' % args.resume)

		loc = 'cuda:{}'.format(args.local_rank)
		checkpoint = torch.load(args.resume, map_location=loc)

		weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

		model.load_state_dict(weights, strict=args.strict_resume)

	return model




def calc_flow(frame1, frame2, bidir=None, occ=None, model=None):
	flow, flow_bwd, fwd_occ, bwd_occ = None, None, None, None

	from plug_repos.opticalflow.gmflow.gmflow.geometry import forward_backward_consistency_check
	from plug_repos.opticalflow.gmflow.utils.utils import InputPadder

	args = get_gmflow_args()
	model.eval()

	if occ:
		assert bidir

	if len(frame1.shape) == 2:  # gray image, for example, HD1K
		frame1 = np.tile(frame1[..., None], (1, 1, 3))
		frame2 = np.tile(frame2[..., None], (1, 1, 3))
	else:
		frame1 = frame1[..., :3]
		frame2 = frame2[..., :3]

	frame1 = torch.from_numpy(frame1).permute(2, 0, 1).float()
	frame2 = torch.from_numpy(frame2).permute(2, 0, 1).float()

	if args.inference_size is None:
		padder = InputPadder(frame1.shape, padding_factor=args.padding_factor)
		frame1, frame2 = padder.pad(frame1[None].cuda(), frame2[None].cuda())
	else:
		frame1, frame2 = frame1[None].cuda(), frame2[None].cuda()

	# resize before inference
	if args.inference_size is not None:
		assert isinstance(args.inference_size, list) or isinstance(args.inference_size, tuple)
		ori_size = frame1.shape[-2:]
		frame1 = F.interpolate(frame1, size=args.inference_size, mode='bilinear', align_corners=True)
		frame2 = F.interpolate(frame2, size=args.inference_size, mode='bilinear', align_corners=True)

	results_dict = model(frame1, frame2,
	                     attn_splits_list=args.attn_splits_list,
	                     corr_radius_list=args.corr_radius_list,
	                     prop_radius_list=args.prop_radius_list,
	                     pred_bidir_flow=bidir,
	                     )

	flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

	# resize back
	if args.inference_size is not None:
		flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear', align_corners=True)
		flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / args.inference_size[-1]
		flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / args.inference_size[-2]

	if args.inference_size is None:
		flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
	else:
		flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

	# also predict backward flow
	if bidir:
		assert flow_pr.size(0) == 2  # [2, H, W, 2]

		if args.inference_size is None:
			flow_bwd = padder.unpad(flow_pr[1]).permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
		else:
			flow_bwd = flow_pr[1].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

		# forward-backward consistency check
		# occlusion is 1
		if occ:
			if args.inference_size is None:
				fwd_flow = padder.unpad(flow_pr[0]).unsqueeze(0)  # [1, 2, H, W]
				bwd_flow = padder.unpad(flow_pr[1]).unsqueeze(0)  # [1, 2, H, W]
			else:
				fwd_flow = flow_pr[0].unsqueeze(0)
				bwd_flow = flow_pr[1].unsqueeze(0)

			fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow)  # [1, H, W] float

	return flow, flow_bwd, fwd_occ, bwd_occ


def get_gmflow_args():
	parser = argparse.ArgumentParser()
	# dataset
	parser.add_argument('--checkpoint_dir', default='tmp', type=str,
	                    help='where to save the training log and models')
	parser.add_argument('--stage', default='chairs', type=str,
	                    help='training stage')
	parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+',
	                    help='image size for training')
	parser.add_argument('--padding_factor', default=16, type=int,
	                    help='the input should be divisible by padding_factor, otherwise do padding')
	parser.add_argument('--max_flow', default=400, type=int,
	                    help='exclude very large motions during training')
	parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+',
	                    help='validation dataset')
	parser.add_argument('--with_speed_metric', action='store_true',
	                    help='with speed metric when evaluation')
	# training
	parser.add_argument('--lr', default=4e-4, type=float)
	parser.add_argument('--batch_size', default=12, type=int)
	parser.add_argument('--num_workers', default=4, type=int)
	parser.add_argument('--weight_decay', default=1e-4, type=float)
	parser.add_argument('--grad_clip', default=1.0, type=float)
	parser.add_argument('--num_steps', default=100000, type=int)
	parser.add_argument('--seed', default=326, type=int)
	parser.add_argument('--summary_freq', default=100, type=int)
	parser.add_argument('--val_freq', default=10000, type=int)
	parser.add_argument('--save_ckpt_freq', default=10000, type=int)
	parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)
	# resume pretrained model or resume training
	parser.add_argument('--resume', default=None, type=str,
	                    help='resume from pretrain model for finetuing or resume from terminated training')
	parser.add_argument('--strict_resume', action='store_true')
	parser.add_argument('--no_resume_optimizer', action='store_true')
	# GMFlow model
	parser.add_argument('--num_scales', default=1, type=int,
	                    help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
	parser.add_argument('--feature_channels', default=128, type=int)
	parser.add_argument('--upsample_factor', default=8, type=int)
	parser.add_argument('--num_transformer_layers', default=6, type=int)
	parser.add_argument('--num_head', default=1, type=int)
	parser.add_argument('--attention_type', default='swin', type=str)
	parser.add_argument('--ffn_dim_expansion', default=4, type=int)
	parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
	                    help='number of splits in attention')
	parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
	                    help='correlation radius for matching, -1 indicates global matching')
	parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
	                    help='self-attention radius for flow propagation, -1 indicates global attention')
	# loss
	parser.add_argument('--gamma', default=0.9, type=float, help='loss weight')
	# evaluation
	parser.add_argument('--eval', action='store_true')
	parser.add_argument('--save_eval_to_file', action='store_true')
	parser.add_argument('--evaluate_matched_unmatched', action='store_true')
	# inference on a directory
	parser.add_argument('--inference_dir', default=None, type=str)
	parser.add_argument('--inference_size', default=None, type=int, nargs='+', help='can specify the inference size')
	parser.add_argument('--dir_paired_data', action='store_true', help='Paired data in a dir instead of a sequence')
	parser.add_argument('--save_flo_flow', action='store_true')
	# predict on sintel and kitti test set for submission
	parser.add_argument('--submission', action='store_true',
	                    help='submission to sintel or kitti test sets')
	parser.add_argument('--output_path', default='output', type=str,
	                    help='where to save the prediction results')
	parser.add_argument('--save_vis_flow', action='store_true',
	                    help='visualize flow prediction as .png image')
	parser.add_argument('--no_save_flo', action='store_true',
	                    help='not save flow as .flo')
	# distributed training
	parser.add_argument('--local_rank', default=0, type=int)
	parser.add_argument('--distributed', action='store_true')
	parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
	parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')
	parser.add_argument('--count_time', action='store_true',
	                    help='measure the inference time on sintel')

	from src import plugins
	args = parser.parse_args(shlex.split(
		# "--padding_factor 32 "
		# "--upsample_factor 4 "
		# "--num_scales 2 "
		# "--attn_splits_list 4 16 "
		# "--corr_radius_list -1 16 "
		# "--prop_radius_list -1 16 "
		f"--resume {plugins.get('opticalflow').res('gmflow_sintel-0c07dcb3.pth').as_posix()}"))
	return args
