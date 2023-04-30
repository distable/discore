import cv2
import numpy as np1

import plugins
from classes import paths
from lib.printlib import trace_decorator
from src.party import maths, tricks
from src.party.lang import confmap
from src.rendering.hud import hud
from src.party import tricks
from src.party.tricks import img_to_hed
from src.party.audio import load_dbnorm
from src.party.maths import *
from src.party.pnodes import bake_prompt, eval_prompt
from src.party.signals import load_vr
from src import renderer

rv = renderer.rv
session = renderer.session

eps = np.finfo(float).eps


def init_media(demucs=True):
    img = session.res('init', ext=paths.image_exts)
    mus = session.res('init', ext=paths.audio_exts) or session.res('music', ext=paths.audio_exts)
    vid = session.res('init', ext=paths.video_exts)

    if vid:
        print(f"Init mode: video ({vid})")
        session.extract_frames(vid)
        init_music(demucs)
    elif img and mus:
        print(f"Init mode: image ({img}) + music ({mus})")
        init_music(demucs)
    elif mus:
        print(f"Init mode: music ({mus})")
        init_music(demucs)
    elif img:
        print(f"Init mode: image ({img})")
        init_music(demucs)

    if not session.res_music().exists():
        session.extract_music('init')

def init_music(demucs=False):
    # Only music, no instruments
    path = session.res_music()

    if demucs:
        if path.exists():
            music = load_dbnorm(path)
            tricks.demucs(path)

            piano = load_dbnorm(session.res(f'piano{paths.audio_ext}'))
            guitar = load_dbnorm(session.res(f'guitar{paths.audio_ext}'))
            other = load_dbnorm(session.res(f'other{paths.audio_ext}'))
            drum = load_dbnorm(session.res(f'drums{paths.audio_ext}'), 5)
            vocals = load_dbnorm(session.res(f'vocals{paths.audio_ext}'), 5)
            piano, guitar, other, drum, vocals = \
                np1.nan_to_num(piano), \
                    np1.nan_to_num(guitar), \
                    np1.nan_to_num(other), \
                    np1.nan_to_num(drum), \
                    np1.nan_to_num(vocals)


            rv.has_music = True
        else:
            print("No music file found, filling with zeroes ...")
            music = np1.zeros(rv.n)
            piano = np1.zeros(rv.n)
            guitar = np1.zeros(rv.n)
            other = np1.zeros(rv.n)
            drum = np1.zeros(rv.n)
            vocals = np1.zeros(rv.n)
            rv.has_music = False

        rv.save_signals(**locals())
        rv.has_demucs = True
        return music, drum, guitar, other, piano
    else:
        if path.exists():
            music = load_dbnorm(path)
            rv.has_music = True
        else:
            print("std.init_music: No music file found, filling with zeroes ...")
            music = np.zeros(rv.n)
            rv.has_music = False

        rv.has_demucs = False
        rv.save_signals(**locals())
        return music


def init_vr(rv, xy_vel, z_vel, r_vel):
    """
    Load the VR data into rv
    """
    # note: the sensor is not good enough for rotation with Oculus Rift S in my experience
    path = session.res('vr.json')
    if path.exists():
        vr = load_vr(path, beta=30.005, min_cutoff=0.15, one_euro=True)
        # rv.set_fps(vr.samples_per_second)

        x = vr.HeadPos.dd1 * xy_vel
        y = vr.HeadPos.dd2 * xy_vel
        z = clamp(vr.HeadPos.dd3 * z_vel, 0, 1)
        xl = vr.LPos.dd1 * xy_vel
        yl = vr.LPos.dd2 * xy_vel
        zl = vr.LPos.dd3 * z_vel
        xr = vr.RPos.dd1 * xy_vel
        yr = vr.RPos.dd2 * xy_vel
        zr = vr.RPos.dd3 * z_vel
        rl = vr.LRot.dd3 * r_vel
        rr = vr.RRot.dd3 * r_vel
        z = np.minimum(1, z)
        zl = np.minimum(1, zl)
        zr = np.minimum(1, zr)
        lpos = [(vr.LPos.d1[i], vr.LPos.d2[i]) for i in range(len(vr.LPos.d1))]
        rpos = [(vr.RPos.d1[i], vr.RPos.d2[i]) for i in range(len(vr.RPos.d1))]

        dt_camera = abs(xl / xy_vel) + \
                    abs(yl / xy_vel) + \
                    abs(zl / z_vel) + \
                    abs(xr / xy_vel) + \
                    abs(yr / xy_vel) + \
                    abs(zr / z_vel)
        dt_camera /= dt_camera.max()
    else:
        print("No VR.json found, filling with zeroes ...")
        n = rv.n
        x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
        xl, yl, zl, rl = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        xr, yr, zr, rr = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        dt_camera = np.zeros(n)

    rv.save_signals(**locals())
    return (x, y, z), (xl, yl, zl, rl), (xr, yr, zr, rr), dt_camera

last_seed = None
seed_frames_elapsed = 0

def render_switch(diffusers, get_guidance=None, get_detail=None, after_cn=None):
    global last_seed, seed_frames_elapsed

    if get_guidance is None:
        get_guidance = get_cn_guidance

    cn_render = tricks.schedule_01(rv.render_switch)
    if cn_render == 1:
        if get_detail is not None:
            rv.img = get_detail()
        guidance = get_guidance()

        if rv.depth is not None:
            rv.ccg4 = 0  # lerp(0.1, 0.2, abs(rv.depth.max() - rv.depth.min()))  # Maximize depth

        # Attenuate seed changes
        if last_seed != rv.seed:
            last_seed = rv.seed
            seed_frames_elapsed = 0
        else:
            seed_frames_elapsed += rv.dt

        rv.ccg1 += lerp(rv.seed_atten_ccg1, 0, seed_frames_elapsed / rv.seed_atten_time * rv.fps)
        rv.ccg2 += lerp(rv.seed_atten_ccg2, 0, seed_frames_elapsed / rv.seed_atten_time * rv.fps)
        rv.ccg3 += lerp(rv.seed_atten_ccg3, 0, seed_frames_elapsed / rv.seed_atten_time * rv.fps)

        rv.img = diffusers.txt2img_cn(
                # image=[canny, hed, last_frame_img],
                # ccg=[1.35, 0, chgtm - lerp(0, chgtm, chg)],
                # controlnet=['hed', 'temporal'],
                image=guidance,
                ccg=[rv.ccg1, rv.ccg2, rv.ccg3],  # 0.1 + sin1(v.t, 0.1, 0.75), ccg3],
                prompt=rv.prompt, promptneg=rv.promptneg,
                seed=rv.seed, steps=rv.steps, chg=0.05, w=rv.w, h=rv.h, cfg=rv.cfg, sampler=rv.sampler)

        if after_cn is not None:
            after_cn()

        # cc = tricks.palette(rv, rv.img, last_frame_img)
        # rv.img = tricks.alpha_blend(rv.img, cc, lerp(0.845, 0.5, rv.dtcam))

        # Shade the image with the depth
        # shade_depth()
    else:
        rv.seed = rv.nextseed
        rv.img = diffusers.img2img(
                image=rv.img,
                prompt=rv.prompt, promptneg=rv.promptneg,
                seed=rv.nextseed, steps=rv.steps, chg=rv.chg, w=rv.w, h=rv.h, cfg=rv.cfg)

def get_cn_guidance():
    current_frame_img = rv.img
    guidance_img1 = current_frame_img
    guidance_img2 = current_frame_img
    guidance_img3 = rv.depth

    init = session.res_frame_cv2('init')
    if init is not None:
        guidance_img1 = tricks.alpha_blend(guidance_img1, init, rv.init_ccg1)
        guidance_img2 = tricks.alpha_blend(guidance_img2, init, rv.init_ccg2)

        midas = plugins.get('midas')
        init_depth, tensor = midas.get_depth(init)
        guidance_img3 = tricks.alpha_blend(guidance_img3, init_depth, rv.init_ccg3)

    hud(injection1=rv.img_injection1, injection2=rv.img_injection2, injection3=rv.img_injection3)

    # canny = img_to_canny(guidance_img1, rv.canny_lo, rv.canny_hi)
    guidance_hed = img_to_hed(guidance_img1)

    return guidance_hed, guidance_img2, guidance_img3

def refresh_prompt(**nodes):
    from src.party import lang
    if rv.nprompt is None or rv.prompt != rv.prompt_last:
        maths.set_seed(rv.session.name, with_torch=False)
        dic = {**lang.__dict__, **nodes}
        rv.nprompt = bake_prompt(rv.prompt, confmap, dic)
        rv.prompt_last = rv.prompt

    rv.prompt = eval_prompt(rv.nprompt, rv.t)

# def rv_frame_detail():
# Edge detection
# edge_white = 0.0065 * jcurve(v.chg, 0.5)
# edge_black = 0.0150 * v.chg / 2
# do_canny_edge_reinforce(v, edge_white, edge_black)
# rv.img = grain(rv, rv.grain)
# rv.img = saltpepper(rv, rv.saltpepper)

# Detail generator
# dot_hsv = ((0, 1), (0.6, 1), (0, 1))
# draw_dot_matrix(v, spacing=32, radius=(10, 30), hsv=(0, 0, 0), opacity=(0, 0.5 * v.chg), dropout=0.85, noise=3, op='blend')
# draw_dot_matrix(v, spacing=16, radius=(1, 3), hsv=dot_hsv, opacity=(0.2, 0.985), dropout=0.925, noise=3)
# draw_dot_matrix(v, spacing=32, radius=(7, 30), hsv=(0, 0, 1), opacity=(0.0, 0.3), dropout=0.925, noise=3)

# Alignment dots
# draw_circles(v, v.w2, v.h2, 1, 3, 0, 1, 0, (0, 0, 1.0))
# draw_circles(v, v.w2 - v.w2 * 0.5, v.h2, 1, 3, 0, 1, 0, (0, 0, 1))
# draw_circles(v, v.w2 + v.w2 * 0.5, v.h2, 1, 3, 0, 1, 0, (0, 0, 1))
# draw_circles(v, v.w2, v.h2 - v.h2 * 0.5, 1, 3, 0, 1, 0, (0, 0, 1))
# draw_circles(v, v.w2, v.h2 + v.h2 * 0.5, 1, 3, 0, 1, 0, (0, 0, 1))

# multiply brightness with the mask
# hsv = cv2.cvtColor(v.image_cv2, cv2.COLOR_BGR2HSV).astype(np.float32)
# hsv[...,2] = hsv[...,2] * mask
# hsv[...,2] = np.clip(hsv[...,2], 0, 255)
# v.image_cv2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# pro.perlin(v, alpha=0.5)
# draw_sun_guide(v, v.w2, v.h2, 1 * v.chg, 1, ((0, 255), 1, 1))
# draw_sun_guide(v, v.w2 * 0.5, v.h2, 0.5, 1 * v.chg, ((0, 255), 1, 1))
# draw_sun_guide(v, v.w2 * 1.5, v.h2, 0.5, 1 * v.chg, ((0, 255), 1, 1))

# count = 30
# for i in range(count):
#     t = i / count
#     tw = t * v.w
#     a = 1 - abs(v.w2 - tw) / v.w2
#     a = rcurve(a, 0.15)
#     draw_sun_guide(v, tw, v.h2, a, t*0.25, ((0, 255), 1, 1))

# v.cfg += v.pkdrum[v.f] * 1.35

def sacade(amp=1):
    rv.x += nprng(-amp, amp)
    rv.y += nprng(-amp, amp)
    rv.rx += nprng(-amp, amp)
    rv.ry += nprng(-amp, amp)

def dtcam():
    rv.dtcam = rv.zero()
    rv.dtcam = np.maximum(rv.dtcam, rv.x / 30)
    rv.dtcam = np.maximum(rv.dtcam, rv.y / 30)
    rv.dtcam = np.maximum(rv.dtcam, rv.z / 60)
    rv.dtcam = np.clip(rv.dtcam, 0, 1)

def colors():
    hsv = cv2.cvtColor(rv.img, cv2.COLOR_RGB2HSV)
    if rv.hue > eps: tricks.hue(hsv, rv.hue)
    if rv.saturation > eps: tricks.saturation(hsv, rv.saturation)
    if rv.brightness > eps: tricks.brightness(hsv, rv.brightness)
    if rv.contrast > eps: tricks.contrast(hsv, rv.contrast)
    rv.img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

@trace_decorator
def rv_depth():
    midas = plugins.get('midas3d')
    depth, tensor = midas.get_depth(rv.img)
    newdepth = tricks.get_depth(rv.img)

    # print("----------------------------------------")
    # print("DEPTH TENSOR")
    # print(tensor)
    # print("")
    # print(tensor.shape)
    # print(tensor.dtype, tensor.min(), tensor.max())
    # print("----------------------------------------")
    # print("DEPTH")
    # print(depth)
    # print("")
    # print(depth.shape)
    # print(depth.dtype, depth.min(), depth.max())
    # print("----------------------------------------")
    # print("NEWDEPTH")
    # print(newdepth)
    # print("")
    # print(newdepth.shape)
    # print(newdepth.dtype, newdepth.min(), newdepth.max())
    # print("----------------------------------------")

    # print(depth.shape)
    # print(midas.get_depth(rv.img)[1].shape)
    # depth_inv = ImageOps.invert(convert.cv2pil(depth))
    # depth_inv = pil2cv(depth_inv)
    # depth_tensor = depth_inv
    rv.depth = depth
    rv.depth_tensor = tensor
    return depth, tensor

