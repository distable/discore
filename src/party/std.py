"""
This script implements bread & butter techniques for AI animation,
it's the core of all my renders and essentially the standard render
process
"""
import shlex
from pathlib import Path
from typing import Optional, Callable, List, Tuple

import cv2
import numpy as np1
import torch

import jargs
from scripts.interfaces import diffusers_lib
from src import plugins, renderer
from src.classes import convert, paths
from src.classes.common import first_non_null
from src.lib import loglib
from src.lib.corelib import shlexrun
from src.lib.loglib import trace_decorator_noargs
from src.party import audio, maths, signals, tricks
from src.party.audio import load_lufs
from src.party.lang import confmap
from src.party.maths import *
from src.party.ravelang import pnodes
from src.rendering import hud

rv = renderer.rv
session = renderer.session

eps = np.finfo(float).eps

do_init_flow = tricks.do_init_flow

log: callable = loglib.make_log('std')
logerr: callable = loglib.make_logerr('std')


def init_img_exists():
    img, mus, vid = session.res_init()
    return img is not None or vid is not None


def init_media(demucs=True):
    img = session.res('init', extensions=paths.image_exts)
    mus = session.res('init', extensions=paths.audio_exts) or session.res('music', extensions=paths.audio_exts)
    vid = session.res('init', extensions=paths.video_exts)

    if vid:
        log(f"Init mode: image ({img}) + music ({mus})")
        init_music(demucs)
    elif mus:
        log(f"Init mode: music ({mus})")
        init_music(demucs)
    elif img:
        log(f"Init mode: image ({img})")
        init_music(demucs)


def get_demucs_paths(dirpath):
    pianos = dirpath / f'.demucs/piano{paths.audio_ext}'
    drums = dirpath / f'.demucs/drums{paths.audio_ext}'
    bass = dirpath / f'.demucs/bass{paths.audio_ext}'
    other = dirpath / f'.demucs/other{paths.audio_ext}'
    vocals = dirpath / f'.demucs/vocals{paths.audio_ext}'
    guitar = dirpath / f'.demucs/guitar{paths.audio_ext}'
    return pianos, bass, drums, guitar, other, vocals


def run_demucs(fpath):
    if jargs.args.remote:
        log("Demucs not supported in remote mode. Run locally first and generate the audio data, then deploy again to send that data.")
        return

    fpath = Path(fpath)
    dirpath = fpath.parent

    piano, bass, drums, guitar, other, vocals = get_demucs_paths(dirpath)

    if piano.exists() and drums.exists() and bass.exists() and other.exists() and vocals.exists() and guitar.exists():
        return

    log("Running Demucs ...")

    model = 'htdemucs_6s'

    import demucs.separate
    demucs_args = shlex.split(f"-n {model} {fpath.as_posix()} -o {dirpath.as_posix()}")
    demucs.separate.main(demucs_args)

    dst = dirpath / model / fpath.stem
    for f in dst.iterdir():
        paths.rm(dirpath / f.name)
        # shutil.move(f, dirpath)

        outpath = f'{piano.parent}/{f.stem}{paths.audio_ext}'
        paths.mktree(outpath)
        shlexrun(f"ffmpeg -i '{f}' -acodec {paths.audio_codec} -b:a 224k '{outpath}' -y", silent=True)
        if f is not None:  # wat
            f.unlink()

    paths.rmtree(dirpath / model)


def init_music(demucs=False):
    # Only music, no instruments
    path = session.res_music()
    rv.music = np1.zeros(rv.n)
    rv.piano = np1.zeros(rv.n)
    rv.bass = np1.zeros(rv.n)
    rv.guitar = np1.zeros(rv.n)
    rv.other = np1.zeros(rv.n)
    rv.drum = np1.zeros(rv.n)
    rv.vocals = np1.zeros(rv.n)

    if path.exists():
        # rv.has_music = True
        # rv.music = load_lufs(path)

        if demucs:
            rv.has_demucs = True
            run_demucs(path)

            # fpiano = session.res(f'.demucs/piano{paths.audio_ext}')
            # fbass = session.res(f'.demucs/bass{paths.audio_ext}')
            # fguitar = session.res(f'.demucs/guitar{paths.audio_ext}')
            # fother = session.res(f'.demucs/other{paths.audio_ext}')
            # fdrum = session.res(f'.demucs/drums{paths.audio_ext}')
            # fvocals = session.res(f'.demucs/vocals{paths.audio_ext}')
            fpiano, fbass, fdrum, fguitar, fother, fvocals = get_demucs_paths(path.parent)

            rv.piano = load_lufs(fpiano)
            rv.bass = load_lufs(fbass)
            rv.guitar = load_lufs(fguitar)
            rv.other = load_lufs(fother)
            rv.drum = load_lufs(fdrum, 5)
            rv.vocals = load_lufs(fvocals, 5)

            # rv.pca1, rv.pca2, rv.pca3 = audio.load_pca(path)
            # rv.piano_pca1, rv.piano_pca2, rv.piano_pca3 = audio.load_pca(fpiano)
            # rv.bass_pca1, rv.bass_pca2, rv.bass_pca3 = audio.load_pca(fbass)
            # rv.guitar_pca1, rv.guitar_pca2, rv.guitar_pca3 = audio.load_pca(fguitar)
            # rv.other_pca1, rv.other_pca2, rv.other_pca3 = audio.load_pca(fother)
            # rv.drum_pca1, rv.drum_pca2, rv.drum_pca3 = audio.load_pca(fdrum)
            # rv.vocals_pca1, rv.vocals_pca2, rv.vocals_pca3 = audio.load_pca(fvocals)

            pca = audio.load_pca(path, 3)
            for i, pc in enumerate(pca):
                rv.set_signal(f'pca{i + 1}', norm(pc))

            # rv.piano_pca1, = audio.load_pca(fpiano, 1)
            rv.bass_pca1, = audio.load_pca(fbass, 1)
            # rv.guitar_pca1, = audio.load_pca(fguitar, 1)
            # rv.other_pca1, = audio.load_pca(fother, 1)
            # rv.drum_pca1, = audio.load_pca(fdrum, 1)
            rv.vocals_pca1, = audio.load_pca(fvocals, 1)

            rv.onset = audio.load_onset(path)
            # rv.piano_onset = audio.load_onset(fpiano)
            # rv.bass_onset = audio.load_onset(fbass)
            # rv.guitar_onset = audio.load_onset(fguitar)
            # rv.other_onset = audio.load_onset(fother)
            rv.drum_onset = audio.load_onset(fdrum)
            rv.vocals_onset = audio.load_onset(fvocals)

            rv.flatness = audio.load_flatness(path)
        # rv.piano_flatness = audio.load_flatness(fpiano)
        # rv.bass_flatness = audio.load_flatness(fbass)
        # rv.guitar_flatness = audio.load_flatness(fguitar)
        # rv.other_flatness = audio.load_flatness(fother)
        # rv.drum_flatness = audio.load_flatness(fdrum)
        # rv.vocals_flatness = audio.load_flatness(fvocals)

        # rv.changes, rv.onset, rv.beat, rv.chroma, rv.spectral, rv.mfcc, rv.bandwidth, rv.flatness, rv.centroid, rv.sentiment = audio.load_rosa(path)
        # rv.piano_changes, rv.piano_onset, rv.piano_beat, rv.piano_chroma, rv.piano_spectral, rv.piano_mfcc, rv.piano_bandwidth, rv.piano_flatness, rv.piano_centroid, rv.piano_sentiment = audio.load_rosa(fpiano)
        # rv.bass_changes, rv.bass_onset, rv.bass_beat, rv.bass_chroma, rv.bass_spectral, rv.bass_mfcc, rv.bass_bandwidth, rv.bass_flatness, rv.bass_centroid, rv.bass_sentiment = audio.load_rosa(fbass)
        # rv.guitar_changes, rv.guitar_onset, rv.guitar_beat, rv.guitar_chroma, rv.guitar_spectral, rv.guitar_mfcc, rv.guitar_bandwidth, rv.guitar_flatness, rv.guitar_centroid, rv.guitar_sentiment = audio.load_rosa(fguitar)
        # rv.other_changes, rv.other_onset, rv.other_beat, rv.other_chroma, rv.other_spectral, rv.other_mfcc, rv.other_bandwidth, rv.other_flatness, rv.other_centroid, rv.other_sentiment = audio.load_rosa(fother)
        # rv.drum_changes, rv.drum_onset, rv.drum_beat, rv.drum_chroma, rv.drum_spectral, rv.drum_mfcc, rv.drum_bandwidth, rv.drum_flatness, rv.drum_centroid, rv.drum_sentiment = audio.load_rosa(fdrum)
        # rv.vocals_changes, rv.vocals_onset, rv.vocals_beat, rv.vocals_chroma, rv.vocals_spectral, rv.vocals_mfcc, rv.vocals_bandwidth, rv.vocals_flatness, rv.vocals_centroid, rv.vocals_sentiment = audio.load_rosa(fvocals)
        else:
            rv.has_demucs = False
    else:
        log("std.init_music: No music file found, filling with zeros ...")
        rv.has_music = False


def load_vr(xy_vel=1000, z_vel=500, r_vel=1 / 15):
    """
    Load the VR data into rv
    """
    # note: the sensor is not good enough for rotation with Oculus Rift S in my experience
    path = session.res('vr.json')
    if path:
        # vr = load_vr(path, beta=30.005, min_cutoff=0.15, one_euro=True)
        vr = signals.load_vr(path, beta=0.25, min_cutoff=0.004, one_euro=True)

        x = vr.HeadPos.dd1 * xy_vel
        y = vr.HeadPos.dd2 * xy_vel
        z = vr.HeadPos.dd3 * z_vel
        rx = vr.HeadRot.dd1 * r_vel
        ry = vr.HeadRot.dd2 * r_vel
        # rz = vr.HeadRot.dd3 * r_vel
        xl = vr.LPos.dd1 * xy_vel
        yl = vr.LPos.dd2 * xy_vel
        zl = vr.LPos.dd3 * z_vel
        xr = vr.RPos.dd1 * xy_vel
        yr = vr.RPos.dd2 * xy_vel
        zr = vr.RPos.dd3 * z_vel
        rl = vr.LRot.dd3 * r_vel
        rr = vr.RRot.dd3 * r_vel

        if renderer.enable_dev:
            x += rx / r_vel
            y += ry / r_vel

        # z = np.minimum(1, z)
        # zl = np.minimum(1, zl)
        # zr = np.minimum(1, zr)
        lpos = [(vr.LPos.d1[i], vr.LPos.d2[i]) for i in range(len(vr.LPos.d1))]
        rpos = [(vr.RPos.d1[i], vr.RPos.d2[i]) for i in range(len(vr.RPos.d1))]

        dt_vr = abs(xl / xy_vel) + \
                abs(yl / xy_vel) + \
                abs(zl / z_vel) + \
                abs(xr / xy_vel) + \
                abs(yr / xy_vel) + \
                abs(zr / z_vel)
        dt_vr /= dt_vr.max()
    else:
        log("No VR.json found, filling with zeroes ...")
        n = rv.n
        x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
        xl, yl, zl, rl = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        xr, yr, zr, rr = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        dt_vr = np.zeros(n)

    # rv.save_signals(**locals())
    return (x, y, z), (xl, yl, zl, rl), (xr, yr, zr, rr), dt_vr


last_seed = None
seed_frames_elapsed = 0
frames_since_cn = 0


def is_cn_switch():
    global frames_since_cn
    every, _ = tricks.get_v1v2_schedule_stride(rv.cn_switch)
    ret = tricks.schedule_01(rv.cn_switch) == 1 or frames_since_cn >= every

    frames_since_cn += 1
    if ret:
        frames_since_cn = 0
    return ret


def save_guidance_imgs(guidance, frequency=1):
    # It can produce too much data to download quickly enough and we may fall behind on frames, because ssh tunnel can be slow as shit with a bunch of tiny files
    if jargs.args.remote and frequency < 5:
        frequency = 5

    if rv.f % frequency == 0:
        # save each guidance image
        log("Saving guidance imgs...")
        for i, im_guidance in enumerate(guidance):
            hud.snap(f'ccg_img{i + 1}', im_guidance)
            convert.save_img(im_guidance, session.res(f'dbg_guidance_{i}.jpg', return_missing=True), with_async=True)


prompt_last = None


def refresh_prompt(**kwargs):
    set_prompt(kwargs)


@trace_decorator_noargs
def set_prompt(prompt, **kwargs):
    from src.party import lang
    if kwargs is None:
        kwargs = renderer.script.__dict__

    global prompt_last
    if rv.nprompt is None or prompt != prompt_last:
        prompt_last = prompt
        maths.set_seed(rv.session.name, with_torch=False)
        dic = {**lang.__dict__, **kwargs}
        rv.nprompt = pnodes.bake_prompt(prompt, confmap, dic)


# rv.nprompt.update_state()


class DynamicFlow:
    def __init__(self):
        self.current_flow = None
        self.current_flow_index = 0
        self.current_flow_pos = 0
        self.current_flow_len = 0


dynflow = DynamicFlow()


def do_dynamic_flow():
    if not renderer.enable_dev: pass

    if not rv.flows:
        pass

    for flowpath in rv.flows:
        dirpath = session.extract_frames(flowpath)
        if dirpath:
            tricks.precompute_flows(flowpath)

    change = dynflow.current_flow is None or \
             dynflow.current_flow_pos >= dynflow.current_flow_len or \
             rv.flow_changes > 0.5

    if change:
        indices = np.linspace(0, len(rv.flows) - 1, len(rv.flows))
        dynflow.current_flow_pos = 0
        dynflow.current_flow_index = int(choose(indices.tolist(), exclude=dynflow.current_flow_index))
        dynflow.current_flow = rv.flows[dynflow.current_flow_index]
        dynflow.current_flow_len = session.res_frame_count(dynflow.current_flow)

    if dynflow.current_flow is not None and dynflow.current_flow_pos >= 1:
        idx = int(dynflow.current_flow_pos)
        flow = tricks.res_flow_cv2(dynflow.current_flow, idx, loop=True)
        rv.img = tricks.flow_warp(rv.img, flow, strength=rv.flow_strength)

    dynflow.current_flow_pos += rv.flow_speed


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
    rv.rz += nprng(-amp, amp)


def dtcam(xy=30, z=40, flower=2):
    rv.dtcam = rv.zeros()
    rv.dtcam = np.maximum(rv.dtcam, abs(rv.x) / xy)
    rv.dtcam = np.maximum(rv.dtcam, abs(rv.y) / xy)
    rv.dtcam = np.maximum(rv.dtcam, abs(rv.z) / z)
    rv.dtcam = np.maximum(rv.dtcam, abs(rv.flower) / flower)
    rv.dtcam = np.clip(rv.dtcam, 0, 1)


@trace_decorator
def colors():
    hsv = cv2.cvtColor(rv.img, cv2.COLOR_RGB2HSV)
    printdic = {}

    if rv.has_signal('hue'):
        hsv = tricks.hue(hsv, rv.hue)
        printdic['hue'] = rv.hue
    if rv.has_signal('saturation'):
        hsv = tricks.saturation(hsv, rv.saturation)
        printdic['saturation'] = rv.saturation
    if rv.has_signal('brightness'):
        hsv = tricks.brightness(hsv, rv.brightness)
        printdic['brightness'] = rv.brightness

    rv.img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # if rv.f % 5 == 0:
    #     rv.img = tricks.do_histogram_norm(rv.img)
    # rv.img = tricks.do_histogram_stretch(rv.img)

    hud.hud(**printdic)
    if rv.has_signal('contrast') and rv.contrast > eps:
        tricks.contrast(rv.img, rv.contrast)
        printdic['contrast'] = rv.contrast


last_depth = None


@trace_decorator
def rv_depth():
    global last_depth
    log("rv_depth")
    if rv.f % 1 == 0 or last_depth is None:  #
        depth = tricks.get_depth(rv.img)
        last_depth = depth
    else:
        depth = last_depth

    rv.depth = depth
    return depth


def set_prev_spike(signal, distance=1):
    rv.cur_spike_index = rv.f
    rv.prev_spike_index = rv.f
    if rv.f <= 0:
        return

    passed = 0
    for i in range(rv.f - 1, 0, -1):
        if signal[i] > 0.5:
            passed += 1
            if passed >= distance:
                rv.prev_spike_index = i
                rv.spike_distance = rv.cur_spike_index - rv.prev_spike_index
                break


def is_flowr_switch():
    return tricks.schedule_01(rv.flower_switch) == 1


def react_flowr(signals, lo=1, hi=3.5):
    a, b, c = 0, 0, 0
    while not lo < a + b + c < hi:
        # a = rng(0, 3)
        # b = rng(0, 1.5)
        # c = rng(5, 10)
        a = rng(lo / 3, hi * 3)
        b = rng(lo / 3, hi / 3)
        c = rng(lo / 3, hi / 3)

    S = choices(signals, 2)

    rv.flowr = a * jc(S[0]) + b * S[1] + c * jc(rv.vr_dt)


def react_zoom(signals, lo=10, hi=25, freq=(.03, 0.15)):
    S = choices(signals, n=4)

    freq = rng(*freq)
    a, b, c = get_rand_sum(3, lo, hi)

    rv.z += a * S[0] * perlin01(freq) + S[1] * b * S[2] + c * blur(S[3], 1)
    rv.z *= 0.5

    dtcam()


def react_xy(signals, lo=5, hi=10):
    S = choices(signals, n=2)
    x, y = get_rand_sum(2, lo, hi)

    rv.x += perlin01(0.08, -x, x) * S[0]
    rv.y += perlin01(0.1, -y, y) * jc(S[1])


# rv.y += 10 * pdiff(rv.amp1)


def react_sacade(signals, lo=0.5, hi=1.5):
    S = choices(signals)
    a, b, c = get_rand_sum(3, lo, hi)

    sacade(a + rv.dtcam + b * S[0] + c * rv.vr_dt)


def react_smooth_rot(signals, lo=0, hi=30):
    S = choices(signals)
    a = rng(lo, hi)

    rv.rz = a * perlin11(.06) * S[0]


@trace_decorator
def react_rxy(signals, lo=0, hi=1):
    S = choices(signals, n=2)
    a, b = get_rand_sum(2, lo, hi)

    rv.rx = a * perlin11() * S[0]
    rv.ry = b * perlin11() * 1 - S[1]


def react_hue_pca(signals, lo=0, hi=0.25):
    S = choices(signals)
    a = rng(lo, hi)
    rv.hue = a * jc(S[0], rng(0.33, 0.66))


def react_chg(lo=0.55, hi=0.685):
    a = rng(lo, hi)
    b = rng(lo, hi)
    c = rng(lo, hi)
    S = choices([rv.amp1, rv.amp2, rv.amp3, rv.amp4], n=3)

    # rv.motion, rv.motion_x, rv.motion_y = tricks.get_init_motion_stats()

    rv.ccg1 = a - 0.31 * blur(rv.spk2, 1)
    # rv.ccg1 += lerp(0, 0.155, jc(rv.amp1, 0.1))
    # rv.ccg1 += lerp(0.5, 0, rcurve(rv.amp1))
    rv.ccg1 -= 0.235 * rv.spk2
    rv.ccg1 -= 0.500 * jcurve(rv.onset, 0.4)
    rv.ccg1 *= 1 - norm(jcurve(rv.drum_onset, 0.75, 24 * 8))  # strong spikes

    rv.ccg2 = b - 0.05 * S[0]
    rv.ccg2 -= schedule([0, 0, 0, 0, 0, 0.1])
    rv.ccg2 -= 0.25 * jc(rv.spk2)
    rv.ccg2 -= 0.15 * rv.spk2
    rv.ccg2 -= 0.55 * jc(rv.onset, 0.85)
    rv.ccg2 -= lerp(0.0, 0.2, S[1])
    rv.ccg2 -= 0.100 * rv.drum_onset

    rv.ccg3 = c - 0.2 * S[2]  # cos01(0.75)
    rv.ccg3 -= 0.5 * rv.onset + 0.2 * sin01(0.5)

    rv.iccg1 += lerp(1, 0.5, rv.db)
    rv.iccg1 += 0.05 * perlin11(0.5)
    rv.iccg1 *= rc(S[2], 0.175)
    # rv.iccg1 = max(0.1, rv.iccg1)

    rv.iccg2 = lerp(0.35, 0.075, rv.db)
    rv.iccg3 = lerp(0.40, 0.0, rv.db)
    # rv.iccg3 += 0.1
    # rv.iccg3 *= rc(S[0])

    rv.cfg = 9.5 * jc(rv.spk) \
             + lerp(2.5, 4.5, norm(rv.vocals, 2 * 24)) \
             + 3 * (1 - rv.pcadiff) \
             + 3 * (1 - blur(rcurve(rv.vocals, 0.99), 10))
    rv.cfg = clamp(rv.cfg, 7, 999)

    rv.chg = 0.085
    rv.chg += 0.2 * lerp(rv.spk, rv.spk2, perlin01(0.075))
    rv.chg += 0.2 * rv.pcadiff
    rv.chg += rv.chg * 0.5
    rv.chg += rv.chapter_changes * rv.onset
    rv.chg = clamp(rv.chg, 0.02, 1)

    rv.render_switch = 0.85  # max(lerp(0.3, 0.4, rc(rv.dtcam)), lerp(0.3, 0.4, rc(rv.flowr / 2)))


def react_base():
    rv.steps = 10
    rv.db = norm(blur(rv.music, 200), 10 * 24)
    rv.pca1 = norm(rv.pca1)
    rv.othern = jcurve(norm(rv.other, 3 * 24), 0.7)
    rv.chapters = get_peaks(blur(rv.pca1, 5), 0.5) + get_peaks(blur(rv.vocals_onset, 5), 0.25)  # get_peaks(blur(rv.pca1, 5), 0.5)
    rv.amp1 = blur(rv.pca1, 2)  # jcurve(norm(blur(rv.pca1 + 1, 3) + blur(rv.pca2 + 3, 3) + blur(rv.pca3 + 5, 3)), 0.45)
    rv.amp2 = blur(rv.pca2, 2)  # jcurve(norm(blur(rv.pca1 + 1, 3) + blur(rv.pca2 + 3, 3) + blur(rv.pca3 + 5, 3)), 0.45)
    rv.amp3 = blur(rv.pca3, 2)  # jcurve(norm(blur(rv.pca1 + 1, 3) + blur(rv.pca2 + 3, 3) + blur(rv.pca3 + 5, 3)), 0.45)
    rv.amp4 = blur(rv.pca1, 25)
    rv.spk2 = jcurve(norm(rv.onset, 24 * 8), .65)
    rv.spk2 = blur(rv.spk2, 1)
    rv.spk = norm(rv.drum_onset)
    rv.curved_sharpness = lerp(0.35, 0.75, blur(rv.amp1, 45))  # used to increase the sharpness of the curve as the music gets louder (more dramatic!)


@trace_decorator
def react_composite():
    # Create the final composite between multiple sets using the binary chapter signal

    rv.select_gsignal('base')
    indices = np.where(rv.chapters > 0.5)[0]
    if indices[0] != 0:
        indices = np.insert(indices, 0, 0)

    for i, i2 in zip(indices, indices[1:]):
        rv.select_gsignal('chapter')
        rv.copy_gsignal('base')

        signals = [rv.amp1, rv.amp2, rv.amp3, rv.amp4, rv.spk, rv.spk2, rv.pca1, rv.drum_onset, rv.onset]  # rv.music, rv.vocals, rv.drums,
        react_chg()
        react_rxy(signals)
        react_xy(signals)
        react_sacade(signals)
        react_smooth_rot([rv.amp1, rv.amp2, rv.amp3, rv.amp4, rv.pca1, rv.pca2, rv.pca3, rv.drum_onset, rv.onset], 0, 20)
        react_hue_pca(signals)
        react_flowr([rv.vocals], 4, 6)

        rv.flowr *= 1 - sin01(0.5) * 0.75

        if rng() < 0.25:
            react_zoom([rv.amp1, rv.amp2, rv.amp3, rv.pca1, rv.pca2, rv.pca3], 25, 75)
        else:
            react_zoom([rv.amp1, rv.amp2, rv.amp3, rv.pca1, rv.pca2], 75, 150)

        rv.copy_gframes('chapter', 'composite', i, i2)

    rv.select_gsignal('composite')


def apply():
    flower = plugins.get('flower')
    colors()
    tricks.mat2d(rv.img, x=rv.x, y=rv.y, z=rv.z, rx=rv.rx, ry=rv.ry, rz=rv.rz)
    rv.img = flower.flow(rv.img, rv.flower)


@trace_decorator
def render_switch(
        fn_guidance: Optional[Callable] = None,
        fn_detail: Optional[Callable] = None,
        after_cn: Optional[Callable] = None,
        txt2img: Optional[Callable] = None,
        img2img: Optional[Callable] = None,
        preserve_palette: bool = False,
        shade_depth: bool = True
) -> None:
    """
    A controllable render process which alternates between txt2img and img2img without controlnet.
    This allows the imagery to organically move between two different composition spaces, latent and physical.
    img2img can be more subtle as well, which tempers the animation and makes it less jittery.
    """
    global last_seed, seed_frames_elapsed

    if rv['skip_diffusion'] > 0.5:
        return

    # Validate and set defaults
    if txt2img is None or img2img is None:
        diffusers = plugins.get('sd_diffusers')
        if txt2img is None:
            logerr("DEPRECATED use of render_switch, please set txt2img argument to specify how to render")
            txt2img = diffusers.txt2img_cn
        if img2img is None:
            logerr("DEPRECATED use of render_switch, please set img2img argument to specify how to render")
            img2img = diffusers.img2img_cn

    # Prepare arguments
    args = {
        'rv': rv,
        'prompt': rv.prompt, 'promptneg': rv.promptneg,
        'seed': rv.nextseed,
        'steps': rv.steps,
        'w': rv.w, 'h': rv.h,
        'chg': rv.chg,
        'cfg': rv.cfg,
        'sampler': rv.sampler,
        'img': rv.img,
        'image': rv.img,
        'seed_grain': rv['seed_grain'],
        # 'seed_grain': rv.cn_chg,
        'guidance': None,
        'ccg': None,
    }

    fn_guidance = fn_guidance or get_guidance

    is_cn_switch = tricks.schedule_01(rv['cn_switch'], hudname='cn_render')
    is_cn_img2img = tricks.schedule_01(rv['i2i_switch'], hudname='i2i')

    if rv.f <= 1:
        rv.chg = 1
        rv.force_cn = 1

    if is_cn_switch or rv['force_cn'] > 0.5:
        # Process CN switch
        if fn_detail is not None:
            rv.img = fn_detail()

        guidance = fn_guidance()
        args['guidance'] = guidance
        args['ccg'] = get_ccgs(len(guidance))[0]
        rv.img = txt2img(**args)

        if after_cn is not None:
            after_cn()

        # Post-process
        if preserve_palette > 0:
            cc = tricks.palette(rv, rv.img, rv.cn_img2)
            rv.img = tricks.alpha_blend(rv.img, cc, lerp(0.845, 0.5, rv.dtcam))

        if shade_depth:
            tricks.shade_depth()
    else:
        # Process img2img
        rv.seed = rv.nextseed
        rv.img = img2img(**args)


# @trace_decorator
# def render_switch(fn_guidance=None,
#                   fn_detail=None,
#                   after_cn=None,
#                   txt2img=None,
#                   img2img=None,
#                   preserve_palette=False,
#
#                   shade_depth=True):
#     """
#     A controllable render process which alternates between txt2img and img2img without controlnet.
#     This allows the imagery to organically move between two different composition space, latent and physical.
#     img2img can be more subtle as well, which tempers the animation and makes it less jittery.
#     """
#     global last_seed, seed_frames_elapsed
#
#     if rv.skip_diffusion:
#         return
#
#
#     if txt2img is None:
#         logerr("DEPRECATED use of render_switch, please set txt2img argument to specify how to render")
#         diffusers = plugins.get('sd_diffusers')
#         rv.img = diffusers.txt2img_cn
#     if img2img is None:
#         logerr("DEPRECATED use of render_switch, please set img2img argument to specify how to render")
#         diffusers = plugins.get('sd_diffusers')
#         rv.img = diffusers.img2img_cn
#
#     args = dict(
#         rv=rv,
#         prompt=rv.prompt, promptneg=rv.promptneg,
#         seed=rv.seed,
#         steps=rv.steps,
#         w=rv.w, h=rv.h,
#         chg=rv.chg,
#         cfg=rv.cfg,
#         sampler=rv.sampler,
#         image=rv.img,
#         seed_grain=rv.seed_grain,
#         guidance=None,
#         ccg=None,
#     )
#
#     fn_guidance = fn_guidance or get_guidance
#
#     is_cn_switch = tricks.schedule_01(rv.cn_switch, hudname='cn_render')
#     is_cn_img2img = tricks.schedule_01(rv.i2i_switch, hudname='i2i')
#     if is_cn_switch or rv.force_cn > 0.5:
#         if fn_detail is not None:
#             rv.img = fn_detail()
#
#
#         args['guidance'] = fn_guidance()  # OBSOLETE
#         args['ccg'], = get_ccgs()  # OBSOLETE
#         args['seed_grain'] = rv.cn_chg  # OBSOLETE
#         rv.img = txt2img(**args)
#
#         if after_cn is not None:
#             after_cn()
#
#         # POST-PROCESS
#         if preserve_palette > 0:
#             cc = tricks.palette(rv, rv.img, rv.cn_img2)
#             rv.img = tricks.alpha_blend(rv.img, cc, lerp(0.845, 0.5, rv.dtcam))
#
#         if shade_depth:
#             tricks.shade_depth()
#     else:
#         rv.seed = rv.nextseed
#         rv.img = img2img(**args)


def get_guidance(*types: str) -> List[np.ndarray]:
    if not types:
        types = ['hed', 'temporal', 'depth']

    def apply_image_transformations(img: np.ndarray) -> np.ndarray:
        if rv.has_signal('init_contrast'):
            img = tricks.contrast(img, rv.init_contrast)
        if rv.has_signal('init_blur'):
            img = tricks.get_blurred(img, rv.init_blur)
        if any(rv.has_signal(f'init_{axis}') for axis in 'xyz'):
            img = tricks.mat2d(img, x=rv.init_x, y=rv.init_y, z=rv.init_z, r=rv.init_r)
        return img

    def get_init_anchor() -> Tuple[float, float]:
        init_anchor = (.5, .5)
        if rv.has_signal('init_anchor_x'):
            init_anchor = (rv.init_anchor_x, init_anchor[1])
        if rv.has_signal('init_anchor_y'):
            init_anchor = (init_anchor[0], rv.init_anchor_y)
        return init_anchor

    def apply_depth_mode(img: np.ndarray, type_: str, depth_map: Optional[np.ndarray], depth_result: Optional[np.ndarray]) -> np.ndarray:
        if type_.endswith(('*depth', '*depth_result')):
            depth_to_use = depth_result if type_.endswith('*depth_result') else depth_map
            img = np.clip(img * rc(depth_to_use / 255, 0.75), 0, 255).astype(np.uint8)
        return img

    def process_canny(cn_img: np.ndarray) -> np.ndarray:
        canny_src = tricks.alpha_blend(rv.img, cn_img, rv['iccg1'])
        return tricks.img_to_canny(canny_src, rv.canny_lo, rv.canny_hi)

    def process_hed(cn_img: np.ndarray) -> np.ndarray:
        hed_src = tricks.alpha_blend(rv.img, cn_img, rv['iccg1'])
        return tricks.img_to_misto(hed_src)

    def process_temporal(cn_img: np.ndarray) -> np.ndarray:
        temporal_src = rv.img
        if cn_img is not None:
            temporal_src = tricks.alpha_blend(temporal_src, cn_img, rv['iccg2'])
        return temporal_src

    def process_depth(cn_img: np.ndarray) -> np.ndarray:
        if rv.init_img is None:
            cn_img_depth = rv.depth
        else:
            init_depth = tricks.get_depth(rv.init_img)
            cn_img_depth = tricks.alpha_blend(rv.depth, init_depth, rv['iccg3'])

        if rv.has_signal('ground'):
            perp = tricks.get_yperp_mask() + cv2.flip(tricks.get_yperp_mask(), 0)
            guidance_img3_ground = tricks.alpha_blend(cn_img_depth, perp, rv.ground)
            ground_clip_plane = 0.5 * 255
            cn_img_depth[cn_img_depth < ground_clip_plane] = guidance_img3_ground[cn_img_depth < ground_clip_plane]

        return cn_img_depth

    cn_img = next((img for img in (rv['cn_img'], rv['init_img'], rv['img']) if img is not None), None)

    if cn_img is not None:
        cn_img = apply_image_transformations(cn_img)

    hud.snap('cn_img', cn_img)

    init_anchor = get_init_anchor()

    priorities = {
        'canny': 0, 'hed': 1, 'hed*depth': 2, 'temporal': 3, 'depth': 4,
    }
    types = sorted(types, key=lambda t: priorities[t])

    depth_map, depth_result = None, None

    processors = {
        'canny': process_canny,
        'hed': process_hed,
        'temporal': process_temporal,
        'depth': process_depth
    }

    ret = []
    for type_ in types:
        base_type = type_.split('*')[0]
        img = processors[base_type](cn_img)
        img = apply_depth_mode(img, type_, depth_map, depth_result)
        ret.append(img)

    for i, img in enumerate(ret, 1):
        hud.snap(f'guidance_{i}', img)

    return ret


@trace_decorator_noargs
def get_ccgs(n, apply_seed_change_attenuation=False, outputs='ccg', print_hud=False):
    if isinstance(outputs, str):
        outputs = [outputs]
    output_lists = [[] for _ in range(len(outputs))]

    global last_seed, seed_frames_elapsed
    if last_seed != rv.seed:
        last_seed = rv.seed
        seed_frames_elapsed = 0
    else:
        seed_frames_elapsed += rv.dt

    for i in range(n):
        i += 1
        key_ccg = f'ccg{i}'
        key_ccg_a = f'ccg{i}_a'
        key_ccg_b = f'ccg{i}_b'
        key_img = f'ccg{i}_img'
        key_ccg_atten = f'seed_atten_ccg{i}'
        key_iccg = f'iccg{i}'

        # Valid values for outputs
        img = rv[key_img]
        ccg = rv[key_ccg]
        ccga = rv[key_ccg_a]
        ccgb = rv[key_ccg_b]
        ccg_atten = rv[key_ccg_atten]
        iccg = rv[key_iccg]

        if apply_seed_change_attenuation and rv.seed_atten_time > 0:
            ccg += lerp(ccg_atten, 0, seed_frames_elapsed / rv.seed_atten_time * rv.fps)

        if print_hud:
            hud.snap(f'ccg{i}_img', img)
            hud.hud(**{
                key_ccg: ccg,
                key_ccg_a: ccga,
                key_ccg_b: ccgb,
                key_iccg: iccg,
                key_ccg_atten: ccg_atten,
            })

        for j, output in enumerate(outputs):
            output_lists[j].append(locals()[output])

    return output_lists


@trace_decorator_noargs
def txt2img_cn_dev(self, *args, **kwargs):
    hud(chg=kwargs.get('chg'), cfg=kwargs.get('cfg'), ccg=kwargs.get('ccg'), seed=kwargs.get('seed'))
    if 'image' in kwargs:
        img = tricks.grain(rv, strength=kwargs['chg'])
        img = tricks.saltpepper(rv, coverage=kwargs['chg'] * 0.04)
        return img
    else:
        # random color
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = tricks.saltpepper(rv, coverage=kwargs['chg'] * 0.04)
        return img


@trace_decorator_noargs
def txt2img_dev(self, *args, **kwargs):
    hud(chg=kwargs.get('chg'), cfg=kwargs.get('cfg'), ccg=kwargs.get('ccg'), seed=kwargs.get('seed'))
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img = tricks.saltpepper(rv, coverage=kwargs['chg'] * 0.04)
    return img


@trace_decorator_noargs
def img2img_dev(self, *args, **kwargs):
    hud(chg=kwargs.get('chg'), cfg=kwargs.get('cfg'), ccg=kwargs.get('ccg'), seed=kwargs.get('seed'))
    img = tricks.grain(rv, strength=kwargs['chg'])
    img = tricks.saltpepper(rv, coverage=kwargs['chg'] * 0.04)
    return img


def get_cn_guidance(edge_type=None):
    midas = plugins.get('midas')

    guidance_img1 = rv.img
    guidance_img2 = rv.img
    guidance_img3 = rv.depth  # tricks.get_depth(rv.img)

    # convert.save_img(depth_better, session.res('guidance_3_better.jpg'))
    # guidance_img3 = tricks.get_depth(rv.img)

    # This will make the colors bleed all over the place
    # while ccg1 preserves the composition
    # cool as fuck
    # guidance_img2 = midas.mat3d(
    #         z=25 * jcurve(rv.amp, rv.curved_sharpness) + 30 * jcurve(rv.spk2, 0.5),
    #         far=20000,
    #         fov=82.5,
    #         w_midas=2.2,
    #         sampling_mode='nearest')

    # Either init image or cn_img
    init_anchor = (.5, .5)
    if rv.has_signal('init_anchor_x'): init_anchor = (rv.init_anchor_x, init_anchor[1])
    if rv.has_signal('init_anchor_y'): init_anchor = (init_anchor[0], rv.init_anchor_y)
    cn_img = first_non_null(rv.cn_img, rv.init_img, rv.img)

    # cn_img = tricks.load_cv2(cn_img)
    if rv.has_signal('init_contrast'):
        # cn_img = tricks.do_histogram_norm(cn_img)
        # cn_img = tricks.do_histogram_stretch(cn_img)
        # cn_img = tricks.selective_contrast_enhancement(cn_img, 0.3, 0.75) #dbg.v1, dbg.v2)
        cn_img = tricks.contrast(cn_img, rv.init_contrast)

    if rv.has_signal('init_blur'):
        cn_img = tricks.get_blurred(cn_img, rv.init_blur)

    # Horizontal duplication
    # d = 20
    # cn_imgl = cn_img
    # cn_imgl = tricks.mat2d(cn_imgl, x=-d, y=0)
    # cn_imgr = cn_img
    # cn_imgr = tricks.mat2d(cn_imgr, x=d, y=0)
    # cn_img = tricks.alpha_blend(cn_img, cn_imgl, 0.5)
    # cn_img = tricks.alpha_blend(cn_img, cn_imgr, 0.5)

    cn_img = tricks.mat2d(cn_img, x=rv.init_x, y=rv.init_y, z=rv.init_z)

    # cn_img = tricks.do_canny_edge_reinforce(cn_img, 0.025, 0.01)

    # Inject the cn_img
    if cn_img is not None:
        # cn_img_flow = tricks.res_flow_cv2('init')
        # flow_mask = tricks.get_flow_mask()

        # depth_img = rv.depth
        # if not isinstance(depth_img, np.ndarray):
        #     depth_img = tricks.get_depth(rv.init_img)

        guidance_img1 = tricks.alpha_blend(guidance_img1, cn_img, rv['iccg1'])
        guidance_img2 = tricks.alpha_blend(guidance_img2, cn_img, rv['iccg2'])
        guidance_img3 = tricks.alpha_blend(guidance_img3, tricks.get_depth(cn_img), rv['iccg3'])
        # depth_better = tricks.get_depth(cn_img)

        if renderer.enable_dev:
            rv.img = cn_img

    # Inject perspective depth
    perp = tricks.get_yperp_mask() + cv2.flip(tricks.get_yperp_mask(), 0)
    guidance_img3_ground = tricks.alpha_blend(guidance_img3, perp, rv.ground)
    ground_clip_plane = 0.5 * 255
    guidance_img3[guidance_img3 < ground_clip_plane] = guidance_img3_ground[guidance_img3 < ground_clip_plane]

    # TODO autodetect
    if edge_type == 'hed':
        guidance_img1 = tricks.img_to_hed(guidance_img1)
        # Use CV2 to sharpen guidance_img2 with convolutions
        guidance_img1 = cv2.detailEnhance(guidance_img1, sigma_s=10, sigma_r=0.15)
    # Multiply with guidance_img3 which is depth
    # guidance_img1 = guidance_img1 * rc(guidance_img3/255, 0.75)
    # guidance_img1 = np.clip(guidance_img1, 0, 255)
    # guidance_img1 = guidance_img1.astype(np.uint8)
    elif edge_type == 'canny':
        guidance_img1 = tricks.img_to_canny(guidance_img1, rv.canny_lo, rv.canny_hi)

    # Multiply with guidance_img3 which is depth
    # guidance_img2 = guidance_img2 * rc(guidance_img3/255, 0.75)
    # guidance_img2 = np.clip(guidance_img2, 0, 255)
    # guidance_img2 = guidance_img2.astype(np.uint8)

    # img_bw = cv2.cvtColor(guidance_img1, cv2.COLOR_BGR2GRAY)
    # img_bw = np.expand_dims(img_bw, axis=2)
    # img_bw = np.concatenate((img_bw, img_bw, img_bw), axis=2)
    # guidance_img1 = tricks.alpha_blend(img_bw, guidance_img1, 0.5)

    return guidance_img1, guidance_img2, guidance_img3
