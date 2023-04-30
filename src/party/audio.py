from pathlib import Path


from src.lib.printlib import trace_decorator
from src.party.maths import *

# TODO melodic speed
# TODO harmonic chg speed


@trace_decorator
def load_crepe_keyframes(filename):
    import pandas as pd
    df = pd.read_csv(filename)
    freq = to_keyframes(df['frequency'], len(df['frequency']) / df['time'].values[-1])
    confidence = to_keyframes(df['confidence'], len(df['frequency']) / df['time'].values[-1])
    return freq, confidence

@trace_decorator
def load_harmonics(filename):
    import librosa
    from src.party.keyfinder import Tonal_Fragment

    # This audio takes a long time to load because it has a very high sampling rate; be patient.
    # the load function generates a tuple consisting of an audio object y and its sampling rate sr
    y, sr = librosa.load(filename)

    # This function filters out the harmonic part of the sound file from the percussive part, allowing for
    # more accurate harmonic analysis
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    unebarque_fsharp_min = Tonal_Fragment(y_harmonic, sr, tend=22)
    unebarque_fsharp_min.print_chroma()
    # ipd.Audio(filename)

    unebarque_fsharp_min.print_key()
    unebarque_fsharp_min.corr_table()
    unebarque_fsharp_min.chromagram("Une Barque sur l\'Ocean")


@trace_decorator
def load_dbnorm(filename, window=None, caching=True):
    window = window if window is None else window * rv.fps
    return load_db_keyframes(filename, caching)


@trace_decorator
def load_db_keyframes(filename, caching=True):
    dbs = load_db(filename, caching)
    return dbs
    # return to_keyframes(dbs, audio.frame_rate)


@trace_decorator
def load_db(filename, caching=True):
    from src.party.audio_processing.modules.loudness import lufs_meter

    cachepath = Path(filename).with_suffix(".npy")

    if caching and cachepath.exists():
        print(f"Restoring cached decibels: {cachepath}")
        loudness = np.load(cachepath.as_posix())
    else:
        loudness = load_loudness(filename)
        if caching:
            np.save(cachepath.as_posix(), loudness)

    import resampy
    loudness = resampy.resample(loudness, 60, rv.fps)


    return loudness
    # return audio, decibels
def load_loudness(filename):
    import soundfile as sf
    from src.party.audio_processing.modules.loudness import lufs_meter
    print(f"Running loudness detection on {filename}...")

    y, sr = sf.read(filename)  # load audio (with shape (samples, channels))
    # y = filter.butter(y, sr, 'highpass', 1, 400)
    meter = lufs_meter(sr, 1 / 60, overlap=0)
    loudness = meter.get_mlufs(y)
    # loudness = meter.integrated_loudness(y)  # measure loudness
    loudness[np.isinf(loudness)] = 0  # Replace infinities and nans with zero
    loudness = norm(loudness)  # Normalize to 0-1 in a 12 second window
    return loudness


@trace_decorator
def to_keyframes(dbs, original_sps):
    start = 0
    total_seconds = len(dbs) / original_sps
    # print(len(dbs), original_sps, total_seconds)
    # start=0
    # total_seconds=5

    frames = int(rv.fps * total_seconds)

    dt = np.zeros(frames)
    for i in range(frames):
        # frame --> seconds
        t = (i) / rv.fps + start
        t1 = (i + 1) / rv.fps + start
        # print(t, t1)

        d = dbs[int(t * original_sps):int((t1) * original_sps)]
        dt[i] = np.mean(d)

        # remove infinities and nans
        if np.isinf(dt[i]) or np.isnan(dt[i]):
            dt[i] = dt[i - 1]

    return dt
    # return smooth_1euro(dt)


# @trace_decorator
# def convert_to_decibel(arr):
#     ref = 1
#     if arr != 0:
#         return 20 * np.log10(np.abs(arr) / ref)
#     else:
#         return -60

@trace_decorator
def convert_to_decibel(arr):
    ref = 1
    decibel = np.where(arr != 0, 20 * np.log10(np.abs(arr) / ref), -60)
    return decibel

@trace_decorator
def play_wav(audioseg, t):
    import simpleaudio
    if t is not None:
        audioseg = audioseg.get_sample_slice(int(t * audioseg.frame_rate))

    return simpleaudio.play_buffer(
            audioseg.raw_data,
            num_channels=audioseg.channels,
            bytes_per_sample=audioseg.sample_width,
            sample_rate=audioseg.frame_rate
    )
