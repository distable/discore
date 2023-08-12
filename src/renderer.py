"""
A renderer with a common interface to communicate with.
The renderer has all the logic you would find in a simple game engine,
so it keeps track of time and targets for a specific FPS.

A 'render script' must be loaded which is a python file which implements a render logic.
The file is watched for changes and automatically reloaded when it is modified.
A valid render script has the following functions:

- def on_callback(rv, name)  (required)

Various renderer events can be hooked with this on_callback by checking the name,
and will always come in this order:

- 'load' is called when the script is loaded for the very first time.
- 'setup' is called whenever the script is loaded or reloaded.

The render script is universal and can be used for other purpose outside of the renderer
by simply calling on_callback with a different name.

Devmode:
    - We will not check for script changes every frame.

CLI Mode:
    - Always rendering on
    - Starts from last frame

GUI Mode:
    - Initialized in STOP mode
    - Starts from first frame

"""
import datetime
import importlib
import math
import os
import signal
import sys
import threading
import time
import traceback
from enum import Enum
from pathlib import Path

import numpy as np
from yachalk import chalk

import jargs
import userconf
from jargs import args, get_discore_session
from src.classes import paths
from src.classes.convert import load_cv2
from src.classes.paths import get_script_file_path, parse_action_script
from src.classes.Session import Session
from src.gui.AudioPlayback import AudioPlayback
from src.lib.corelib import invoke_safe
from src.lib.printlib import trace, trace_decorator
from src.rendering import hud
from src.rendering.rendervars import RenderVars

allowed_module_reload_names = [
    'scripts.',
    'src.party.',
    # 'src.classes.',
]

script_change_detection_paths = [
    paths.scripts,
    paths.code_core,
]

enable_dev = args.dev or args.readonly  # Are we in dev mode?
enable_gui = False  # Are we in GUI mode? (hobo, ryusig, audio)
enable_readonly = args.readonly  # Cannot render, used as an image viewer
enable_save = True  # Enable saving the frames
enable_save_hud = False  # Enable saving the HUD frames
enable_unsafe = args.unsafe  # Should we let calls to the script explode so we can easily debug?
detect_script_every = 1
auto_populate_hud = False  # Dry runs to get the HUD data when it is missing, can be laggy if the script is not optimized

class RenderMode(Enum):
    PAUSE = 0
    PLAY = 1
    RENDER = 2

class RenderingRepetition(Enum):
    ONCE = 0
    FOREVER = 1


class InitState(Enum):
    UNINITIALIZED = 0
    LIGHT = 1
    HEAVY = 2

class RendererState(Enum):
    INITIALIZATION = 0
    READY = 1
    RENDERING = 2
    STOPPED = 3

class SeekImageMode(Enum):
    NORMAL = 0
    NO_IMAGE = 1
    IMAGE_ONLY = 2

class SeekRequest:
    def __init__(self, f_target, img_mode=SeekImageMode.NORMAL):
        self.f_target = f_target
        self.img_mode = img_mode

class PauseRequest(Enum):
    NONE = 0
    PAUSE = 1
    PLAY = 2
    TOGGLE = 3

class RendererRequests:
    """
    The renderer runs in a loop and we wish to do everything within that loop
    so we can handle any edge cases. The session must not change while rendering,
    for example.
    """

    def __init__(self):
        self.seek: SeekRequest | None = None  # Frames to seek to
        self.script_check = False  # Check if the script has been modified
        self.render = None
        self.pause = PauseRequest.NONE  # Pause the playback/renderer
        self.stop = False  # Stop the whole renderer

class LoopState:
    def __init__(self):
        self.last_frame_prompt = ""
        self.was_paused = False
        self.start_f = 0  # Frame we started the frame on
        self.start_img = None  # Image we started the frame on
        self.n_rendered = 0  # Number of frames rendered
        self.time_started = 0  # Time we started rendering
        self.last_rendered_frame = None
        self.last_frame_dt = 0
        self.frame_callback = None
        self.last_script_check = 0

class PlaybackState:
    def __init__(self):
        self.end_frame = None  # The frame to stop at. None means never end
        self.looping = False  # Enable looping
        self.loop_frame = 0  # Frame to loop back to
        self.last_play_start_frame = 1

state = RendererState.INITIALIZATION
init_state = InitState.UNINITIALIZED
mode = RenderMode.PAUSE
render_repetition = RenderingRepetition.FOREVER
rv = RenderVars()
requests = RendererRequests()
play = PlaybackState()
session: Session | None = None  # Current session
loop = LoopState()
callbacks = []
enable_gui_dispatch = True  # Whether or not the GUI handles the dispatching of iter_loop

# State (internal)
script = None  # The script module
script_name = ''  # Name of the script file
script_error = False  # Whether initialization errored out
script_time_cache = {}
frame_error = False  # Whether the last frame errored out

# Signals
invalidated = True  # This is to be handled by a GUI, such a hobo (ftw)
on_frame_changed = []
on_t_changed = []
on_script_loaded = []
on_stop_playback = []
on_start_playback = []

# gui
audio = AudioPlayback()

def is_cli():
    return not enable_gui

def is_gui():
    return enable_gui

def is_gui_dispatch():
    return mode.value < RenderMode.RENDER.value and enable_gui_dispatch

# region Emits
@trace_decorator
def _emit(name):
    if script:
        script.rv = rv
        script.s = session
        script.ses = session
        script.f = rv.f
        script.dt = rv.dt
        script.fps = rv.fps

    if hasattr(script, name):
        func = getattr(script, name)
        func()
    if hasattr(script, 'callback'):
        script.callback(name)

def _invoke_safe(*args, **kwargs):
    return invoke_safe(*args, unsafe=enable_unsafe, **kwargs)


# endregion

# region Script

def detect_script_modified():
    # TODO this is slow to do every frame
    def check_dir(path):
        modified = False

        # print(f'check_dir({path})')

        # Check all .py recursively
        for file in Path(path).rglob('*.py'):
            # Compare last modified time of the file with the cached time
            # print("Check file:", file.relative_to(path).as_posix())
            key = file.relative_to(path).as_posix()
            is_new = key not in script_time_cache
            if is_new or script_time_cache[key] < file.stat().st_mtime:
                script_time_cache[key] = file.stat().st_mtime
                if not is_new:
                    modified = True
                # print(key, file.stat().st_mtime)

        return modified

    return any([check_dir(p) for p in [*script_change_detection_paths, session.dirpath]])

def reload_script_on_change():
    if detect_script_modified():
        print(chalk.dim(chalk.blue("Change detected in scripts, reloading")))
        reload_script()

def reload_script():
    global script
    global mode

    load_script()

    if not mode == RenderMode.RENDER and is_cli():
        mode = RenderMode.RENDER

def load_script(name=None):
    global script, script_name
    global script_error, frame_error

    callbacks.clear()
    script_error = False
    frame_error = False

    with trace('renderer.load_script'):
        # Determine the script path
        # ----------------------------------------
        fpath = ''

        if not name:
            _, name = parse_action_script(jargs.args.action)
            if name is not None:
                fpath = get_script_file_path(script_name)

        if not fpath:
            fpath = session.res_script(name, touch=True)

        script_name = name

        # Get the old globals
        oldglobals = None
        if script is not None:
            oldglobals = script.__dict__.copy()
            # Don't keep functions
            oldglobals = {k: v for k, v in oldglobals.items() if not callable(v)}

        # Reload all modules in the scripts folder
        mpath = paths.get_script_module_path(fpath)
        reload_modules(mpath)

        if os.path.exists(fpath):
            invoke_import(mpath)
            invoke_start()

        # Restore the globals
        # if script is not None and oldglobals is not None:
        #     script.__dict__.update(oldglobals)

    _invoke_safe(on_script_loaded)

def reload_modules(exclude):
    modules_to_reload = []
    for x in sys.modules:
        for m in allowed_module_reload_names:
            if x.startswith(m):
                modules_to_reload.append(x)

    for m in modules_to_reload:
        if m != exclude:
            importlib.reload(sys.modules[m])

def invoke_import(mpath):
    global script, script_error
    if script is None:
        rv.init()

        with trace(f'renderer.load_script.importlib.import_module({mpath})'):
            print(f'--> {session.name}.{script_name}.import')
            if enable_unsafe:
                script = importlib.import_module(mpath, package='imported_renderer_script')
            else:
                try:
                    script = importlib.import_module(mpath, package='imported_renderer_script')
                    invoke_init()
                except Exception as e:
                        chalk.red("<!> SCRIPT ERROR <!>")
                        traceback.print_exc()
                        script_error = True
    else:
        print(f'--> {session.name}.{script_name}.reload')
        if enable_unsafe:
            importlib.reload(script)
            invoke_init()
        else:
            try:
                importlib.reload(script)
            except Exception as e:
                chalk.red("<!> SCRIPT ERROR <!>")
                traceback.print_exc()
                script_error = True


def invoke_init():
    # Things that are not strictly necessary, but are useful for most scripts
    global init_state

    if init_state.value < InitState.LIGHT.value:
        print(f'--> {session.name}.{script_name}.init')
        _invoke_safe(_emit, 'init')
        init_state = InitState.LIGHT

    if init_state.value < InitState.HEAVY.value and not enable_dev:
        print(f'--> {session.name}.{script_name}.init_heavy')
        _invoke_safe(_emit, 'init_heavy')
        init_state = InitState.HEAVY

def script_bool(name):
    if script is not None:
        v = script.__dict__.get(name, False)
        if v is None:
            return False
        return True

def script_get(name, default=None):
    if script is not None:
        return script.__dict__.get(name, None)

def before_invoke():
    from src.party import std

    invoke_init()

    if not enable_dev and rv.n > 0:
        if 'promptneg' in script.__dict__:
            rv.promptneg = script.__dict__['promptneg']
        if 'prompt' in script.__dict__:
            std.refresh_prompt(**script.__dict__)

    if rv.nprompt is not None and not enable_dev:
        from src.party import pnodes
        rv.prompt = pnodes.eval_prompt(rv.nprompt, rv.t)

    if std.init_img_exists():
        init_anchor = (.5, .5)
        if rv.has_signal('init_anchor_x'): init_anchor = (rv.init_anchor_x, init_anchor[1])
        if rv.has_signal('init_anchor_y'): init_anchor = (init_anchor[0], rv.init_anchor_y)
        rv.init_img = session.res_frame_cv2('init', fps=rv.fps, anchor=init_anchor)
        hud.snap('init', rv.init_img)

    session.w = rv.w
    session.h = rv.h


def invoke_start():
    from src.party import std
    from src.party import maths

    rv.reset_signals()
    rv.load_signal_arrays()
    before_invoke()
    maths.reset()

    print(f'--> {session.name}.{script_name}.start')

    rv.is_array_mode = True
    std.init_media(demucs=script_bool('demucs'))
    _invoke_safe(_emit, 'start')
    rv.is_array_mode = False

def invoke_frame():
    global frame_error

    before_invoke()
    if rv.f == 1:
        if session.res('init'):
            rv.img = session.res_frame_cv2('init')
        else:
            rv.img = np.zeros((rv.h, rv.w, 3), dtype=np.uint8)
        rv.resize(True)

    frame_error = not _invoke_safe(_emit, 'frame', failsleep=0.25)

# endregion

# region Core functionality

def is_paused():
    return mode == RenderMode.PAUSE

def is_running():
    return state == RendererState.READY or state == RendererState.RENDERING

def should_render():
    ret = mode == RenderMode.RENDER or is_cli()
    ret = ret and (session.f != loop.last_rendered_frame or mode == RenderMode.PAUSE)  # Don't render the same frame twice (if we render faster than playback)
    ret = ret and not script_error and not frame_error  # User must fix the script, it's too likely to cause crashes
    return ret

def should_save():
    return enable_save and not enable_dev


def init(_session=None, scriptname='', gui=True):
    """
    Initialize the renderer
    This will load the script, initialize the core
    Args:
        _session:
        scriptname: Name of the script to load. If not specified, it will load the default session script.
        gui: Whether or not we are running in GUI mode. In CLI, we immediately resume the render and cannot do anything else.

    Returns:
    """

    global state
    global enable_dev, enable_gui
    global session
    global invalidated

    if gui:
        enable_dev = True

    set_session(_session or get_discore_session())

    rv.start_frame(1)
    rv.init()

    # Setup the session
    # ----------------------------------------
    set_session(session)

    # Load the script
    # ----------------------------------------
    with trace('Script loading'):
        _invoke_safe(load_script, scriptname)

    # GUI setup
    # ----------------------------------------
    enable_gui = gui
    # if gui and not userconf.gui_main_thread:
    #     # Run the GUI on 2nd thread
    #     threading.Thread(target=ui_thread_loop, daemon=True).start()

    invalidated = True
    state = RendererState.READY

    hud.enable_printing = is_cli()

    rv.trace = 'initialized'
    signal.signal(signal.SIGTERM, handle_sigterm)

    print("Renderer initialized.")

    return session


def ui_thread_loop():
    from src.rendering import hobo

    hobo.rv = rv
    hobo.audio = audio
    hobo.run()

    requests.stop = True


def handle_sigterm():
    print("renderer sigterm handler")

def loop_thread_loop(lo, hi, callback):
    run_loop(lo, hi, inner=True, callback=callback)


def run_loop(lo=None, hi=None, callback=None, inner=False):
    global state
    global enable_gui
    global invalidated
    global mode

    if args.dry:
        return

    if not inner and enable_gui:
        if userconf.gui_main_thread:
            t = threading.Thread(target=loop_thread_loop, args=(lo, hi, callback))
            t.start()
            ui_thread_loop()
            return
        else:
            threading.Thread(target=ui_thread_loop, daemon=True).start()

    loop.last_frame_time = 0
    loop.last_script_check = 0
    loop.dt_accumulated = 0
    loop.frame_callback = callback

    if lo is not None:
        session.seek(lo)
    elif enable_gui:
        session.seek_max()
    else:
        session.seek_new()

    if jargs.args.remote:
        enable_gui = False
    # Go immediately into rendering
    if not enable_gui:
        mode = RenderMode.RENDER
    if hi is None:
        hi = math.inf


    print('Starting render loop...')

    while session.f < hi and not requests.stop:
        if is_gui_dispatch() and enable_gui and enable_dev:
            time.sleep(0.1)
            continue
        iter_loop()

    session.save_data()
    state = RendererState.STOPPED

# @trace_decorator
def iter_loop():
    global mode, invalidated
    frame_changed = False

    # Script live reload detection
    with trace("renderiter.reload_script_check"):
        script_elapsed = time.time() - loop.last_script_check
        if requests.script_check \
                or script_elapsed > detect_script_every > 0 \
                or is_cli():
            requests.script_check = False
            reload_script_on_change()
            loop.last_script_check = time.time()

    # Flush render request
    just_started_render = False
    if requests.render:
        mode = RenderMode.RENDER
        just_started_render = True
        requests.render = False

    # Flush pause request
    if requests.pause != PauseRequest.NONE:
        if requests.pause == PauseRequest.TOGGLE:
            if mode == RenderMode.PLAY:
                mode = RenderMode.PAUSE
            elif mode == RenderMode.PAUSE:
                mode = RenderMode.PLAY
                play.last_play_start_frame = session.f
        elif requests.pause == PauseRequest.PAUSE:
            mode = RenderMode.PAUSE
            print("SET_PAUSE FROM REQEST")
        elif requests.pause == PauseRequest.PLAY:
            mode = RenderMode.PLAY

        requests.pause = None

    # elif its_pizza_time and requests.request_render == False:
    #     set_render('toggle')
    # elif session.f >= session.f_last + 1:
    #     set_render('toggle')
    # else:
    #     requests.pause = not paused
    # Flush seek requests
    sreq = requests.seek
    requests.seek = None
    if sreq:
        f_prev = session.f
        img_prev = session.img

        session.f = sreq.f_target
        if sreq.img_mode == SeekImageMode.NORMAL:
            session.load_f(clamped_load=True)
            session.load_frame(session.f)
        elif sreq.img_mode == SeekImageMode.IMAGE_ONLY:
            session.load_frame(f_prev)
        elif sreq.img_mode == SeekImageMode.NO_IMAGE:
            session.load_f(clamped_load=True)
            session.img = img_prev

        frame_changed = True

        invalidated = frame_changed = True
    just_seeked = sreq

    # Delay
    if mode == RenderMode.PAUSE:
        time.sleep(1 / 60)

    # Playback accumulator
    if not mode == RenderMode.PAUSE and not just_started_render:
        loop.last_frame_dt = time.time() - (loop.last_frame_time or time.time())
        loop.last_frame_time = time.time()

        loop.dt_accumulated += loop.last_frame_dt
    else:
        loop.last_frame_time = None
        loop.last_frame_dt = 9999999
        loop.dt_accumulated = 0

    # Fast-mode means that it takes less than a second (>1 FPS) to render
    is_fast_mode = mode == RenderMode.PLAY and loop.last_frame_dt < 1
    if is_fast_mode and mode == RenderMode.RENDER:
        is_fast_mode = render_repetition == RenderingRepetition.FOREVER

    # Playback advance
    f_start = session.f
    f_changed = False

    while loop.dt_accumulated >= 1 / rv.fps:
        # print("FRAME += 1", session.f, "->", session.f + 1)
        session.f += 1
        loop.dt_accumulated -= 1 / rv.fps
        f_changed = True
        frame_changed = True

        # Stop after one iteration since we are saving, we want
        # to render every frame without fail
        if should_save() or is_cli():
            loop.dt_accumulated = 0
            break

    if f_changed:
        session.load_f(img=not mode == RenderMode.RENDER)
        hud.update_draw_signals()
        catchedup_max = session.f >= rv.n

        if mode == RenderMode.PLAY:
            frame_exists = session.f_exists
            catchedup_end = not frame_exists and not loop.was_paused and play.last_play_start_frame < session.f_last
            catchedup = play.end_frame and session.f >= play.end_frame > f_start
            if (catchedup_end or catchedup) and mode != RenderMode.RENDER:
                # -> LOOPING
                if play.looping:
                    seek(play.loop_frame)
                    mode = RenderMode.PLAY
                else:
                    # -> AUTOSTOP
                    play.end_frame = None
                    mode = RenderMode.PAUSE
                    requests.pause = True

        if catchedup_max and rv.n > 0:
            if enable_dev:
                # Loop back to start and keep playing, better for creative coding
                print("Catched up to max frame, looping.")
                seek(session.f_first)
            else:
                mode = RenderMode.PAUSE
                requests.pause = True
                seek(session.f_last)
                print(f"Finished rendering. (n={rv.n})")  # TODO UI notification
            return

    just_paused = mode == RenderMode.PAUSE and not loop.was_paused
    just_unpaused = mode != RenderMode.PAUSE and loop.was_paused
    if just_unpaused: _invoke_safe(on_start_playback)
    if just_paused: _invoke_safe(on_stop_playback)
    if frame_changed:
        _invoke_safe(on_frame_changed, session.f)
        _invoke_safe(on_t_changed, session.t)

    # Audio commands
    if enable_gui:
        if not audio.is_playing() and is_fast_mode:
            # print("PLAYING AUDIO at ", session.t)
            audio.play(session.t)
        elif audio.is_playing() and not is_fast_mode:
            audio.stop()
        if audio.is_playing() and just_seeked:
            audio.seek(session.t)

    # Actual rendering
    if should_render():
        if loop.frame_callback:
            loop.frame_callback(session.f)
        else:
            frame()

        if frame_error or render_repetition == RenderingRepetition.ONCE:
            mode = RenderMode.PAUSE
    elif frame_changed:
        require_dry_run = not session.has_frame_data('hud') and session.f_exists and auto_populate_hud
        if require_dry_run:
            frame(dry=True)
    else:
        pass

    if not mode == RenderMode.RENDER:
        loop.time_started = time.time()
        loop.n_frames = 0

    loop.was_paused = mode == RenderMode.PAUSE

def set_session(s):
    global session, invalidated
    session = s

    rv.session = s
    rv.fps = session.fps

    session.width = rv.w
    session.height = rv.h
    if session.img is None:
        session.img = np.zeros((rv.h, rv.w, 3), dtype=np.uint8)

    if enable_gui:
        session.seek_min(log=not enable_gui)
        requests.pause = True  # In GUI mode we let the user decide when to start the render
    else:
        session.seek_max()
        session.seek_new()

    invalidated = True

def set_image(img):
    global invalidated
    img = load_cv2(img)
    session.img = img
    rv.img = img
    invalidated = True

@trace_decorator
def frame(f=None, scalar=1, dry=False, auto_advance=True):
    """
    Render a frame.
    Args:
        f:
        scalar:
        dry:
        auto_advance: If true, automatically advance to the next frame after rendering

    Returns:

    """
    global mode, state, frame_error, invalidated

    f = int(f or session.f)

    state = RendererState.RENDERING

    session.f = f
    session.dev = enable_dev

    rv.start_frame(f, scalar)
    rv.dry = dry
    rv.trace = 'frame'

    # Print the header
    if rv.n > 0:
        ss = f'frame {rv.f} / {rv.n}'
    else:
        ss = f'frame {rv.f}'

    elapsed_seconds = (time.time() - loop.time_started) / 1000
    timedelta = datetime.timedelta(seconds=elapsed_seconds)
    elapsed_str = str(timedelta)
    ss += f' :: {loop.n_rendered / rv.fps:.2f}s rendered in {elapsed_str} ----------------------'
    if is_cli():
        print("")
        print(ss)

    hud.clear()

    # hud(x=rv.x, y=rv.y, z=rv.z, rx=rv.rx, ry=rv.ry, rz=rv.rz)
    # hud(music=rv.music, drum=rv.drum)
    # hud(amp=rv.amp, amp2=rv.amp2, l1=rv.l1)
    # hud(dtcam=rv.dtcam)

    invoke_frame()

    loop.start_f = session.f
    loop.start_img = session.img
    session.img = rv.img
    session.fps = rv.fps

    restore = frame_error or dry
    if restore:
        # Restore the frame number
        session.seek(loop.start_f)
        session.img = loop.start_img
    else:
        session.set_frame_data('hud', list(hud.rows))

        if should_save():
            session.f = rv.f
            session.load_f()
            session.save()
            if f > session.f_last:
                session.f_last = f
            # session.save_data()

        loop.last_rendered_frame = session.f

        if enable_save_hud and not enable_dev:
            hud.save(session, hud.to_pil(session))

        loop.n_rendered += 1

    state = RendererState.READY
    invalidated = True
    if frame_error:
        requests.pause = True


# endregion

# region Control Commands

def stop():
    requests.stop = True

def toggle_pause():
    play.looping = False
    play.end_frame = 0
    requests.pause = PauseRequest.TOGGLE
    global mode
    if mode == RenderMode.RENDER:
        mode = RenderMode.PLAY

def set_pause():
    """
    Set the pause state.
    """
    play.looping = False
    play.end_frame = 0
    requests.pause = PauseRequest.PAUSE

def set_play():
    """
    Set the play state.
    """
    play.looping = False
    play.end_frame = 0
    requests.pause = PauseRequest.PLAY

def seek(f_target, *, with_pause=None, clamp=True, img_mode=SeekImageMode.NORMAL):  # TODO we need to check usage of this to update imgmode
    """
    Seek to a frame.
    Note this is not immediate, it is a request that will be handled as part of the render loop.
    """
    # if its_pizza_time: return # TODO move to gui

    if clamp:
        if session.f_first is not None and f_target < session.f_first:
            f_target = session.f_first
        if session.f_last is not None and f_target >= session.f_last + 1:
            f_target = session.f_last + 1

    requests.seek = SeekRequest(f_target, img_mode)
    play.looping = False

    if with_pause is not None:
        requests.pause = with_pause

    # if not pause:
    #     print(f'NON-PAUSING SEEK MAY BE BUGGY')


def seek_t(t_target):
    """
    Seek to a time in seconds.
    Note this is not immediate, it is a request that will be handled as part of the render loop.
    Args:
        t_target: The time in seconds.
    """
    f_target = int(rv.fps * t_target)
    seek(f_target)

def toggle_render_repetition():
    global render_repetition
    if render_repetition == RenderingRepetition.ONCE:
        render_repetition = RenderingRepetition.FOREVER
    else:
        render_repetition = RenderingRepetition.ONCE


def toggle_render(repetition=RenderingRepetition.FOREVER):
    global mode, render_repetition
    render_repetition = repetition
    if mode == RenderMode.RENDER:
        set_render_mode(RenderMode.RENDER)
    else:
        set_render_mode(RenderMode.RENDER)

def render_once(m):
    """
    Render a frame.
    Note this is not immediate, it is a request that will be handled as part of the render loop.
    """
    global mode, render_repetition
    set_render_mode(RenderMode.RENDER)
    render_repetition = RenderingRepetition.ONCE

def render_forever():
    global mode, render_repetition
    set_render_mode(RenderMode.RENDER)
    render_repetition = RenderingRepetition.FOREVER

def set_render_mode(newmode):
    global mode
    mode = newmode

    global frame_error
    frame_error = False

# endregion

# region Event Hooks
# def on_audio_playback_start(t):
#     global request_pause
#
#     seek_t(t)
#     requests.request_pause = False
#
#
# def on_audio_playback_stop(t_start, t_end):
#     global request_pause
#
#     seek_t(t_start)
#     requests.request_pause = True


# audio.on_playback_start.append(on_audio_playback_start)
# audio.on_playback_stop.append(on_audio_playback_stop)
# endregion
