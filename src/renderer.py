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
import sys
import threading
import time
import traceback
import types
from enum import Enum
from pathlib import Path

import cv2
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
from src.lib import loglib
from src.lib.corelib import invoke_safe
from src.lib.loglib import printbold, trace, trace_decorator
from src.party import tricks
from src.party import maths
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
	# -> Also checks the session path automatically as part of the logic
]

# These globals will NOT be persisted when reloading the script
module_reloader_blacklisted_global_persistence = [
	'prompt'
]

enable_dev = args.dev or args.readonly  # Are we in dev mode?
enable_gui = False  # Are we in GUI mode? (hobo, ryusig, audio)
enable_readonly = args.readonly  # Cannot render, used as an image viewer
enable_save = True  # Enable saving the frames
enable_save_hud = False  # Enable saving the HUD frames
enable_unsafe = args.unsafe  # Should we let calls to the script explode so we can easily debug?
enable_bake_on_script_reload = True  # Should we bake the script on reload?
detect_script_every = 1
auto_populate_hud = False  # Dry runs to get the HUD data when it is missing, can be laggy if the script is not optimized
render_progress = 0

log = loglib.make_log('renderer')
logerr = loglib.make_logerr('renderer')


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

	def clear(self):
		self.seek = None
		self.script_check = False
		self.render = None
		self.pause = PauseRequest.NONE
		self.stop = False


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
script_modpath = None
script_time_cache = {}
last_hotreload_error = False  # Whether a hotreload
last_frame_error = False  # Whether the last frame errored out

# Signals
invalidated = True  # This is to be handled by a GUI, such as hobo
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
	return enable_gui and not args.remote


def is_gui_dispatch():
	return enable_gui and enable_gui_dispatch and mode.value < RenderMode.RENDER.value


# region Emits
@trace_decorator
def _emit(name):
	set_lib_refs()

	if hasattr(script, name):
		func = getattr(script, name)
		func()
	if hasattr(script, 'callback'):
		script.callback(name)


def _invoke_safe(*args, **kwargs):
	return invoke_safe(*args, unsafe=enable_unsafe, **kwargs)


# endregion

# region Script

def get_script():
	if script is None:
		load_script()

	return script

def detect_hotreload_changes():
	"""
	Detect changes in the scripts folder and reload all modules.
	Returns:
		Changed files (list of Path),
		Added files (list of Path)
	"""
	if session is None:
		return [], []

	def detect_changes(path):
		changed = []
		added = []

		# mprint(f'detect_changes({path})')

		# Check all .py recursively
		for file in Path(path).rglob('*.py'):
			key = file.relative_to(paths.root).as_posix()
			is_new = key not in script_time_cache
			if is_new:
				script_time_cache[key] = file.stat().st_mtime
				added.append(file)
				continue

			# Compare last modified time of the file with the cached time
			# mprint("Check file:", file.relative_to(path).as_posix())
			if script_time_cache[key] < file.stat().st_mtime:
				script_time_cache[key] = file.stat().st_mtime
				changed.append(file)
				modified = True
		# mprint(key, file.stat().st_mtime)

		return changed, added

	all_changed = []
	all_added = []
	for p in [*script_change_detection_paths, session.dirpath]:
		changed, added = detect_changes(p)
		all_changed.extend(changed)
		all_added.extend(added)

	return all_changed, all_added


def apply_hotreload_if_changed():
	global last_hotreload_error, last_frame_error

	had_errored = last_hotreload_error
	last_hotreload_error = False

	changed, added = detect_hotreload_changes()
	for filepath in changed:
		modpath = paths.filepath_to_modpath(filepath)
		if modpath is None:
			continue

		log('')
		log('==============================================')
		log(chalk.dim(chalk.blue(f"Change detected in {modpath} ({filepath})")))

		reload_module(modpath)

		if modpath == script_modpath:
			log(f"Saving script backup for frame {session.f}")
			session.save_script_backup()

	# Cache the script for this frame
	is_rendering = mode == RenderMode.RENDER
	if is_rendering and last_hotreload_error:
		# Stop rendering if there was an error
		last_frame_error = True
		set_pause()
	elif not is_rendering and not last_hotreload_error:
		last_frame_error = False

# if last_frame_error or had_errored:
#     render_forever()


# if changed or added:
#     reload_script()
# if changed or added:
#     mprint(chalk.dim(chalk.blue("Change detected in scripts, reloading")))

def reload_script(hard=False):
	global script
	global mode
	global init_state

	if session is None:
		log("No session, cannot reload script")
		return

	if hard:
		init_state = InitState.UNINITIALIZED
		log("Reloading script (hard) ...")
	else:
		log("Reloading script...")

	load_script()

	# Switch back to rendering automatically if it was stopped (error)
	if not mode == RenderMode.RENDER and is_cli():
		mode = RenderMode.RENDER


def load_script(scriptname=None):
	if session is None: return

	global script, script_name, script_modpath
	global last_hotreload_error, last_frame_error

	callbacks.clear()
	last_frame_error = False

	with trace('renderer.load_script'):
		filepath = ''

		# Guess script name from action arg
		if not scriptname:
			_, scriptname = parse_action_script(jargs.args.action)
			if scriptname is not None:
				filepath = get_script_file_path(script_name)

		if not filepath:
			filepath = session.res_script(scriptname, touch=True)
		if not scriptname and filepath:
			scriptname = filepath.stem

		# Reload all modules in the scripts folder
		modpath = paths.get_script_module_path(filepath)

		script_name = scriptname
		script_modpath = modpath

		reload_module(modpath=modpath)
		session.save_script_backup()

	_invoke_safe(on_script_loaded)


def reload_modules(exclude):
	modules_to_reload = []
	for x in sys.modules:
		for modpath in allowed_module_reload_names:
			if x.startswith(modpath) and modpath != exclude:
				modules_to_reload.append(x)

	for modpath in modules_to_reload:
		reload_module(modpath)


def reload_module(modpath, keep_globals=True):
	global script, last_hotreload_error

	# Get the old
	if enable_unsafe:
		module = importlib.import_module(modpath)
	else:
		try:
			module = importlib.import_module(modpath)
		except Exception as e:
			chalk.red("<!> MODULE ERROR <!>")
			traceback.print_exc()
			last_hotreload_error = True
			return

	oldglobals = module.__dict__.copy()
	oldglobals = {k: v for k, v in oldglobals.items() if not isinstance(v, types.FunctionType) and k not in module_reloader_blacklisted_global_persistence}  # Don't keep functions
	if modpath == script_modpath:
		mpath = modpath
		if script is None:
			rv.__init__()

			with trace(f'renderer.load_script.importlib.import_module({mpath})'):
				log(f'--> {session.name}.{script_name}.import')
				if enable_unsafe:
					script = importlib.import_module(mpath, package='imported_renderer_script')
				else:
					try:
						script = importlib.import_module(mpath, package='imported_renderer_script')
					except Exception as e:
						chalk.red("<!> SCRIPT ERROR <!>")
						traceback.print_exc()
						last_hotreload_error = True
		else:
			log(f'--> {session.name}.{script_name}.reload')
			if enable_unsafe:
				importlib.reload(script)
			else:
				try:
					importlib.reload(script)
				except Exception as e:
					chalk.red("<!> SCRIPT ERROR <!>")
					traceback.print_exc()
					last_hotreload_error = True
	else:
		module = sys.modules[modpath]  # can't we just use the module we already have?
		try:
			importlib.reload(module)
		except Exception as e:
			chalk.red("<!> MODULE ERROR <!>")
			traceback.print_exc()
			last_hotreload_error = True

	# Restore the globals
	if module is not None and oldglobals is not None:
		# # Remove globals that were initialized by the module (newer)
		# for k in list(module.__dict__.keys()):
		# 	if k not in oldglobals:
		# 		del module.__dict__[k]
		module.__dict__.update(oldglobals)

	# update functions for every instance of each class defined in the module using gc
	# import inspect
	# classdefs = dict()
	# classtypes = list()
	# for name, classobj in module.__dict__.items():
	# 	if inspect.isclass(classobj) and classobj.__module__ == module.__name__:
	# 		classdefs[name] = classobj
	# 		classtypes.append(classobj)
	# # convert classtypes to a tuple
	# classtypes = tuple(classtypes)
	# import gc
	# for obj in gc.get_objects():
	# 	if isinstance(obj, classtypes):
	# 		log(f"Updating '{obj.__class__.__name__}' class type")
	# 		newclass = classdefs[obj.__class__.__name__]
	# 		# Is one of the updated classes --> update the functions
	# 		for name, obj in obj.__dict__.items():
	# 			# Get the new one and set
	# 			if isinstance(obj, types.FunctionType):
	# 				log(f'Updating {obj.__name__}')
	# 				newobj = getattr(newclass, name)
	# 				setattr(obj, name, newobj)

	# module.__dict__.update(oldglobals.get('annotations', {}))
	# module.__dict__.update(oldglobals.get('__annotations__', {}))

	if modpath == script_modpath and enable_bake_on_script_reload:
		invoke_bake()


def invoke_init():
	# Things that are not strictly necessary, but are useful for most scripts
	global init_state

	if init_state.value < InitState.LIGHT.value:
		log(f'--> {session.name}.{script_name}.init')
		_invoke_safe(_emit, 'init')
		init_state = InitState.LIGHT

	if init_state.value < InitState.HEAVY.value and not enable_dev:
		log(f'--> {session.name}.{script_name}.init_heavy')
		try:
			invoke_safe(_emit, 'init_heavy', unsafe=True)
			init_state = InitState.HEAVY
			invoke_bake()
		except Exception as e:
			chalk.red("<!> SCRIPT ERROR <!>")
			traceback.print_exc()
			last_hotreload_error = True


def script_bool(name):
	if script is not None:
		v = script.__dict__.get(name, False)
		if v is None:
			return False
		return True

def script_get(name, default=None):
	if script is not None:
		return script.__dict__.get(name, None)


def script_has(name):
	if script is not None:
		return script.__dict__.get(name, None) is not None


@trace_decorator
def before_invoke():
	invoke_init()

# session.w = rv.w
# session.h = rv.h


@trace_decorator
def invoke_bake():
	from src.party import std
	from src.party import maths

	rv.clear_signals()
	# rv.load_signal_arrays()
	rv.resize_signals_to_n()
	before_invoke()
	maths.reset()

	log(f'--> {session.name}.{script_name}.bake')

	rv.is_array_mode = True
	std.init_media(demucs=script_bool('demucs'))
	if script_has('bake'):
		_invoke_safe(_emit, 'bake')
	elif script_has('start'):
		log("script.start is a deprecated name, please rename to bake (same functionality)")
		_invoke_safe(_emit, 'start')
	rv.is_array_mode = False

	if enable_gui:
		from src.rendering import hobo
		hobo.on_script_baked()


@trace_decorator
def invoke_frame():
	from src.party import std
	global last_frame_error

	invoke_init()
	before_invoke()

	if not enable_dev and rv.n > 0:
		if 'promptneg' in script.__dict__:
			rv.promptneg = script.__dict__['promptneg']
		if 'prompt' in script.__dict__:
			_invoke_safe(std.set_prompt, **script.__dict__)
	# std.refresh_prompt(**script.__dict__)

	if rv.nprompt is not None and not enable_dev:
		from src.party.ravelang import pnodes
		rv.prompt = pnodes.eval_prompt(rv.nprompt, rv.t)

	if std.init_img_exists():
		init_anchor = (.5, .5)
		if rv.has_signal('init_anchor_x'): init_anchor = (rv.init_anchor_x, init_anchor[1])
		if rv.has_signal('init_anchor_y'): init_anchor = (init_anchor[0], rv.init_anchor_y)
		size_mode = 'crop'
		if rv.has_signal('init_stretch') and rv.init_stretch > 0.5:
			size_mode = 'resize'

		rv.init_img = session.res_frame_cv2('init', fps=rv.fps, anchor=init_anchor, size_mode=size_mode)
		rv.init_img = cv2.resize(rv.init_img, (rv.w, rv.h), interpolation=cv2.INTER_AREA)
		hud.snap('init', rv.init_img)


	if session.f > 1:
		rv.last_img = session.res_frame_cv2(session.f - 1, fps=rv.fps, size_mode=False)
	else:
		rv.last_img = np.zeros((rv.h, rv.w, 3), dtype=np.uint8)

	if is_run_complete():
		global mode
		mode = RenderMode.PAUSE
		log(f"Render complete, no data left. (f={session.f} n={rv.n})")
		return

	with trace('invoke_frame.f1_init_load'):
		if rv.f == 1:
			if session.res('init'):
				rv.img = session.res_frame_cv2('init')
			else:
				rv.img = np.zeros((rv.h, rv.w, 3), dtype=np.uint8)
			rv.resize(True)

	with trace('invoke_frame.invocation'):
		last_frame_error = not _invoke_safe(_emit, 'frame', failsleep=0.25)


# endregion

# region Core functionality

def is_paused():
	return mode == RenderMode.PAUSE


def is_running():
	return state == RendererState.READY or state == RendererState.RENDERING


def should_render():
	ret = mode == RenderMode.RENDER or is_cli()
	ret = ret and (session.f != loop.last_rendered_frame or mode == RenderMode.PAUSE)  # Don't render the same frame twice (if we render faster than playback)
	ret = ret and not last_hotreload_error and not last_frame_error  # User must fix the script, it's too likely to cause crashes
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
	global detect_script_every

	enable_dev = gui

	set_lib_refs()
	change_session(_session or get_discore_session())

	# GUI setup
	# ----------------------------------------
	enable_gui = gui
	detect_script_every = -1 if gui else 1
	# if gui and not userconf.gui_main_thread:
	#     # Run the GUI on 2nd thread
	#     threading.Thread(target=ui_thread_loop, daemon=True).start()

	invalidated = True
	state = RendererState.READY

	# signal.signal(signal.SIGTERM, handle_sigterm)


	printbold("Renderer initialized.")

	return session


def set_lib_refs():
	from scripts.interfaces import comfy_lib
	from src.party.ravelang import pnodes

	if script:
		script.rv = rv
		script.s = session
		script.ses = session
		script.session = session
		script.f = rv.f
		script.dt = rv.dt
		script.fps = rv.fps

	tricks.rv = rv
	hud.rv = rv
	maths.rv = rv
	pnodes.rv = rv
	tricks.session = session
	hud.session = session
	maths.session = session
	pnodes.session = session
	comfy_lib.session = session
	comfy_lib.rv = rv


# def handle_sigterm():
#     print("renderer sigterm handler")


def run(lo=None, hi=None, callback_frame=None):
	global state
	global enable_dev
	global enable_gui
	global invalidated
	global mode

	if args.dry:
		return

	print()
	print("==== RUNNING RENDERER ===")

	def start_ui():
		from src.rendering import hobo

		hobo.rv = rv
		hobo.audio = audio
		hobo.start()

		requests.stop = True

	def start_loop():
		while not requests.stop or (session is not None and session.f < hi):
			if is_gui_dispatch():
				time.sleep(0.1)
				continue

			run_iter()

	# LOOP CONF
	# ----------------------------------------
	loop.last_frame_time = 0
	loop.last_script_check = 0
	loop.dt_accumulated = 0
	loop.frame_callback = callback_frame
	if session:
		if lo is not None:
			session.seek(lo)
		elif enable_gui:
			session.seek_max()
		else:
			session.seek_new()

	# Go immediately into rendering
	immediately_start_rendering = not enable_gui or jargs.args.start is not None

	if immediately_start_rendering:
		if isinstance(jargs.args.start, bool):
			seek(session.f_last + 1)
		elif isinstance(jargs.args.start, (int, str)):
			seek(session.f_last + 1)
		else:
			seek(session.f_last + 1)
		render_forever()
		enable_dev = False

	if hi is None:
		hi = math.inf

	# LOOP STARTUP
	# ----------------------------------------
	if enable_gui:
		if userconf.gui_main_thread:
			printbold('Starting UI (main thread)')
			threading.Thread(target=start_loop).start()  # args=(lo, hi, callback_frame))
			printbold('Starting render loop (off thread)')
			start_ui()
		else:
			printbold('Starting UI (off thread)')
			threading.Thread(target=start_ui, daemon=True).start()
			start_loop()
	else:
		start_loop()

	# LOOP SHUTDOWN
	# ----------------------------------------
	if session is not None:
		session.save_data()

	state = RendererState.STOPPED


# @trace_decorator
def run_iter():
	global mode, invalidated
	frame_changed = False

	# Script live reload detection
	with trace("renderiter.reload_script_check"):
		script_elapsed = time.time() - loop.last_script_check
		if requests.script_check \
			or -1 < detect_script_every < script_elapsed \
			or is_cli():
			requests.script_check = False
			apply_hotreload_if_changed()
			loop.last_script_check = time.time()

	# Flush render request
	just_started_render = False
	if requests.render:
		mode = RenderMode.RENDER
		just_started_render = True
		requests.render = False
		loop.time_started = time.time()
		loop.n_frames = 0

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
		elif requests.pause == PauseRequest.PLAY:
			mode = RenderMode.PLAY

		requests.pause = PauseRequest.NONE

	# Flush seek requests
	sreq = requests.seek
	requests.seek = None
	if sreq:
		f_prev = session.f
		img_prev = session.img

		session.f = sreq.f_target
		if sreq.img_mode == SeekImageMode.NORMAL:
			session.load_f(clamped_load=True)
			session.load_f_img(session.f)
		elif sreq.img_mode == SeekImageMode.IMAGE_ONLY:
			session.load_f_img(f_prev)
		elif sreq.img_mode == SeekImageMode.NO_IMAGE:
			session.load_f(clamped_load=True)
			session.img = img_prev

		if session.f > session.f_last:
			session.f_last = session.f
			session.f_last_path = session.f_path
		elif session.f < session.f_first:
			session.f_first = session.f
			session.f_first_path = session.f_path

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
	f_start = session.f if session else 0
	f_changed = False

	while loop.dt_accumulated >= 1 / rv.fps:
		# print("FRAME += 1", session.f, "->", session.f + 1)
		session.f += 1
		loop.dt_accumulated -= 1 / rv.fps
		f_changed = True
		frame_changed = True
		invalidated = True

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
			catchedup_end = not frame_exists and not loop.was_paused and play.last_play_start_frame >= session.f_last - 1
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
			run_frame()

		if last_frame_error or render_repetition == RenderingRepetition.ONCE:
			mode = RenderMode.PAUSE
	elif frame_changed:
		require_dry_run = not session.has_frame_data('hud') and session.f_exists and auto_populate_hud
		if require_dry_run:
			run_frame(dry=True)
	else:
		pass

	loop.was_paused = mode == RenderMode.PAUSE

def is_run_complete():
	return mode == RenderMode.RENDER and session.f >= rv.n - 1

def change_session(s):
	global session, invalidated
	global script

	session = s
	invalidated = True
	rv.session = s
	rv.__init__()

	requests.clear()

	if session is None:
		return

	rv.fps = session.fps
	rv.n = int(session.f_last)

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

	with trace('Script loading'):
		script = None
		_invoke_safe(load_script, None)

	rv.init_frame(1)
	set_lib_refs()

	paths.store_last_session_name(s.name)

	if enable_gui:
		from src.rendering import hobo
		hobo.on_session_changed(rv.session)


def change_image(img):
	global invalidated
	img = load_cv2(img)
	session.img = img
	rv.img = img
	invalidated = True


@trace_decorator
def run_frame(f=None, scalar=1, dry=False, auto_advance=True):
	"""
	Render a frame.
	Args:
		f:
		scalar:
		dry:
		auto_advance: If true, automatically advance to the next frame after rendering

	Returns:

	"""
	global mode, state, last_frame_error, invalidated

	invoke_init()

	global render_progress
	render_progress = -1

	f = int(f or session.f)
	state = RendererState.RENDERING
	session.dev = enable_dev

	# Remove corrupted frames and walk back
	# with trace('corruption_check'):
	#     rv.start_frame(f, scalar, cleanse_nulls=False)
	#     while rv.prev_img is None:
	#         print("Deleting corrupted frame", f)
	#         session.delete_f(f)
	#         f -= 1
	#         rv.start_frame(f, scalar, cleanse_nulls=False)

	rv.init_frame(f, scalar)
	rv.dry = dry
	rv.trace = 'frame'

	with trace('printing'):
		# Print the header
		if rv.n > 0:
			ss = f'frame {rv.f} / {rv.n}'
		else:
			ss = f'frame {rv.f}'

		elapsed_seconds = (time.time() - loop.time_started) / 1000
		timedelta = datetime.timedelta(seconds=elapsed_seconds)
		elapsed_str = str(timedelta)
		ss += f' :: {loop.n_rendered / rv.fps:.2f}s rendered in {elapsed_str} ----------------------'
		# if is_cli():
		print("")
		print(ss)

	hud.clear()
	# hud(x=rv.x, y=rv.y, z=rv.z, rx=rv.rx, ry=rv.ry, rz=rv.rz)
	# hud(music=rv.music, drum=rv.drum)
	# hud(amp=rv.amp, amp2=rv.amp2, l1=rv.l1)
	# hud(dtcam=rv.dtcam)
	invoke_frame()

	with trace('post_render'):
		loop.start_f = session.f
		loop.start_img = session.img
		session.img = rv.img
		session.fps = rv.fps

		restore = last_frame_error or dry
		if restore:
			# Restore the frame number
			session.seek(loop.start_f)
			session.img = loop.start_img
		else:
			session.set_frame_data('hud', list(hud.rows))

			if should_save():
				session.f = rv.f
				session.load_f(img=False)  # Allows overwriting existing images, otherwise we lose this new frame here
				session.save()
				if f > session.f_last:
					session.f_last = f
					session.f_last_path = session.f_path
			# session.save_data()

			loop.last_rendered_frame = session.f

			if enable_save_hud and not enable_dev:
				hud.save(session, hud.to_pil(session))

			loop.n_rendered += 1

	rv.unset_signals()

	state = RendererState.READY
	invalidated = True
	if last_frame_error:
		requests.pause = True


# endregion

# region Control Commands

def stop():
	if not session: return
	requests.stop = True


def toggle_pause():
	global mode
	if not session: return
	if mode == RenderMode.RENDER: return
	play.looping = False
	play.end_frame = 0
	requests.pause = PauseRequest.TOGGLE
	if mode == RenderMode.RENDER:
		mode = RenderMode.PLAY


def set_pause():
	"""
	Set the pause state.
	"""
	if not session: return
	if mode == RenderMode.RENDER: return
	play.looping = False
	play.end_frame = 0
	requests.pause = PauseRequest.PAUSE


def set_play():
	"""
	Set the play state.
	"""
	if not session: return
	if mode == RenderMode.RENDER: return
	play.looping = False
	play.end_frame = 0
	requests.pause = PauseRequest.PLAY


def seek(f_target, *, with_pause=None, clamp=True, img_mode=SeekImageMode.NORMAL):  # TODO we need to check usage of this to update imgmode
	"""
	Seek to a frame.
	Note this is not immediate, it is a request that will be handled as part of the render loop.
	"""
	if not session:
		return False
	if mode == RenderMode.RENDER:
		return False

	if isinstance(f_target, str):
		f_seconds = paths.parse_time_to_seconds(f_target)
		f_target = rv.to_frame(f_seconds)

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
	return True


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
	if not session: return
	global render_repetition
	if render_repetition == RenderingRepetition.ONCE:
		render_repetition = RenderingRepetition.FOREVER
	else:
		render_repetition = RenderingRepetition.ONCE


def toggle_render(repetition=RenderingRepetition.FOREVER):
	if not session: return
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
	if not session: return
	global mode, render_repetition
	set_render_mode(RenderMode.RENDER)
	render_repetition = RenderingRepetition.ONCE


def render_forever():
	if not session: return
	global mode, render_repetition
	set_render_mode(RenderMode.RENDER)
	render_repetition = RenderingRepetition.FOREVER


def set_render_mode(newmode):
	if not session: return
	global mode
	global last_frame_error
	mode = newmode
	last_frame_error = False

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
