import math
import os
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from munch import Munch
from tqdm import tqdm

import userconf
from src.lib.printlib import cputrace, trace, trace_decorator
from . import convert, paths
from .convert import load_json, load_pil, save_img, save_json, load_cv2
from .logs import logsession, logsession_err
from .paths import get_leadnum, get_leadnum_zpad, get_max_leadnum, get_min_leadnum, is_leadnum_zpadded, leadnum_zpad, parse_frames
from ..lib import corelib
from ..lib.corelib import shlexproc

FRAME_EXTRACTION_INFO_FILE_NAME = 'info.json'


class Session:
	"""
	A kind of wrapper for a directory with number sequence padded with 8 zeroes, and
	optionally some saved metadata.

	00000001.png
	00000002.png
	00000003.png
	...
	00000020.png
	00000021.png
	00000022.png

	The session tracks a 'current' frame (f). Upon first loading a session,
	the current frame is set to the last frame in the directory.
	The session defines 'seeking' functions to advance the current frame.

	The session also tracks a context data.
	"""

	def __init__(self, name_or_abspath=None, load=True, fixpad=False, log=True):
		self.args = dict()
		self.data = Munch()

		self.processing_thread = None
		self.cancel_processing = False
		self.dev = False

		# Context properties
		self.prompt = ''
		self.file = None
		self.fps = 24
		self._img = None
		self.w = 512
		self.h = 512

		# Directory properties, cached for performance
		self.f = 1
		self.f_first = 0
		self.f_last = 0
		self.f_path = ''
		self.f_exists = False
		self.f_first_path = ''
		self.f_last_path = ''
		self.suffix = ''

		if Path(name_or_abspath).is_absolute():
			self.dirpath = Path(name_or_abspath)
			self.name = Path(name_or_abspath).stem
		elif name_or_abspath is not None:
			self.name = name_or_abspath
			self.dirpath = paths.sessions / name_or_abspath
		else:
			self.valid = False
			logsession_err("Cannot create session! No name or path given!")
			return

		# self.dirpath = self.dirpath.resolve()

		if self.dirpath.exists():
			if load:
				import jargs
				with cputrace('load', jargs.args.profile_session_load):
					self.load(log=log)
		else:
			if log:
				logsession("New session:", self.name)

		if fixpad:
			if self.dirpath.is_dir() and not is_leadnum_zpadded(self.dirpath):
				logsession("Session directory is not zero-padded. Migrating...")
				self.make_zpad(leadnum_zpad)

	def __str__(self):
		return f"Session({self.name} ({self.w}x{self.h}) at {self.dirpath} ({self.file}))"

	def exists(self):
		return self.dirpath.exists()

	def rmtree(self):
		shutil.rmtree(self.dirpath.as_posix())

	@staticmethod
	def now(prefix='', log=True):
		"""
		Returns: A new session which is timestamped to now
		"""
		name = datetime.now().strftime(paths.session_timestamp_format)
		return Session(f"{prefix}{name}", log=log)

	@staticmethod
	def recent_or_now(recent_window=math.inf):
		"""
		Returns: The most recent session, or a new session if none exists.
		args:
			recent_window: The number of seconds to consider a session recent. Outside of this window, a new session is created.
		"""
		if any(paths.sessions.iterdir()):
			latest = max(paths.sessions.iterdir(), key=lambda p: p.stat().st_mtime)
			# If the latest session fits into the recent window, use it
			if (datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)).total_seconds() < recent_window:
				return Session(latest)

		return Session.now()

	@trace_decorator
	def load(self, *, log=True):
		"""
		Load the session state from disk.
		If new frames have been added externally without interacting directly with a session object,
		this will update the session state to reflect the new frames.
		"""
		if not self.dirpath.exists():
			return

		lo, hi = get_leadnum(self.dirpath)
		self.f_first = lo or 0
		self.f_last = hi or 0
		self.f_exists = False

		self.suffix = self.det_suffix()
		self.f_first_path = self.det_frame_path(self.f_first) or 0
		self.f_last_path = self.det_frame_path(self.f_last) or 0
		self.f_path = self.f_last_path
		self.f = self.f_last
		self.load_f()
		self.load_frame()

		# Save the session data
		self.load_data()
		if self.f_exists:
			self.w = self._img.shape[1]
			self.h = self._img.shape[0]

		if log:
			if not any(self.dirpath.iterdir()):
				logsession(f"Loaded session {self.name} at {self.dirpath}")
			else:
				logsession(f"Loaded session {self.name} ({self.w}x{self.h}) at {self.dirpath} ({self.file})")

	@trace_decorator
	def load_f(self, f=None, *, clamped_load=False, img=True):
		"""
		Load the statistics for the current frame or any specified frame,
		and load the frame image.
		"""
		f = f or self.f

		self.f = f
		self.f_path = self.get_frame_path(f)
		self.file = self.get_frame_name(self.f)
		self.f_exists = False
		if img:
			if self.f <= self.f_last:
				self.f_exists = self.load_frame()
			elif clamped_load:
				self.f_exists = self.load_frame(self.f_last_path)

	@trace_decorator
	def load_frame(self, file: str | int | None = None, *, resize_to_session=False):
		"""
		Load a file, the current file in the session state, or a frame by number.
		Args:
			file: A path to a file

		Returns:
		"""
		if file is None:
			file = self.file
		if isinstance(file, str):
			file = Path(file)

		if isinstance(file, (int, np.int64, float)):
			file = self.det_frame_path(int(file))
		elif isinstance(file, Path) and file.is_absolute():
			file = file
		elif isinstance(file, Path):
			file = self.dirpath / self.file

		if file.suffix in paths.image_exts:
			if file.exists():
				self.img = convert.load_cv2(file, (self.w, self.h) if resize_to_session else None)
				if self.img is None:
					self.img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
				return True

			return False

	def load_data(self):
		self.data = load_json(self.dirpath / "session.json", None)
		if self.data:
			self.data = Munch(self.data)
		else:
			self.data = Munch()

		self.fps = self.data.get("fps", self.fps)

	def save(self):
		# if not path and self.file:
		#     path = self.res(self.file)
		# if not path:
		#     path = self.f_path
		#
		# if not Path(path).is_absolute():
		#     path = self.dirpath / path
		# save_num = self.f
		# if save_num is not None and save_num > self.f_last:
		#     self.f_last = save_num
		#     self.f_last_path = path
		# if save_num is not None and save_num < self.f_first:
		#     self.f_first = save_num
		#     self.f_first_path = path

		path = Path(self.f_path)
		if self.img is not None:
			path = path.with_suffix(self.suffix)
			save_img(self.img, path, with_async=True, img_format=self.suffix)

		self.file = path.name

		# Save the session data
		self.save_data()

		logsession(f"session.save({path})")
		return self

	def save_data(self):
		self.data.fps = self.fps
		save_json(self.data, self.dirpath / "session.json")

	def delete_f(self):
		return self.delete_frame(self.f)

	def delete_frame(self, f=None):
		# Delete the current frame
		f = f or self.f
		if f is None:
			return

		path = None
		exists = None
		if f == self.f:
			# Optimization
			exists = self.f_exists
			path = self.f_path
		else:
			path = self.det_frame_path(f)
			exists = path.exists()

		if exists:
			if f == self.f:
				path.unlink()
				self.f = np.clip(self.f - 1, 0, self.f_last)
				self.f_last = self.f or 0
				self.f_last_path = self.det_frame_path(self.f_last) or 0
				self.load_f()
			else:
				# Offset all frames after to make sequential
				path.unlink()
				self.make_sequential()
				self.load()

				logsession(f"Deleted {path}")
				return True

		return False

	def last_prop(self, propname: str):
		for arglist in self.args:
			for k, v in arglist:
				if k == propname:
					return v

		return None

	@property
	def nice_path(self):
		# If is relative to the sessions folder, return the relative path
		if self.dirpath.is_relative_to(paths.sessions):
			return self.dirpath.relative_to(paths.sessions)
		else:
			return self.dirpath.resolve()

	@property
	def t(self):
		return self.f / self.fps

	@property
	def img(self):
		return self._img

	@img.setter
	def img(self, value):
		self._img = convert.load_cv2(value)

	def get_frame_name(self, f):
		return str(f).zfill(8) + self.suffix

	def get_frame_path(self, f):
		return self.dirpath / self.get_frame_name(f)

	def get_current_frame_name(self):
		return self.get_frame_name(self.f)

	def det_frame_path(self, f, subdir='', suffix=None):
		if suffix is not None:
			p1 = (self.dirpath / subdir / str(f)).with_suffix(suffix)
			p2 = (self.dirpath / subdir / str(f).zfill(8)).with_suffix(suffix)
			if p1.exists(): return p1
			return p2  # padded is now the default
		else:
			if self.suffix:
				return self.det_frame_path(f, subdir, self.suffix)

			jpg = self.det_frame_path(f, subdir, '.jpg')
			png = self.det_frame_path(f, subdir, '.png')
			if jpg.exists():
				return jpg
			else:
				return png

	def det_suffix(self, f=None):
		if f is None:
			return self.det_suffix(self.f_last) \
				or self.det_suffix(self.f_first) \
				or self.det_suffix(1)

		fpath = self.det_frame_path(f)
		if fpath.exists():
			return fpath.suffix
		return '.png'

	def det_frame_pil(self, f, subdir=''):
		return load_pil(self.det_frame_path(f, subdir))

	def det_current_frame_path(self, subdir=''):
		return self.det_frame_path(self.f, subdir)

	def det_current_frame_exists(self):
		return self.det_current_frame_path().is_file()

	def det_f_first_path(self):
		return self.det_frame_path(get_min_leadnum(self.dirpath))

	def det_f_last_path(self):
		return self.det_frame_path(get_max_leadnum(self.dirpath))

	# def set(self, dat):
	#     from PIL import ImageFile
	#     ImageFile.LOAD_TRUNCATED_IMAGES = True
	#
	#     if dat is None:
	#         return
	#
	#     # Image output is moved into the context
	#     with trace(f"session.set({dat.shape if isinstance(dat, np.ndarray) else dat})"):
	#         self.img =
	#         if isinstance(dat, Image.Image):
	#             self._image = dat
	#             self._image_cv2 = None
	#         elif isinstance(dat, str) or isinstance(dat, Path):
	#             if paths.exists(dat):
	#                 import cv2
	#                 self._image_cv2 = cv2.imread(dat.as_posix())
	#                 if self._image_cv2 is not None:
	#                     self._image_cv2 = self._image_cv2[..., ::-1]
	#                 # self._image = Image.open(dat)
	#                 # self.file = dat
	#         elif isinstance(dat, list) and isinstance(dat[0], Image.Image):
	#             printerr("Multiple images in set_image data, using first")
	#             self._image = dat[0]
	#         elif isinstance(dat, np.ndarray):
	#             self._image_cv2 = dat

	def set_frame_data(self, key, v):
		f = self.f - 1
		if not key in self.data:
			self.data[key] = []

		# Make sure the list is long enough for self.f
		while len(self.data[key]) <= f:
			self.data[key].append(None)

		if isinstance(v, bool):
			v = int(v)

		self.data[key][f] = v

	def has_frame_data(self, key):
		f = self.f - 1
		if not key in self.data: return False
		if f >= len(self.data[key]): return False
		if self.data[key][f] is None: return False

		v = self.data[key][f]
		if isinstance(v, list) and len(v) == 0: return False

		return True

	def get_frame_data(self, key, f, *, clamp=True):
		f = self.f
		if clamp and self.f > self.f_last:
			f = self.f_last

		f -= 1
		if key in self.data and f < len(self.data[key]):
			return self.data[key][f]

		return 0

	def seek(self, i=None, *, log=False):
		if i is None:
			# Seek to next
			# self.f = get_next_leadnum(self.dirpath)
			self.f = self.f_last + 1
		elif isinstance(i, int):
			# Seek to i
			i = max(i, 1)  # Frames start a 1
			self.f = i
		else:
			logsession_err(f"Invalid seek argument: {i}")
			return

		# self._image = None
		self.load_f()
		self.load_frame()

		if log:
			logsession(f"({self.name}) seek({self.f})")

	def seek_min(self, *, log=False):
		if self.dirpath.exists() and any(self.dirpath.iterdir()):
			# Seek to next
			minlead = get_min_leadnum(self.dirpath)
			self.seek(minlead, log=log)

	def seek_max(self, *, log=False):
		if self.dirpath.exists() and any(self.dirpath.iterdir()):
			self.f = get_max_leadnum(self.dirpath)
			self.seek(self.f, log=log)

	def seek_next(self, i=1, log=False):
		self.f += i
		self.seek(self.f, log=log)

	def seek_new(self, log=False):
		self.seek(None, log=log)

	def subsession(self, name) -> "Session":
		if name:
			return Session(self.res(name))
		else:
			return self

	def res(self, resname: Path | str, extensions: str | list | tuple = None, *, return_missing=False, hide=False) -> Path | None:
		"""
		Get a session resource, e.g. init video
		- If subpath is absolute, returns it
		- If subpath is relative, tries to find it in the session dir
		- If not found, tries to find it in the parent dir
		- If not found, check userconf.init_paths and copy it to the session dir if it exists
		- If not found, returns None
		- Tries all extensions passed

		args:
			resname: The name of the resource
			extensions: A single extension or a list of extensions to try, e.g. ['mp4', 'avi']
			return_missing: If True, returns the path to the missing file instead of None
		"""
		resname = Path(resname)
		if hide:
			resname = resname.with_name(f'.{resname.name}')

		def try_path(path):
			if path.exists():
				return path
			elif path.with_name(f'.{path.name}').exists():
				return path.with_name(f'.{path.name}')
			return None

		def try_ext(path, ext):
			# Remove dot from extension
			if ext:
				if ext[0] == '.':
					ext = ext[1:]
				path = path.with_suffix('.' + ext)

			# Test
			checkpaths = [self.dirpath / path, paths.sessions / path]
			checkpaths.extend([p / path for p in userconf.init_paths])
			for i, p in enumerate(checkpaths):
				p = try_path(p)
				if p:
					is_copy = i >= 1
					if is_copy:
						# Copy to session dir
						src = p
						dst = self.dirpath / path
						paths.cp(src, dst)
						return dst

					return p

			return checkpaths[0] if return_missing else None

		if resname.is_absolute() and resname.exists():
			return resname
		elif isinstance(extensions, (list, tuple)):
			# Try each extension until we find one that exists
			for e in extensions:
				p = try_ext(resname, e)
				if p and p.exists():
					return p
			return p if return_missing else None
		else:
			return try_ext(resname, extensions)

	@trace_decorator
	def res_cv2(self, subpath: Path | str, *, ext: str = None) -> np.ndarray | None:
		"""
		Get a session resource, e.g. init video
		"""
		if not isinstance(subpath, (Path, str)):
			return convert.load_cv2(subpath)

		path = self.res(subpath, extensions=ext or ('jpg', 'png'))
		if path:
			im = convert.load_cv2(path)
			if im is not None and self.img is not None:
				im = convert.crop_or_pad(im, self.w, self.h)
			return im
		else:
			return None

	@trace_decorator
	def res_frame(self, res_name, framenum=None, *, subdir='', ext=None, loop=False, fps=None) -> Path | None:
		"""
		Get a session resource, and automatically fetch a frame from it.
		Usage:

		resid='video' # Get the current session frame from video.mp4
		resid=3 # Get frame 3 from the current session
		"""

		def get_dir_frame(dirpath, ext=None):
			frame = self.f
			if framenum is not None:
				frame = framenum

			json_path = dirpath / FRAME_EXTRACTION_INFO_FILE_NAME
			if json_path.exists():
				json = convert.load_json(json_path)
				if fps is not None and fps != json['fps']:
					frame = int(frame * (json['fps'] / fps))

			if loop:
				l = list(dirpath.iterdir())
				frame = frame % len(l)

			framestr = str(frame)
			framename = f"{framestr.zfill(paths.leadnum_zpad)}.{ext or 'jpg'}"

			return dirpath / framename

		# return paths.get_first_exist(
		#         self.res(stem / framename),
		#         dirpath / framename)

		# If the resid is a number, assume it is a frame number
		if isinstance(res_name, int):
			return self.det_frame_path(res_name, subdir=subdir)
		elif res_name is None:
			return Path(self.f_path)
		elif isinstance(res_name, str):
			if subdir:
				res_name = f"{res_name}/{subdir}"
			# If the resid is a string, parse it
			file = self.res(res_name, extensions=[*paths.video_exts, *paths.image_exts], return_missing=True)

			# File exists and is not a video --> return directly / same behavior as res(...)
			if file.exists() and paths.is_image(file) and framenum is None:
				return file
			elif file.exists() and paths.is_video(file):
				# Iterate dir and find the matching file, regardless of the extension
				framedir = self.extract_frames(file)
				return get_dir_frame(framedir)
			else:
				return get_dir_frame(self.res(res_name, return_missing=True), ext)

		elif isinstance(res_name, Path) and res_name.is_dir():
			return get_dir_frame(res_name / subdir)
		else:
			return get_dir_frame(self.res(res_name, return_missing=True), ext)

	# for file in l:
	#     if file.stem.lstrip('0') == framestr:
	#         if ext is None or file.suffix.lstrip('.') == ext.lstrip('.'):
	#             return file
	#
	# return None

	@trace_decorator
	def res_frame_count(self, resname, *, fps=None):
		dirpath = self.extract_frames(resname)
		if dirpath is None:
			return 0
		else:
			json_path = dirpath / FRAME_EXTRACTION_INFO_FILE_NAME
			if not json_path.exists():
				# Determine from directory file count, which is slower
				if fps is not None:
					raise ValueError(f"fCannot determine frame count from fps without json stats. (searched for {json_path})")

				return len(list(dirpath.iterdir()))
			else:
				# Determine from json data
				json = convert.load_json(json_path)
				if fps is not None and fps != json['fps']:
					return int(json['frame_count'] * (json['fps'] / fps))
				else:
					return json['frame_count']

	@trace_decorator
	def res_frame_cv2(self, resid, framenum=None, *, ext=None, loop=False, fps=None, anchor=(0.5, 0.5)):
		frame_path = self.res_frame(resid, framenum, ext=ext, loop=loop, fps=fps or self.fps)
		if frame_path is None or not frame_path.exists() and self.img is not None:
			return self.img

		# resize to fit
		with trace(f"res_frame_cv2: imread"):
			ret = cv2.imread(str(frame_path))
			ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)
		if ret is not None and self.img is not None:
			# Resize
			# with trace(f"res_frame_cv2: resize"):
			#     ret = cv2.resize(ret, (self.w, self.h))

			# Crop
			ret = convert.crop_or_pad(ret, self.w, self.h, 'black', anchor=anchor)

		# ret = ret[0:self.h, 0:self.w]

		return ret

	def res_framepil(self, name, ext=None, loop=False, resize=False):
		ret = load_pil(self.res_frame(name, ext, loop=loop))
		if resize and self.img is not None:
			ret = ret.resize((self.w, self.h))

		return ret

	def res_frameiter(self, resname, description="Enumerating...", load=True):
		framedir = self.extract_frames(resname)
		input_files = list(sorted(framedir.glob("*.jpg")))
		n = len(input_files)
		for i in tqdm(range(1, n), desc=description):
			path = input_files[i]
			img = load_cv2(path) if paths.is_image(path) and load else None
			yield i, img, path

	def res_frameiter_pairs(self, resname, description="Enumerating ...", load=True):
		framedir = self.extract_frames(resname)
		input_files = list(sorted(framedir.glob("*.jpg")))
		n = len(input_files)
		for i in tqdm(range(1, n - 1), desc=description):
			path1 = input_files[i]
			path2 = input_files[i+1]
			img1 = load_cv2(path1) if paths.is_image(path1) and load else None
			img2 = load_cv2(path2) if paths.is_image(path2) and load else None

			yield i, img1, img2, path1, path2


	def res_init(self, name=None):
		name = name or 'init'
		img = self.res(name, extensions=paths.image_exts)
		mus = self.res(name, extensions=paths.audio_exts)
		vid = self.res(name, extensions=paths.video_exts)
		return img, mus, vid

	def res_music(self, name=None, *, optional=True):
		"""
		Get the music resource for this session, or specify by name and auto-detect extension.
		"""

		def _res_music(n):
			n = n or 'music'
			if self.exists:
				return self.res(f"{n}", extensions=paths.audio_exts)

		if name:
			v = _res_music(name)
		else:
			v = _res_music('music') or _res_music('init')
		if not v and not optional:
			raise FileNotFoundError("Could not find music file in session directory")
		return v

	def res_script(self, name='script', touch=False):
		"""
		Get the script resource for this session, or specify by name and auto-detect extension.
		"""
		name = name or 'script'
		if not name.endswith('.py'):
			name += '.py'

		path = self.res(name, return_missing=True)
		if touch:
			paths.touch(path)

		return path

	def make_sequential(self, *, fill=False):
		"""
		Rename all session frame files to sequential numbers

		Args:
			fill: Create new frames by copying the last framem, do not rename anything.

		Returns:

		"""
		self.make_zpad()

		files = list(self.dirpath.iterdir())
		files.sort()
		i = 1
		for file in files:
			if file.is_file():
				try:
					v = int(file.stem)
					src = file
					dst = self.det_frame_path(i)
					print(f'Rename {src} -> {dst} / off={v - i}')

					dst = dst.with_stem(f'__{dst.stem}')
					shutil.move(file, dst)

					i += 1
				except:
					pass

		# Iterate again to remove the __ prefix (we do this to avoid overwriting files)
		files = list(self.dirpath.iterdir())
		files.sort()
		for file in files:
			if file.is_file():
				if file.stem.startswith('__'):
					src = file
					dst = file.with_stem(file.stem[2:])
					shutil.move(src, dst)

	def make_full(self):
		"""
		Fill missing frames by copying the last frame.
		"""
		# track an index
		# For each frame
		#   if the frame exists, advance and continue
		#   otherwise create a new frame by copying the last frame

		self.make_zpad()

		files = list(self.dirpath.iterdir())
		files.sort()
		i = 1
		last = None
		for file in files:
			try:
				v = int(file.stem)
				# Fill missing frames
				for j in range(i + 1, v):
					shutil.copy(last, self.det_frame_path(j))
					print(f'Fill {j} / {last} -> {self.det_frame_path(j)}')
				i = v
				last = file
			except:
				pass

	def extract_init(self, name='init'):
		frame_path = self.extract_frames(name)
		music_path = self.extract_music(name)
		return frame_path, music_path

	def extract_music(self, src='init', overwrite=False, hide=False):
		input = self.res(src, extensions=paths.video_exts)
		if not input.exists():
			print(f"Session.extract_music: Could not find video file {input.name}")
			return

		output = self.res(f"{src}.wav", hide=hide, return_missing=True)
		cmd = f'ffmpeg -i {input} -acodec pcm_s16le -ac 1 -ar 44100 {output}'
		if output.exists():
			if not overwrite:
				print(f"Music extraction already exists for {input.name}, skipping ...")
				return
			paths.rm(input)

		os.system(cmd)

	def res_frame_dirpath(self, name, frames: tuple | None = None):
		src = self.res(name, extensions=paths.video_exts, return_missing=True)
		if not src.exists():
			print(f"Session.extract_frames: Could not find video file {src.name}")
			return None

		lo, hi, name = self.parse_frames(frames, src.stem)

		dst = self.res(f'.{name}/', return_missing=True)
		return dst

	def res_frame_pattern(self, path):
		pattern = f'{path}/%0{paths.leadnum_zpad}d.jpg'
		return pattern

	@trace_decorator
	def extract_frames(self, name, nth_frame=1, frames: tuple | None = None, w=None, h=None, overwrite=False, warn_existing=False, minsize=448, fps=None) -> Path | str | None:
		if Path(name).is_dir():
			return name

		src = self.res(name, extensions=paths.video_exts)
		dst = self.res_frame_dirpath(name)
		if dst:
			if dst.exists():
				if not overwrite:
					if warn_existing:
						logsession(f"Frame extraction already exists for {src.name}, skipping ...")
					return dst
				else:
					logsession(f"Frame extraction already exists for {src.name}, overwriting ...")

			pattern = self.res_frame_pattern(dst)
			lo, hi, name = self.parse_frames(frames, src)
			fps = fps or self.fps

			vf = ""
			if frames is not None:
				vf += f'select=between(t\,{lo}\,{hi})'
			else:
				vf = f'select=not(mod(n\,{nth_frame}))'

			vf, w, h = vf_rescale(vf, max(minsize, self.w), max(minsize, self.h), self.w, self.h)

			paths.rmclean(dst)
			corelib.shlexrun(['ffmpeg',
			                  '-i', f'{src}',
			                  '-vf', f'{vf}',
			                  '-q:v', '5',
			                  '-loglevel',
			                  'error',
			                  '-stats',
			                  pattern])

			# Delete frames to reach the target fps
			orig_fps = get_video_fps(src)
			if fps is not None:
				# Delete frames to reach the target fps
				threshold = orig_fps / fps
				elapsed = 0

				for i, img, file in self.res_frameiter(dst, f"Timescaling the extracted frames to match target fps {fps}"):
					# Check if the frame should be deleted based on the threshold
					if elapsed >= threshold:
						elapsed -= threshold
					else:
						paths.rm(file)
					elapsed += 1

				Session(dst).make_sequential()
			else:
				fps = orig_fps

			frame_count = len(list(dst.iterdir()))
			convert.save_json({
				'fps': fps,
				'w': w,
				'h': h,
				'frame_count': frame_count
			}, dst / 'info.json')

			return dst
		else:
			raise FileNotFoundError(f"Could not find video file {src.name}")

	def make_video(self, fps=None, skip=0, bg=False, music='', music_start=None, frames=None, fade_in=None, fade_out=None, w=None, h=None, bv=None, ba='320k'):
		# call ffmpeg to create video from image sequence in session folder
		# do not halt, run in background as a new system process
		if fps is None:
			fps = self.fps

		# Detect how many leading zeroes are in the frame files
		lzeroes = get_leadnum_zpad(self.dirpath)

		pattern_with_zeroes = f'%d{self.suffix}'
		if lzeroes >= 2:
			pattern_with_zeroes = f'%0{lzeroes}d{self.suffix}'

		name = self.dirpath
		vf = ''

		# Frame range
		# ----------------------------------------
		frameargs1 = ['-start_number', str(max(skip, self.f_first + skip))]
		frameargs2 = []
		lo = 0
		hi = 0
		if frames is not None:
			lo, hi, name = self.parse_frames(frames, name)
			print(f'Frame range: {lo} : {hi}')
			frameargs1 = ['-start_number', str(lo)]
			frameargs2 = ['-frames:v', str(hi - lo + 1)]

		# Music
		# ----------------------------------------
		if music_start is None:
			music_start = lo

		musicargs = []
		if music:
			music_start = f'{music_start / fps:0.2f}'
			musicargs = ['-ss', str(music_start), '-i', self.res(music).as_posix()]

		# VF filters
		# ----------------------------------------
		# Determine framecount from filename pattern
		framecount = self.f_last

		vf = ''

		if fade_in or fade_out:
			vf = vf_fade(vf, fade_in, fade_out, framecount, fps)
		vf, w, h = vf_rescale(vf, w, h, self.w, self.h)

		print(f"Making video at {w}x{h}")

		# Bitrate
		# ----------------------------------------
		if bv:
			bv = ['-b:v', bv]
		else:
			bv = []

		if ba:
			ba = ['-b:a', ba]
		else:
			ba = ''

		# Run
		# ----------------------------------------
		out = paths.sessions / f'{name}.mp4'
		pattern = self.dirpath / pattern_with_zeroes
		args = ['ffmpeg', '-y', *musicargs, '-r', str(fps), *frameargs1, '-i', pattern.as_posix(), *frameargs2, '-vf', vf, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', *bv, *ba, out.as_posix()]

		print('')
		print(' '.join(args))

		# Don't print output to console
		if bg:
			subprocess.Popen(args)
		else:
			subprocess.run(args)

		return out

	def make_archive(self, frames=None, archive_type='zip', bg=False):
		if self.processing_thread is not None:
			self.cancel_processing = True
			self.processing_thread.join()
			self.processing_thread = None

		zipfile = self.dirpath / 'frames.zip'
		self.processing_thread = threading.Thread(target=self._make_archive, args=(zipfile, frames, archive_type))
		self.processing_thread.start()

		if not bg:
			self.processing_thread.join()

	def _make_archive(self, zipfile, frames, archive_type):
		for f in self.dirpath.glob(f'*{self.suffix}'):
			if self.cancel_processing:
				break
			try:
				int(f.stem)
				ok = False
				if frames is None:
					ok = True
				else:
					lo, hi, n = self.parse_frames(frames, '')
					ok = lo <= int(f.stem) <= hi
				if ok:
					code = -1
					if archive_type == 'zip':
						code = os.system(f"zip -uj {zipfile} {f}  -1")
					elif archive_type == 'tar':
						code = os.system(f"tar -rf {zipfile} {f}")
					if code != 0:
						break

			except:
				pass
		self.processing_thread = None

	def make_rife_ncnn_vulkan(self, frames=None, name=None, scale=None, fps=None):
		"""
		This invokes the rife-ncnn-vulkan executable to interpolate frames in the current session folder.
		Linux only.
		"""

		# How much interpolation (2 == twice as many frames)
		if scale:
			RESOLUTION = scale
		elif fps:
			RESOLUTION = fps / self.fps
		else:
			RESOLUTION = 2

		lo, hi, name = self.parse_frames(frames, name or 'rife')
		ipath = self.dirpath
		dst = self.dirpath / name

		n_frames = int((hi - lo) * RESOLUTION)

		print(f"make_rife({ipath}, dst={dst}, lo={lo}, hi={hi})")

		# Run RIFE with tqdm progress bar
		# ----------------------------------------

		# start_num = len(list(dst.iterdir()))

		dst = paths.rmclean(dst)
		src = self.copy_frames('rife_src', frames, ipath)
		# src = ipath / 'rife_src'

		# proc = shlexproc(f'rife-ncnn-vulkan -i {src.as_posix()} -o {dst.as_posix()}')
		proc = shlexproc(f'rife-ncnn-vulkan -i {src.as_posix()} -o {dst.as_posix()} -j 3:3:3 -m rife-v4 -n {n_frames} -f jpg')

		paths.file_tqdm(dst,
		                start=0,
		                target=(hi - lo) * RESOLUTION,
		                process=proc,
		                desc=f"Running rife ...")
		paths.rmtree(src)
		paths.remap(dst, lambda num: num + lo * RESOLUTION - 1)

		return dst

	def copy_frames(self, name, frames=None, src=None):
		name = name
		lo, hi, name = self.parse_frames(frames, name)

		src = src or self.dirpath
		dst = self.res(name)

		if dst.exists():
			num_files = len(list(dst.iterdir()))
			if num_files == hi - lo:
				return dst

		paths.rmclean(dst)

		lead = hi - lo
		tq = tqdm(total=lead)
		tq.set_description(f"Copying frames to {dst.relative_to(self.dirpath)} ...")
		lz = max(len(str(lo)), len(str(hi)))
		for f in src.iterdir():
			if f.is_file():
				try:
					leadnum = int(f.stem)
				except:
					leadnum = -1

				if lo <= leadnum <= hi:
					shutil.copy(f, dst / f"{leadnum:0{lz}d}{self.suffix}")
					tq.update(1)

		# Finish the tq
		tq.update(lead - tq.n)
		tq.close()

		return dst

	def make_zpad(self, zeroes=None):
		"""
		Pad the frame numbers with 8 zeroes
		"""
		if zeroes is None:
			zeroes = paths.leadnum_zpad

		for f in self.dirpath.iterdir():
			if f.is_file():
				try:
					num = int(f.stem)
					newname = f"{num:0{zeroes}d}{self.suffix}"
					shutil.move(f, f.with_name(newname))
				except:
					pass

	def make_nopad(self):
		"""
		Remove leading zeroes from frame numbers
		"""
		for f in self.dirpath.iterdir():
			if f.is_file():
				try:
					num = int(f.stem)
					newname = f"{num}{self.suffix}"
					shutil.move(f, f.with_name(newname))
				except:
					pass

	def parse_frames(self, frames=None, name: str | Path = 'none'):
		if isinstance(name, Path):
			name = name.stem

		lo, hi, name = parse_frames(frames, name=name)

		if lo is None: lo = self.f_first
		if hi is None: hi = self.f_last
		if lo < self.f_first: lo = self.f_first
		if hi > self.f_last: hi = self.f_last

		# lead = get_max_leadnum(input)
		# lo = lo or 0
		# hi = hi or lead
		#
		return lo, hi, name

	def framerange(self):
		from jargs import args
		if args.frames:
			ranges = args.frames.split('-')
			for r in ranges:
				yield r
		else:
			yield self.f_first, self.f_last


def concat(s1, s2):
	if s1:
		s1 += ','
	return s1 + s2


def vf_rescale(vf, w, h, ow, oh, crop=False):
	"""
	Rescale by keeping aspect ratio.
	Args:
		vf:
		h: Target height (or none) can be a percent str or an int
		w: Target width (or none) can be a percent str or an int
		ow: original width (for AR calculation)
		oh: original height (for AR calculation)

	Returns:

	"""
	# if isinstance(h, str):
	#     h = int(h.replace('%', '')) / 100
	#     h = f'ih*{str(h)}'
	# if isinstance(w, str):
	#     w = int(w.replace('%', '')) / 100
	#     w = f'iw*{str(w)}'
	# if w is None:
	#     w = f'-1'
	# if h is None:
	#     h = f'-1'

	w = w or ow
	h = h or oh

	if not w: w = '-1'
	if not h: h = '-1'

	if not crop:
		if w > h:
			vf = concat(vf, f"scale={w}:-1")
		elif h > w:
			vf = concat(vf, f"scale=-1:{h}")
		else:
			vf = concat(vf, f"scale={w}:{h}")
	else:
		# vf = concat(vf, f"scale={w}:-1,pad={w}:{h}")
		ar = w / h
		vf = concat(vf, f"crop=iw*{ar}:ih")

	return vf, w, h


def vf_fade(vf, fade_in, fade_out, frames, fps):
	duration = frames / fps
	if fade_in < duration:
		vf = concat(vf, f'fade=in:st=0:d={fade_in}')
	if fade_out < duration:
		vf = concat(vf, f'fade=out:st={frames / fps - fade_out}:d={fade_out}')

	return vf


def get_video_fps(video_path):
	command = [
		'ffprobe',
		'-v', 'error',
		'-select_streams', 'v:0',
		'-show_entries', 'stream=avg_frame_rate',
		'-of', 'default=noprint_wrappers=1:nokey=1',
		video_path
	]

	result = subprocess.run(command, capture_output=True, text=True)
	fps = result.stdout.strip()

	if '/' in fps:
		fps = fps.split('/')
		fps = float(fps[0]) / float(fps[1])

	fps = int(fps)
	return fps
