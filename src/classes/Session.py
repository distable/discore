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
from . import convert, paths
from .convert import load_json, load_pil, save_json, save_png
from .logs import logsession, logsession_err
from .paths import get_leadnum, get_leadnum_zpad, get_max_leadnum, get_min_leadnum, is_leadnum_zpadded, leadnum_zpad, parse_frames
from src.lib.printlib import cputrace, trace, trace_decorator
from ..lib.corelib import shlexproc


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

        # Directory properties, cached for performance
        self.f = 1
        self.f_first = 1
        self.f_last = 1
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

    def load(self, *, log=True):
        """
        Load the session state from disk.
        If new frames have been added externally without interacting directly with a session object,
        this will update the session state to reflect the new frames.
        """
        if not self.dirpath.exists():
            return

        lo, hi = get_leadnum(self.dirpath)
        self.f_first = lo or 1
        self.f_last = hi or 1
        self.f_exists = False

        self.suffix = self.det_suffix()
        self.f_first_path = self.det_frame_path(self.f_first) or 1
        self.f_last_path = self.det_frame_path(self.f_last) or 1
        self.f_path = self.f_last_path
        self.f = self.f_last
        self.load_f()
        self.load_file()

        # Save the session data
        self.load_data()

        if log:
            if not any(self.dirpath.iterdir()):
                logsession(f"Loaded session {self.name} at {self.dirpath}")
            else:
                logsession(f"Loaded session {self.name} ({self.w}x{self.h}) at {self.dirpath} ({self.file})")

    def load_f(self, f=None, *, clamped_load=False):
        with trace("load_f"):
            f = f or self.f

            self.f = f
            self.f_path = self.f_last_path
            self.file = self.get_frame_name(self.f)
            self.f_exists = False
            if self.f <= self.f_last:
                self.f_exists = self.load_file()
            elif clamped_load:
                self.f_exists = self.load_file(self.f_last_path)

    def load_file(self, file: str | int | None = None):
        """
        Load a file, the current file in the session state, or a frame by number.
        Args:
            file: A path to a file

        Returns:

        """
        with trace("load_file"):
            if file is None:
                file = self.file
            if isinstance(file, str):
                file = Path(file)

            if isinstance(file, int):
                file = self.det_frame_path(file)
            elif isinstance(file, Path) and file.is_absolute():
                file = file
            elif isinstance(file, Path):
                file = self.dirpath / self.file

            if file.suffix in paths.image_exts:
                if file.exists():
                    self.img = file
                    return True

                return False


    def load_data(self):
        self.data = load_json(self.dirpath / "session.json", None)
        if self.data:
            self.data = Munch(self.data)
        else:
            self.data = Munch()

        self.fps = self.data.get("fps", self.fps)

    def save(self, path=None):
        if not path and self.file:
            path = self.res(self.file)
        if not path:
            path = self.f_path

        if not Path(path).is_absolute():
            path = self.dirpath / path

        save_num = paths.find_leadnum(path)
        if save_num is not None and save_num > self.f_last:
            self.f_last = save_num
            self.f_last_path = path
        if save_num is not None and save_num < self.f_first:
            self.f_first = save_num
            self.f_first_path = path

        path = Path(path)
        if self.img is not None:
            path = path.with_suffix(".png")
            import cv2
            save_png(self.img, path, with_async=False)

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
        self._img = convert.load_cv2(value, (self.w, self.h))

    @property
    def w(self):
        if self._img is None:
            return userconf.default_width
        return self.img.shape[1]

    @property
    def h(self):
        if self._img is None:
            return userconf.default_height
        return self.img.shape[0]


    def get_frame_name(self, f):
        return str(f).zfill(8) + self.suffix

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

    def get_frame_data(self, key, clamp=False):
        f = self.f
        if clamp and self.f > self.f_last:
            f = self.f_last

        f -= 1
        if key in self.data and f < len(self.data[key]):
            return self.data[key][f]

        return 0

    def seek(self, i=None, *, log=True):
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
        self.load_file()

        if log:
            logsession(f"({self.name}) seek({self.f})")

    def seek_min(self, *, log=True):
        if self.dirpath.exists() and any(self.dirpath.iterdir()):
            # Seek to next
            minlead = get_min_leadnum(self.dirpath)
            self.seek(minlead, log=log)

    def seek_max(self, *, log=True):
        if self.dirpath.exists() and any(self.dirpath.iterdir()):
            self.f = get_max_leadnum(self.dirpath)
            self.seek(self.f, log=log)

    def seek_next(self, i=1, log=True):
        self.f += i
        self.seek(self.f, log=log)

    def seek_new(self, log=True):
        self.seek(None, log=log)

    def subsession(self, name) -> "Session":
        if name:
            return Session(self.res(name))
        else:
            return self

    def res(self, subpath: Path | str, *, ext: str | list | tuple = None) -> Path:
        """
        Get a session resource, e.g. init video
        """
        subpath = Path(subpath)

        if subpath.is_absolute():
            return subpath
        elif isinstance(ext, (list, tuple)):
            # Try each extension until we find one that exists
            for e in ext:
                p = self.res(subpath, ext=e)
                if p.exists():
                    return p
            return None
        else:
            # Remove dot from extension
            if ext is not None and ext[0] == '.':
                ext = ext[1:]
            if subpath.suffix == '' and ext:
                subpath = subpath.with_suffix('.' + ext)

            ret = self.dirpath / subpath
            if not ret.exists():
                ret2 = self.dirpath.parent / subpath
                if ret2.exists():
                    return ret2
            return ret

    @trace_decorator
    def res_cv2(self, subpath: Path | str, *, ext: str = None, mode=None) -> np.ndarray:
        """
        Get a session resource, e.g. init video
        """
        if not isinstance(subpath, (Path, str)):
            return convert.load_cv2(subpath)

        p = self.res(subpath, ext=ext or ('jpg', 'png'))
        if p.exists():
            im = convert.load_cv2(p)
            if im is not None:
                if mode == 'fit':
                    im = convert.fit(im, self.w, self.h, 'black')
            return im
        else:
            return None

    def res_frame(self, resid, framenum=None, subdir='', ext=None, loop=False) -> Path | None:
        """
        Get a session resource, and automatically fetch a frame from it.
        Usage:

        resid='video.mp4:123' # Get frame 123 from video.mp4
        resid='video:3' # Get frame 3 from video.mp4
        resid='video' # Get the current session frame from video.mp4
        resid=3 # Get frame 3 from the current session
        """
        # If the resid is a number, assume it is a frame number
        if isinstance(resid, int):
            return self.det_frame_path(resid)
        elif resid is None:
            return Path(self.f_path)
        elif isinstance(resid, str):
            # If the resid is a string, parse it
            nameparts = resid.split(':')
            file = self.res(nameparts[0], ext=[*paths.video_exts, *paths.image_exts])
            if file is None:
                return None

            stem = Path(file.stem)  # The name of the resource with or without extension
            frame = self.f
            if framenum: frame = framenum
            if len(nameparts) > 1: frame = int(nameparts[-1])
            framedir = self.res(stem / subdir)

            # File exists and is not a video --> return directly / same behavior as res(...)
            if paths.is_image(file):
                return file
            elif paths.is_video(file):
                # Iterate dir and find the matching file, regardless of the extension
                if not framedir.is_dir():
                    self.extract_frames(stem)

                l = list(framedir.iterdir())
                if loop:
                    frame = frame % len(l)

                framestr = str(frame)
                return self.res(stem / f"{framestr.zfill(paths.leadnum_zpad)}.jpg")
        else:
            raise ValueError(f"Invalid resid: {resid}")
        # for file in l:
        #     if file.stem.lstrip('0') == framestr:
        #         if ext is None or file.suffix.lstrip('.') == ext.lstrip('.'):
        #             return file
        #
        # return None

    def res_frame_cv2(self, resid, framenum=None, subdir='', ext=None, loop=False):
        frame_path = self.res_frame(resid, framenum, subdir, ext, loop)
        if frame_path is None or not frame_path.exists():
            return np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # resize to fit
        ret = cv2.imread(str(frame_path))
        ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)
        if ret is not None and self.w and self.h:
            ret = cv2.resize(ret, (self.w, self.h))

        return ret

    def res_framepil(self, name, subdir='', ext=None, loop=False, ctxsize=False):
        ret = load_pil(self.res_frame(name, subdir, ext, loop))
        if ctxsize:
            ret = ret.resize((self.w, self.h))

        return ret

    def res_init(self, name=None):
        name = name or 'init'
        img = self.res(name, ext=paths.image_exts)
        mus = self.res(name, ext=paths.audio_exts)
        vid = self.res(name, ext=paths.video_exts)
        return img, mus, vid

    def res_music(self, name=None, *, optional=True):
        """
        Get the music resource for this session, or specify by name and auto-detect extension.
        """

        def _res_music(name, optional):
            name = name or 'music'
            if self.exists:
                file = self.res(f"{name}.mp3")
                if not file.exists(): file = self.res(f"{name}.ogg")
                if not file.exists(): file = self.res(f"{name}.wav")
                if not file.exists(): file = self.res(f"{name}.flac")
                if not file.exists() and not optional: raise FileNotFoundError("Could not find music file in session directory")
                return file

        if name:
            return _res_music(name, optional)
        else:
            v = _res_music('music', True)
            if not v.exists():
                v2 = _res_music('init', True)
                if v2.exists(): return v2
            if not optional:
                raise FileNotFoundError("Could not find music file in session directory")
            return v

    def res_script(self, name='script', touch=False):
        """
        Get the script resource for this session, or specify by name and auto-detect extension.
        """
        name = name or 'script'
        if not name.endswith('.py'):
            name += '.py'

        path = self.res(name)
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

    def extract_music(self, src='init', overwrite=False):
        input = self.res(src, ext="mp4")
        if not input.exists():
            print(f"Session.extract_music: Could not find video file {input.name}")
            return

        output = self.res(f"{src}.wav")
        cmd = f'ffmpeg -i {input} -acodec pcm_s16le -ac 1 -ar 44100 {output}'
        if output.exists():
            if not overwrite:
                print(f"Music extraction already exists for {input.name}, skipping ...")
                return
            paths.rm(input)

        os.system(cmd)

    def extract_frames(self, name, nth_frame=1, frames: tuple | None = None, w=None, h=None, overwrite=False) -> Path | str | None:
        src = self.res(name, ext='mp4')
        if not src.exists():
            print(f"Session.extract_frames: Could not find video file {src.name}")
            return

        lo, hi, name = self.parse_frames(frames, src.stem)

        vf = ""
        if frames is not None:
            vf += f'select=between(t\,{lo}\,{hi})'
        else:
            vf = f'select=not(mod(n\,{nth_frame}))'

        vf, w, h = vf_rescale(vf, self.w, self.h, self.w, self.h)

        if src.exists():
            dst = self.res(name)
            if dst.exists():
                if not overwrite:
                    logsession(f"Frame extraction already exists for {src.name}, skipping ...")
                    return dst
                else:
                    logsession(f"Frame extraction already exists for {src.name}, overwriting ...")

            paths.rmclean(dst)

            try:
                subprocess.run(['ffmpeg', '-i', f'{src}', '-vf', f'{vf}', '-q:v', '2', '-loglevel', 'error', '-stats', f'{dst}/%0{paths.leadnum_zpad}d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')
            except:
                subprocess.run(['ffmpeg.exe', '-i', f'{src}', '-vf', f'{vf}', '-q:v', '2', '-loglevel', 'error', '-stats', f'{dst}/%0{paths.leadnum_zpad}d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')

            return dst
        else:
            return None


    def make_video(self, fps=None, skip=3, bg=False, music='', music_start=None, frames=None, fade_in=.0, fade_out=.0, w=None, h=None, bv=None, ba='320k'):
        # call ffmpeg to create video from image sequence in session folder
        # do not halt, run in background as a new system process
        if fps is None:
            fps = self.fps

        # Detect how many leading zeroes are in the frame files
        lzeroes = get_leadnum_zpad(self.dirpath)

        pattern_with_zeroes = f'%d{self.suffix}'
        if lzeroes >= 2:
            pattern_with_zeroes = f'%0{lzeroes}d{self.suffix}'

        name = 'video'
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
        out = self.dirpath / f'{name}.mp4'
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

            except: pass
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

    def parse_frames(self, frames=None, name='none'):
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


def vf_rescale(vf, w, h, ow, oh):
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
    if isinstance(h, str):
        h = int(h.replace('%', '')) / 100
        h = f'ih*{str(h)}'
    if isinstance(w, str):
        w = int(w.replace('%', '')) / 100
        w = f'iw*{str(w)}'
    if w is None:
        w = f'-1'
    if h is None:
        h = f'-1'

    vf = concat(vf, f"scale={w}:{h}")

    # if h == '-1':
    #     ratio = w / ow
    #     h = int(oh * ratio)
    #     vf = concat(vf, f'scale={w}:{h}')
    # if w == '-1'
    #     ratio = h / oh
    #     w = int(ow * ratio)
    #     vf = concat(vf, f'scale={w}:{h}')
    # elif w is None and h is None:
    #     pass
    #     # print(f"Making video at {ow}x{oh}")

    return vf, w, h


def vf_fade(vf, fade_in, fade_out, frames, fps):
    duration = frames / fps
    if fade_in < duration:
        vf = concat(vf, f'fade=in:st=0:d={fade_in}')
    if fade_out < duration:
        vf = concat(vf, f'fade=out:st={frames / fps - fade_out}:d={fade_out}')

    return vf
