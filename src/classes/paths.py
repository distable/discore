import math
import os
import re
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent  # TODO this isn't very robust

src_name = 'src'  # conflict with package names, must be underscore
src_plugins_name = 'src_plugins'  # conflict with package names, must be underscore
plug_res_name = 'plug-res'
plug_repos_name = 'plug_repos'

userconf_name = 'userconf.py'
scripts_name = 'scripts'
sessions_name = 'sessions'
tmp_name = 'tmp'


code_core = root / src_name  # Code for the core
code_plugins = root / src_plugins_name  # Downloaded plugin source code
plugins = root / 'src_plugins'  # Contains the user's downloaded plugins (cloned from github)
plug_res = root / plug_res_name  # Contains the resources for each plugin, categorized by plugin id
plug_logs = root / 'plug-logs'  # Contains the logs output by each plugin, categorized by plugin id
plug_repos = root / plug_repos_name  # Contains the repositories cloned by each plugin, categorized by plugin id
scripts = root / 'scripts'  # User project scripts to run
tmp = root / tmp_name  # Temporary files

gui_font = code_core / 'gui' / 'vt323.ttf'

# Image outputs are divied up into 'sessions'
# Session logic can be customized in different ways:
#   - One session per client connect
#   - Global session on a timeout
#   - Started manually by the user
sessions = root / sessions_name

session_timestamp_format = '%Y-%m-%d_%Hh%M'

plug_res.mkdir(exist_ok=True)
plug_logs.mkdir(exist_ok=True)
plug_repos.mkdir(exist_ok=True)
sessions.mkdir(exist_ok=True)

# These suffixes will be stripped from the plugin IDs for simplicity
plugin_suffixes = ['_plugin']

video_exts = ['.mp4', '.mov', '.avi', '.mkv']
image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
audio_exts = ['.wav', '.mp3', '.ogg', '.flac', '.aac']
text_exts = ['.py', '.txt', '.md', '.json', '.xml', '.html', '.css', '.js', '.csv', '.tsv', '.yml', '.yaml', '.ini']

audio_ext = '.ogg'
audio_codec = 'libopus'
leadnum_zpad = 8


# sys.path.insert(0, root.as_posix())
def is_image(file):
    return file.suffix in image_exts

def is_video(file):
    return file.suffix in video_exts

def short_pid(pid):
    """
    Convert 'user/my-repository' to 'my_repository'
    """
    if isinstance(pid, Path):
        pid = pid.as_posix()

    if '/' in pid:
        pid = pid.split('/')[-1]

    # Strip suffixes
    for suffix in plugin_suffixes:
        if pid.endswith(suffix):
            pid = pid[:-len(suffix)]

    # Replace all dashes with underscores
    pid = pid.replace('-', '_')

    return pid


def split_jid(jid, allow_jobonly=False):
    """
    Split a plugin jid into a tuple of (plug, job)
    """
    if '.' in jid:
        s = jid.split('.')
        return s[0], s[1]

    if allow_jobonly:
        return None, jid

    raise ValueError(f"Invalid plugin jid: {jid}")


def is_session(v):
    v = str(v)
    return sessions / v in sessions.iterdir()


def parse_frames(frames, name='none'):
    """
    parse_frames('example', '1:5') -> ('example_1_5', 1, 5)
    parse_frames('example', ':5') -> ('example_5', None, 5)
    parse_frames('example', '1:') -> ('example_1', 1, None)
    parse_frames("banner", None) -> ("banner", None, None)
    """
    if frames is not None:
        sides = frames.split(':')
        lo = sides[0]
        hi = sides[1]
        if frames.endswith(':'):
            lo = int(lo)
            return lo, None, f'{name}_{lo}'
        elif frames.startswith(':'):
            hi = int(hi)
            return None, hi, f'{name}_{hi}'
        else:
            lo = int(lo)
            hi = int(hi)
            return lo, hi, f'{name}_{lo}_{hi}'
    else:
        return None, None, name


# region Leadnums
def get_leadnum_zpad(path=None):
    """
    Find the amount of leading zeroes for the 'leading numbers' in the directory names and return it
    e.g.:
    0003123 -> 7
    00023 -> 5
    023 -> 3
    23 -> 2
    23_session -> 2
    48123_session -> 5
    """
    biggest = 0
    smallest = math.inf


    for file in Path(path).iterdir():
        if file.suffix in image_exts:
            match = re.match(r"^(\d+)\.", file.name)
            if match is not None:
                num = match.group(1)
                size = len(num)
                print(size, smallest, biggest, file)
                biggest = max(biggest, size)
                smallest = min(smallest, size)

    if smallest != biggest:
        return smallest
    return biggest


def is_leadnum_zpadded(path=None):
    return get_leadnum_zpad(path) >= 2


def get_next_leadnum(path=None):
    return (get_max_leadnum(path) or 0) + 1


def get_max_leadnum(path=None):
    lo, hi = get_leadnum(path)
    return hi


def get_min_leadnum(path=None):
    lo, hi = get_leadnum(path)
    return lo


def get_leadnum(path=None):
    """
    Find the largest 'leading number' in the directory names and return it
    e.g.:
    23_session
    24_session
    28_session
    23_session

    return value is 28
    """
    if isinstance(path, str):
        return find_leadnum(path)

    smallest = math.inf
    biggest = 0
    for parent, dirs, files in os.walk(path):
        for file in files:
            stem, suffix = os.path.splitext(file)
            if suffix in image_exts:
                try:
                    num = int(stem)
                    smallest = min(smallest, num)
                    biggest = max(biggest, num)
                except ValueError:
                    pass
        break

    if biggest == 0 and smallest == math.inf:
        return None, None

    return smallest, biggest


def find_leadnum(path=None, name=None):
    if name is None:
        name = Path(path).name

    # Find the first number
    match = re.match(r"^(\d+)", name)
    if match is not None:
        return int(match.group(1))

    return None


# endregion


# region Utilities
def get_image_glob(path):
    path = Path(path)
    l = []
    # Add files using image_exts and os.walk
    for parent, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] in image_exts:
                l.append(os.path.join(parent, file))
    return l


# endregion

def get_first_match(path, suffix, name=None):
    path = Path(path)
    if not path.exists():
        return None

    for p in path.iterdir():
        if suffix is not None and p.suffix == suffix: return p
        if name is not None and p.name == name: return p
        pass

    return None


# region Scripts
def script_exists(name):
    return get_script_file_path(name).exists()


def parse_action_script(s, default=None):
    """
    script:action
    """
    if s is None:
        return default, None

    v = s.split(':')

    action = v[0]
    script = None

    if len(v) > 1:
        action = v[1]
        script = v[0]

    return action, script


def get_script_file_path(name):
    return Path(scripts / name).with_suffix(".py")


def get_script_module_path(name=None):
    modpath = get_script_file_path(name)

    # Decide if the script is in the scripts folder or part of the session
    if scripts_name in modpath.parts:
        return f'{scripts_name}.{modpath.relative_to(scripts).with_suffix("").as_posix().replace("/", ".")}'
    elif sessions_name in modpath.parts:
        return f'{sessions_name}.{modpath.relative_to(sessions).with_suffix("").as_posix().replace("/", ".")}'

def get_script_paths():
    # Iterate with os.walk
    for root, dirs, files in os.walk(scripts):
        files = sorted(files, key=len)
        if 'libs' not in root:
            for file in files:
                file = Path(file)
                if file.endswith(".py") and not file.startswith("__"):
                    yield os.path.join(root, file)

# endregion
def rmtree(path):
    """
    Remove a directory and all its contents
    """
    path = Path(path)
    if path.exists():
        import shutil
        shutil.rmtree(path)


def remap(dst, fn):
    for f in dst.iterdir():
        if f.is_file():
            try:
                num = int(f.stem)
                f.rename(dst / f"{int(fn(num))}.png")
            except:
                pass


def mktree(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

def rmclean(path):
    """
    Remove a directory and all its contents, and then recreate it clean
    """
    path = Path(path)
    if path.exists():
        import shutil
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    return path

def file_tqdm(path, start, target, process, desc='Processing'):
    from tqdm import tqdm
    import time

    tq = tqdm(total=target)
    tq.set_description(desc)
    last_num = start
    while process.poll() is None:
        cur_num = len(list(path.iterdir()))
        diff = cur_num - last_num
        if diff > 0:
            tq.update(diff)
            last_num = cur_num

        tq.refresh()
        time.sleep(1)

    tq.update(target - tq.n)  # Finish the bar
    tq.close()


def touch(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch(exist_ok=True)


def rm(path):
    path = Path(path)
    if path.exists():
        path.unlink()

def cp(path1, path2):
    path1 = Path(path1)
    path2 = Path(path2)
    if path1.exists():
        shutil.copy(path1, path2)


def exists(dat):
    return Path(dat).exists()

def iter_scripts():
    # Iterate with os.walk
    for parent, dirs, files in os.walk(scripts):
        files = sorted(files, key=len)
        if 'libs' not in parent:
            for file in files:
                file = str(file)
                if file.endswith(".py") and not file.startswith("__"):
                    # Print the relative path to parent without extension
                    yield os.path.relpath(os.path.join(parent, file), scripts)[:-3], os.path.join(parent, file)


# def convert_to_wav():
#     if path.name == 'music.mp3' and jargs.args.remote:
#         # Convert to WAV or use existing (and delete mp3 always)
#         mp3path = path.with_suffix('.mp3')
#         wavpath = path.with_suffix('.wav')
#
#         if wavpath.exists():
#             mp3path.unlink()
#             path = wavpath
#         else:
#             print(f"Converting {mp3path} to wav ...")
#             shlexrun(f'ffmpeg -i {mp3path} -acodec pcm_s16le -ac 2 -ar 44100 -b:v 224k {session.res("music.wav")}')
#             mp3path.unlink()
#             path = session.res('music.wav')
