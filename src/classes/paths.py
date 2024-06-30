import math
import os
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import userconf
from src.classes import folder_paths
from src.lib import loglib

src_name = 'src'  # conflict with package names, must be underscore
src_plugins_name = 'src_plugins'  # conflict with package names, must be underscore
plug_res_name = 'plug-res'
plug_repos_name = 'plug_repos'

userconf_name = 'userconf.py'
scripts_name = 'scripts'
sessions_name = 'sessions'
tmp_name = 'tmp'

# Root and subdirs
# ----------------------------------------
root = Path(__file__).resolve().parent.parent.parent  # TODO this isn't very robust
code_core = root / src_name  # Code for the core
code_plugins = root / src_plugins_name  # Downloaded plugin source code
plugins = root / 'src_plugins'  # Contains the user's downloaded plugins (cloned from github)
plug_res = root / plug_res_name  # Contains the resources for each plugin, categorized by plugin id
plug_logs = root / 'plug-logs'  # Contains the logs output by each plugin, categorized by plugin id
plug_repos = root / plug_repos_name  # Contains the repositories cloned by each plugin, categorized by plugin id
scripts = root / 'scripts'  # User project scripts to run
template_scripts = root / 'scripts' / 'templates'  # User project scripts to run
tmp = root / tmp_name  # Temporary files
hobo_icon = root / src_name / 'gui' / 'icon.png'

# special files (in root)
openai_api_key = root / '.openai_api_key'
gui_font = root / src_name / 'gui' / 'vt323.ttf'

# Other paths
# ----------------------------------------
root_models = Path("D:/ai-models/")

# Image outputs are divied up into 'sessions'
# Session logic can be customized in different ways:
#   - One session per client connect
#   - Global session on a timeout
#   - Started manually by the user
sessions = root / sessions_name

init_paths = [
    sessions,
    sessions / '.inits'
]

session_timestamp_format = '%Y-%m-%d_%Hh%M'

plug_res.mkdir(exist_ok=True)
plug_logs.mkdir(exist_ok=True)
plug_repos.mkdir(exist_ok=True)
sessions.mkdir(exist_ok=True)

# These suffixes will be stripped from the plugin IDs for simplicity
plugin_suffixes = ['_plugin']

video_exts = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
audio_exts = ['.wav', '.mp3', '.ogg', '.flac', '.aac']
text_exts = ['.py', '.txt', '.md', '.json', '.xml', '.html', '.css', '.js', '.csv', '.tsv', '.yml', '.yaml', '.ini']

audio_ext = '.flac'
audio_codec = 'flac'
leadnum_zpad = 8

log: callable = loglib.make_log('paths')
logerr: callable = loglib.make_logerr('paths')

extra_model_paths = [
    '/media/data-team/Projects/ComfyUI_windows_portable/ComfyUI/extra_model_paths.yaml',
    r'D:\Projects\ComfyUI_windows_portable\ComfyUI\extra_model_paths.yaml'
]


# sys.path.insert(0, root.as_posix())
def is_image(file):
    file = Path(file)
    return file.suffix in image_exts


def is_video(file):
    file = Path(file)
    return file.suffix in video_exts


def is_audio(file):
    file = Path(file)
    return file.suffix in audio_exts


def is_youtube(url):
    url = str(url)
    return 'youtube.com' in url or 'youtu.be' in url


def is_timestamp_range(string):
    # Match a timestamp range such as 00:33-00:45
    return re.match(r'^\d{1,2}:\d{2}-\d{1,2}:\d{2}$', string) is not None


def is_temp_extraction_path(path):
    return path.name.startswith('.')


def get_cache_file_path(filepath, cachename, ext='.npy'):
    filepath = Path(filepath)

    # If the file is in a hidden folder, we save the cache in the parent folder
    if filepath.parent.stem.startswith('.'):
        filepath = filepath.parent.parent / filepath.name

    # If the data would be hidden, we unhide. Cached audio data should never be hidden!
    if filepath.name.startswith('.'):
        filepath = filepath.with_name(filepath.name[1:])

    cachepath = filepath.with_stem(f"{Path(filepath).stem}_{cachename}").with_suffix(ext)
    return cachepath


def get_flow_compute_dirpath(resname):
    from src import renderer
    resdir = renderer.session.res_frame_dirpath(resname)
    flowdir = unhide(resdir.parent / f"{resdir.name}.flow/")
    flowdir = hide(flowdir)
    return flowdir


def get_timestamp_range(string):
    """
    Match a timestamp range such as 00:33-00:45 and return min/max seconds (two values)
    """
    if is_timestamp_range(string):
        lo, hi = string.split('-')
        lo = parse_time_to_seconds(lo)
        hi = parse_time_to_seconds(hi)
        return lo, hi
    else:
        raise ValueError(f"Invalid timestamp range: {string}")


def parse_time_to_seconds(time_str):
    def try_parse(fmt):
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError or BaseException:
            return None

    # Parse time string to datetime object
    time_obj = try_parse('%H:%M:%S.%f') or \
               try_parse('%H:%M:%S') or \
               try_parse('%M:%S.%f') or \
               try_parse('%M:%S') or \
               try_parse('%S.%f') or \
               try_parse('%S')

    # Convert datetime object to timedelta object
    time_delta = timedelta(hours=time_obj.hour,
                           minutes=time_obj.minute,
                           seconds=time_obj.second,
                           microseconds=time_obj.microsecond)
    # Return total seconds
    return time_delta.total_seconds()


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


def parse_frame_string(frames, name='none', min=None, max=None):
    """
    parse_frames('1:5', 'example') -> ('example_1_5', 1, 5)
    parse_frames(':5', 'example') -> ('example_5', None, 5)
    parse_frames('3:', 'example') -> ('example_3', 3, None)
    parse_frames(None, 'banner') -> ("banner", None, None)
    """
    if isinstance(frames, tuple):
        return *frames[0:2], name

    if frames is not None:
        sides = frames.split(':')
        lo = sides[0]
        hi = sides[1]
        if frames.endswith(':'):
            lo = int(lo)
            return lo, max, f'{name}_{lo}_{max or "max"}'
        elif frames.startswith(':'):
            hi = int(hi)
            return min, hi, f'{name}_0_{hi}'
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
                # print(size, smallest, biggest, file)
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


def get_latest_session():
    """
    Return the most recently modified session directory (on the session.json file inside)
    If no session is found, return None
    """
    latest = None
    latest_mtime = 0

    baselevel = len(str(sessions).split(os.path.sep))
    # if curlevel <= baselevel + 1:
    #     [do stuff]

    for parent, dirs, files in os.walk(sessions):
        curlevel = len(str(parent).split(os.path.sep))
        if 'session.json' in files and curlevel == baselevel + 1:
            mtime = os.path.getmtime(os.path.join(parent, 'session.json'))
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest = parent

    return latest, latest_mtime


def get_script_file_path(name):
    return Path(scripts / name).with_suffix(".py")


def get_script_module_path(name=None):
    return filepath_to_modpath(get_script_file_path(name))


def filepath_to_modpath(filepath):
    filepath = Path(filepath)
    if filepath.suffix == '.py':
        filepath = filepath.with_suffix('')

    filepath = filepath.relative_to(root)
    # TODO check if root is in filepath

    # modpath = get_script_file_path(name)
    #
    # # Decide if the script is in the scripts folder or part of the session
    # if scripts_name in modpath.parts:
    #     return f'{scripts_name}.{modpath.relative_to(scripts).with_suffix("").as_posix().replace("/", ".")}'
    # elif sessions_name in modpath.parts:
    #     return f'{sessions_name}.{modpath.relative_to(sessions).with_suffix("").as_posix().replace("/", ".")}'

    ret = filepath.as_posix().replace('/', '.')
    if ret.endswith("."): ret = ret[:-1]
    if ret.startswith("."): ret = ret[1:]

    if '..' in ret:  # invalid path, inside a .folder
        return None

    return ret


def get_script_paths():
    # Iterate with os.walk
    for root, dirs, files in os.walk(scripts):
        files = sorted(files, key=len)
        if 'libs' not in root:
            for file in files:
                file = Path(file)
                if str(file).endswith(".py") and not str(file).startswith("__"):
                    yield os.path.join(root, file)


# endregion

def guess_suffix(suffixless_path):
    # Convert the input to a Path object
    path = Path(suffixless_path)

    if path.suffix != "":
        return path

    # Check if the suffixless path already exists
    if path.exists():
        return path

    # Get the directory and stem from the suffixless path
    directory = path.parent
    stem = path.stem

    # Iterate over the files in the directory
    for file_path in directory.iterdir():
        file_stem = file_path.stem

        # Check if the file has the same stem as the suffixless path
        if file_stem == stem:
            return file_path

    # No matching file found, return None
    # printerr(f"Couldn't guess extension: {suffixless_path}")
    return path


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
    if path.suffix:
        path = path.parent
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


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


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

def get_first_exist(*paths):
    for p in paths:
        if p and p.exists():
            return p

    return paths[-1]


def unhide(flowdir):
    flowdir = Path(flowdir)
    if flowdir.name.startswith('.'):
        flowdir = flowdir.with_name(flowdir.name[1:])
    return flowdir


def hide(flowdir):
    flowdir = Path(flowdir)
    if not flowdir.name.startswith('.'):
        flowdir = flowdir.with_name(f'.{flowdir.name}')
    return flowdir


def seconds_to_ffmpeg_timecode(param):
    # Convert absolute seconds (e.g. 342) to ffmpeg timecode (e.g. 00:05:35.00) string
    return str(timedelta(seconds=param))


def iter_session_paths(update_cache=False):
    from src.classes import storage
    if update_cache and storage.has('cached_sessions'):
        return storage.get_paths('cached_sessions')

    files = list(sessions.iterdir())

    # Explore root paths recursively and add folders that contain a script.py
    def explore(root):
        if not root.exists():
            return

        for file in root.iterdir():
            is_dir = file.is_dir()
            has_dir_script = (file / 'script.py').exists()
            is_image_file = is_image(file)
            is_numbered_stem = file.stem.isdigit()

            if is_dir and has_dir_script:
                print(f"Found session: {file}")
                files.append(file)
                return
            elif is_image_file and is_numbered_stem:
                print(f"Found session: {file.parent}")
                return
            elif is_dir:
                explore(file)

    for path in userconf.session_paths:
        explore(Path(path))

    files.sort(key=lambda x: x.stat().st_mtime, reverse=False)
    files = [f for f in files if f.is_dir()]

    storage.set_paths('cached_sessions', files)
    return files


def is_init(name_or_path):
    path = Path(name_or_path)
    return path.name.startswith('.init') or path.name.startswith('init')


def store_last_session_name(name):
    """
    Write the session name to root/last_session.txt
    """
    with open(root / 'last_session.txt', 'w') as f:
        f.write(name)


def fetch_last_session_name():
    """
    Read the session name from root/last_session.txt
    """
    try:
        with open(root / 'last_session.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return None


def has_last_session():
    return Path(root / 'last_session.txt').exists()


def parse_extra_model_paths(print_path):
    for yaml_path in extra_model_paths:
        load_extra_path_config(yaml_path, print_path)


def load_extra_path_config(yaml_path, print_path=False):
    if not Path(yaml_path).exists():
        return

    with open(yaml_path, 'r') as stream:
        import yaml
        config = yaml.safe_load(stream)
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        base_path = None
        if "base_path" in conf:
            base_path = conf.pop("base_path")
        for x in conf:
            for y in conf[x].split("\n"):
                if len(y) == 0:
                    continue
                full_path = y
                if base_path is not None:
                    full_path = os.path.join(base_path, full_path)
                if print_path: log("Adding extra search path", x, full_path)
                folder_paths.add_model_folder_path(x, full_path)


def get_model_path(ckpt_name, required=False):
    """
    Get a SD model full path from its name, searching all the defined model locations
    """
    ret = folder_paths.get_full_path('checkpoints', ckpt_name)
    if ret is not None:
        return ret

    if required:
        raise ValueError(f"Model {ckpt_name} not found")
    return None


def get_controlnet_path(ckpt_name, required=False):
    """
    Get a controlnet model full path from its name, searching all the defined model locations
    """
    ret = folder_paths.get_full_path('controlnet', ckpt_name)
    if ret is not None:
        return ret

    if required:
        raise ValueError(f"Controlnet model {ckpt_name} not found")
    return None


parse_extra_model_paths(False)
