import sys

from src.classes import paths

# Constants
VAST_AI_PYTHON_BIN = '/opt/conda/bin/python3' if not sys.platform.startswith('win') else 'python'

DISK_SPACE = '32'

DOCKER_IMAGE = 'pytorch/pytorch'

# These packages will be installed when first connecting to the instance
APT_PACKAGES = [
    'python3-venv', 'libgl1', 'zip', 'ffmpeg', 'gcc'
]

# Files to upload every time we connect
FAST_UPLOADS = [
    ('requirements-vastai.txt', 'requirements.txt'),
    'discore.py', 'deploy.py', 'jargs.py', paths.userconf_name,
    paths.scripts_name, paths.src_plugins_name, paths.src_name,
]

# Files to upload the first time we install the instance or when specified
SLOW_UPLOADS = [paths.plug_res_name]

DEPLOY_UPLOAD_BLACKLIST_PATHS = [
    "video.mp4", "video__*.mp4", "*.jpg", "__pycache__", "tmp"
]

RCLONE_JOB_DOWNLOAD_EXCLUSION = [
    "video.mp4", "video__*.mp4", "script.py", "*.npy", "__pycache__/*", "tmp/*"
]

RCLONE_JOB_UPLOAD_PATHS = [
    paths.scripts, paths.code_core / 'party',
]

MODEL_URLS = [
    'https://civitai.com/models/129666?modelVersionId=356366'
]
