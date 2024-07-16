from src.classes import paths

# Constants
VAST_PYTHON_BIN = '/opt/conda/bin/python3'
VAST_PIP_BIN = '/opt/conda/bin/pip3'
VASTAI_DOCKER_IMAGE = 'pytorch/pytorch'
VASTAI_DISK_SPACE = '32'

# These packages will be installed when first connecting to the instance
APT_PACKAGES = [
    'python3-venv', 'libgl1', 'zip', 'ffmpeg', 'gcc', 'tree'
]

# Files to upload every time we connect
UPLOADS_ON_CONNECT = [
    ('requirements-vastai.txt', 'requirements.txt'),
    'discore.py',
    'deploy.py',
    'jargs.py',
    paths.userconf_name,
    paths.scripts_name,
    # paths.src_plugins_name,
    paths.src_name,
    (paths.root_comfy_nodes / 'ComfyUI-uiapi').relative_to(paths.root).as_posix(),
    (paths.root_comfy_nodes / 'ComfyUI-DownloadOnDemand').relative_to(paths.root).as_posix(),
    (paths.root_comfy / 'web/scripts/app.js').relative_to(paths.root).as_posix(),
    (paths.root_comfy / 'web/scripts/ui.js').relative_to(paths.root).as_posix(),
]

# Files to upload the first time we install the instance or when specified
SLOW_UPLOADS = [
    # paths.plug_res_name
]

DEPLOY_UPLOAD_BLACKLIST_PATHS = [
    "video.mp4", "video__*.mp4", "*.jpg", "__pycache__", "tmp"
]

# Download with RClone
DOWNLOAD_JOB_EXCLUSIONS = [
    "video.mp4", "video__*.mp4", "script.py", "*.npy", "__pycache__/*", "tmp/*"
]

# Upload with RClone
UPLOAD_JOB_PATHS = [
    paths.scripts, paths.code_core / 'party',
]

MODEL_URLS = [
    'https://civitai.com/models/129666?modelVersionId=356366'
]

enable_auto_connect = False
