import argparse
import sys


def int_or_str(value):
    try:
        # Try converting to an integer
        return int(value)
    except ValueError:
        # If it fails, treat it as a string
        return value

argp = argparse.ArgumentParser()

# Add positional argument 'project' for the script to run. optional, 'project' by default
argp.add_argument("session", nargs="?", default=None, help="Session or script")
argp.add_argument("action", nargs="?", default=None, help="Script or action to run")
argp.add_argument("subdir", nargs="?", default='', help="Subdir in the session")

argp.add_argument('--frames', '-f', type=str, default=None, help='The frames to render in first:last,first:last,... format')
argp.add_argument('--init', type=str, default=None, help='Specify init medias paths separated by semicolon. Can be youtube links or local files. If passing timestamps such as 00:00-00:10, the previous init data will be trimmed to that range.')
argp.add_argument('--script', type=str, default=None, help='Specify script to use, can also be a session name to copy from.')

argp.add_argument('-start', nargs='?', const=True, default=None, type=int_or_str, help='Immediately start rendering when the GUI is ready, can also specify a frame to start at. (-start 100)')
# argp.add_argument('-start', action='store_true', help='Immediately start rendering when the GUI is ready.')
argp.add_argument('-cli', action='store_true', help='Run the renderer in CLI mode. (no gui)')
argp.add_argument('-opt', action='store_true', help='Run in optimized mode, enables various VRAM optimizations.')
argp.add_argument('-ro', '--readonly', action='store_true', help='Run the renderer in read-only mode. (for viewing directories with auto-refresh)')
argp.add_argument("-dry", action="store_true")

argp.add_argument('--run', action='store_true', help='Perform the run in a subprocess')
argp.add_argument('--remote', action='store_true', help='Indicates that we are running on a remote server, used to enable more expensive computations or parameters.')
argp.add_argument('--print', action='store_true', help='Enable printing.')
argp.add_argument('--dev', action='store_true', help='Enable dev mode on startup.')
argp.add_argument('--trace', action='store_true', help='Enable tracing.')
argp.add_argument('--trace-gpu', action='store_true', help='Enable tracing of models and VRAM.')
argp.add_argument('--unsafe', action='store_true', help='Don\" catch exceptions in renderer, allow them to blow up the program.')
argp.add_argument('--profile', action='store_true', help='Enable profiling.')
argp.add_argument('--profile_jobs', action='store_true', help='Profile each job one by one.')
argp.add_argument('--profile_session_run', action='store_true', help='Profile session.run')
argp.add_argument('--profile_session_load', action='store_true', help='Profile session.load')
argp.add_argument('--profile_run_job', action='store_true', help='Profile jobs.run')
argp.add_argument("--recreate_venv", action="store_true")
argp.add_argument("--no_venv", action="store_true")
argp.add_argument('--upgrade', action='store_true', help='Upgrade to latest version')
argp.add_argument('--install', action='store_true', help='Install plugins requirements and custom installations.')
# argp.add_argument('--python_exec', type='str', default='python3', action='store_true', help='Specify the python binary to run in commands.')

argp.add_argument("--newplug", action="store_true", help="Create a new plugin with the plugin wizard")

# Renderer arguments
argp.add_argument('--draft', type=int, default=0, help='Cut down the resolution to render a draft quickly.')
argp.add_argument('--zip_every', type=int, default=None, help='Create a zip of the frames every specified number of frames.')
argp.add_argument('--preview_every', type=int, default=None, help='Create a preview video every number of frames. (with ffmpeg)')
argp.add_argument('--preview_command', type=str, default='', help='The default ffmpeg command to use for preview videos.')

# Deployment
argp.add_argument('--shell', action='store_true', default=None, help='Open a shell in the deployed remote.')
argp.add_argument('--local', action='store_true', help='Deploy locally. (test)')
argp.add_argument('-vai', '--vastai', action='store_true', help='Deploy to VastAI or continue the existing deploy.')
argp.add_argument('-vaig', '--vastai_gui', action='store_true', help='Open the deployment gui.')
argp.add_argument('-vaiq', '--vastai_quick', action='store_true', help='Continue a previous deployment without any copying')
argp.add_argument('-svai', '--vastai_stop', action='store_true', help='Stop the VastAI instance after running.')
argp.add_argument('-rvai', '--vastai_reboot', action='store_true', help='Reboot the VastAI instance before running.')
argp.add_argument('-dvai', '--vastai_delete', action='store_true', help='Delete the VastAI instance after running.')
argp.add_argument('-lsvai', '--vastai_list', action='store_true', help='List the running VastAI instances.')
argp.add_argument('-vaiu', '--vastai_upgrade', action='store_true', help='Upgrade the VastAI environment. (pip installs and plugins)')
argp.add_argument('-vaii', '--vastai_install', action='store_true', help='Upgrade the VastAI environment. (install plugins)')
argp.add_argument('-vaicp', '--vastai_copy', action='store_true', help='Copy files even with vastai_quick')
argp.add_argument('-vais', '--vastai_search', type=str, default=None, help='Search for a VastAI server')
argp.add_argument('-vaird', '--vastai_redeploy', action='store_true', help='Delete the Discore installation and start over. (mostly used for Discore development)')
argp.add_argument('-vait', '--vastai_trace', action='store_true', help='Trace on the VastAI discore execution.')
argp.add_argument('-vaish', '--vastai_shell', action='store_true', help='Only start a shell on the VastAI instance.')
argp.add_argument('-vaify', '--vastai_comfy', action='store_true', help='Only start comfyui on the VastAI instance.')
argp.add_argument('--vastai_no_download', action='store_true', help='Prevent downloading during copy step.')

argv = sys.argv[1:]
args = argp.parse_known_args()
argvr = args[1]
args = args[0]

spaced_args = ' '.join([f'"{arg}"' for arg in argv])

# Eat up arguments
sys.argv = [sys.argv[0]]

is_vastai = args.vastai or \
            args.vastai_gui or \
            args.vastai_upgrade or \
            args.vastai_install or \
            args.vastai_redeploy or \
            args.vastai_quick or \
            args.vastai_copy or \
            args.vastai_search or \
            args.vastai_delete or \
            args.vastai_reboot or \
            args.vastai_list or \
            args.vastai_trace or \
            args.vastai_shell or \
            args.vastai_comfy
is_vastai_continue = args.vastai or args.vastai_quick

def is_gui():
    return not args.cli

def get_discore_session(load=True, *, nosubdir=False):
    from src.classes.Session import Session

    if not args.session:
        return None

    s = Session(args.session or args.action or args.script, load=load)
    if not nosubdir:
        s = s.subsession(args.subdir)

    return s


def get_frameranges():
    if args.frames:
        ranges = args.frames.split(':')
        for r in ranges:
            yield r
    else:
        yield args.frames


def safe_list_remove(l, value):
    if not l: return
    try:
        l.remove(value)
    except:
        pass


def remove_deploy_args(oargs):
    safe_list_remove(oargs, '--dev')
    safe_list_remove(oargs, '--run')

    # Remove all that start with vastai or vai
    for prefixed_arg in oargs:
        # Remove dashes
        arg = prefixed_arg.replace('-', '')
        if arg.startswith('vastai') or arg.startswith('vai') or arg.endswith('vai'):
            safe_list_remove(oargs, prefixed_arg)

    return oargs

    # safe_list_remove(oargs, '--vastai')
    # safe_list_remove(oargs, '-vai')
    # safe_list_remove(oargs, '-vaird')
    # safe_list_remove(oargs, '-vaiq')
    # safe_list_remove(oargs, '-vais')
    # safe_list_remove(oargs, '-vaiu')
    # safe_list_remove(oargs, '-vaii')
    # safe_list_remove(oargs, '--vastai_upgrade')
    # safe_list_remove(oargs, '--vastai_install')
    # safe_list_remove(oargs, '--vastai_redeploy')
    # safe_list_remove(oargs, '--vastai_quick')
