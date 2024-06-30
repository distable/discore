print("Importing libraries")


import logging
import os
import shutil
import sys
from pathlib import Path
import jargs
from jargs import argp, args, spaced_args
from src.lib import corelib
from src.classes import paths

# Constants
DEFAULT_ACTION = 'r'
VENV_DIR = "venv"
PYTHON_EXEC = "python3"


def on_ctrl_c():
    from src.classes.logs import logdiscore
    from src import renderer

    logdiscore("Exiting because of Ctrl+C.")
    renderer.requests.stop = True
    exit(0)


def setup_environment():
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(logging.ERROR)
    os.chdir(Path(__file__).parent)

    global PYTHON_EXEC
    if jargs.args.remote:
        PYTHON_EXEC = sys.executable


def check_requirements():
    if os.name == 'posix' and os.geteuid() == 0:
        print("You are running as root, proceed at your own risks")

    if sys.version_info < (3, 9):
        print(f"Warning: Your Python version {sys.version_info} is lower than 3.9. You may encounter issues.")

    if not corelib.has_exe('git'):
        print("Please install git")
        sys.exit(1)


def setup_virtual_environment():
    if not args.no_venv:
        if not os.path.exists(VENV_DIR) or args.recreate_venv:
            if args.recreate_venv:
                shutil.rmtree(VENV_DIR)
            os.system(f"{PYTHON_EXEC} -m venv {VENV_DIR}")
            argp.upgrade = True
        os.system(f"bash -c 'source {VENV_DIR}/bin/activate'")


def upgrade_requirements():
    if args.no_venv:
        os.system(f"{PYTHON_EXEC} -m pip install -r requirements.txt")
    else:
        os.system(f"{VENV_DIR}/bin/pip install -r requirements.txt")
    print('----------------------------------------')
    print("\n\n")
    sys.exit(0)


def run_script():
    cmd = f"bash -c '"
    if not args.no_venv:
        cmd += f"source {VENV_DIR}/bin/activate && "
    cmd += f"{PYTHON_EXEC} {__file__} {spaced_args} --upgrade --run'"
    os.system(cmd)


def handle_default():
    if args.dry:
        print("Exiting because of -dry argument")
        sys.exit(0)

    # import importlib
    from old import server
    server.run()


def handle_action(action):
    from src.classes.logs import logdiscore_err
    from src.lib.loglib import print_possible_scripts

    apath = paths.get_script_file_path(action)
    if not apath.is_file():
        logdiscore_err(f"Unknown action '{args.action}' (searched at {apath})")
        print_possible_scripts()
        sys.exit(1)

    # import importlib
    import importlib
    amod = importlib.import_module(paths.get_script_module_path(action), package=action)
    if amod is None:
        logdiscore_err(f"Couldn't load '{args.action}'")
        print_possible_scripts()
        sys.exit(1)

    amod.action(args)
    print("Action done.")


def main():
    from src.classes.logs import logdiscore_err, logdiscore
    from yachalk import chalk
    from src.lib.loglib import print_existing_sessions, print_possible_scripts

    if args.newplug:
        plugin_wizard()
        return

    if args.local:
        from deploy import deploy_local
        deploy_local()
    elif jargs.is_vastai:
        from deploy import deploy_vastai
        deploy_vastai()
    else:
        from src.classes import common
        common.setup_ctrl_c(on_ctrl_c)

        if args.session == 'help' and args.action is None:
            print("Sessions:")
            print_existing_sessions()
            print("\nScripts:")
            print_possible_scripts()
            sys.exit(0)

        from src.classes.paths import parse_action_script
        action, script = parse_action_script(args.action, DEFAULT_ACTION)

        logdiscore(chalk.green(f"action: {action}"))
        logdiscore(chalk.green(f"script: {script}"))

        if action is not None:
            if action == 'help':
                print_possible_scripts()
                sys.exit(0)

            handle_action(action)
        else:
            handle_default()


if __name__ == '__main__':
    corelib.setup_annoying_logging()
    setup_environment()

    if not args.run:
        check_requirements()
        setup_virtual_environment()
        if args.upgrade:
            upgrade_requirements()
        run_script()
        sys.exit(0)

    main()

# def install_core():
#     """
# Install all core requirements
# """
#     from src.installer import check_import
#     from src.installer import python
#     from src.installer import pipargs
#     from src.lib.loglib import printerr
#
#     torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113")
#     clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
#     requirements_file = os.environ.get('REQS_FILE', "../requirements_versions.txt")
#
#     paths.plug_repos.mkdir(exist_ok=True)
#     paths.plug_logs.mkdir(exist_ok=True)
#     paths.plug_res.mkdir(exist_ok=True)
#     paths.plugins.mkdir(exist_ok=True)
#     paths.sessions.mkdir(exist_ok=True)
#
#     if not check_import("torch") or not check_import("torchvision"):
#         from src import installer
#         installer.run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch")
#
#     try:
#         import torch
#         assert torch.cuda.is_available()
#     except:
#         printerr('Torch is not able to use GPU')
#         sys.exit(1)
#
#     if not check_import("clip"):
#         pipargs(f"install {clip_package}", "clip")


# def plugin_wizard():
#     import shutil
#     from src import installer
#     import re
#     from src.classes import paths
#     from art import text2art
#
#     PLUGIN_TEMPLATE_PREFIX = ".template"
#
#     print(text2art("Plugin Wizard"))
#
#     # Find template directories (start with .template)
#     templates = []
#     for d in paths.code_plugins.iterdir():
#         if d.name.startswith(PLUGIN_TEMPLATE_PREFIX):
#             templates.append(d)
#
#     template = None
#
#     if len(templates) == 0:
#         print("No templates found.")
#         exit(1)
#     elif len(templates) == 1:
#         template = templates[0]
#     else:
#         for i, path in enumerate(templates):
#             s = path.name[len(PLUGIN_TEMPLATE_PREFIX):]
#             while not s[0].isdigit() and not s[0].isalpha():
#                 s = s[1:]
#             print(f"{i + 1}. {s}")
#
#         print()
#         while template is None:
#             try:
#                 v = int(input("Select a template: ")) - 1
#                 if v >= 0:
#                     template = templates[v]
#             except:
#                 pass
#
#     pid = input("ID name: ")
#
#     clsdefault = f"{pid.capitalize()}Plugin"
#     cls = input(f"Class name (default={clsdefault}): ")
#     if not cls:
#         cls = clsdefault
#
#     plugdir = paths.code_plugins / f"{pid}_plugin"
#     clsfile = plugdir / f"{cls}.py"
#
#     shutil.copytree(template.as_posix(), plugdir)
#     shutil.move(plugdir / "TemplatePlugin.py", clsfile)
#
#     # Find all {{word}} with regex and ask for a replacement
#     regex = r'__(\w+)__'
#     with open(clsfile, "r") as f:
#         lines = f.readlines()
#     for i, line in enumerate(lines):
#         matches = re.findall(regex, line)
#         if matches:
#             vname = matches[0]
#
#             # Default values
#             vdefault = ''
#             if vname == 'classname': vdefault = cls
#             if vname == 'title': vdefault = pid
#
#             # Ask for a value
#             if vdefault:
#                 value = input(f"{vname} (default={vdefault}): ")
#             else:
#                 value = input(f"{vname}: ")
#
#             if not value and vdefault:
#                 value = vdefault
#
#             # Apply the value
#             lines[i] = re.sub(regex, value, line)
#
#     # Write lines back to file
#     with open(clsfile, "w") as f:
#         f.writelines(lines)
#
#     # Open plugdir in the file explorer
#     installer.open_explorer(plugdir)
#     print("Done!")
#     input()
#     exit(0)
