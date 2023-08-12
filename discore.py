#!/usr/bin/python3


import importlib
import logging
import os
import shutil
import sys
from pathlib import Path

# print(' '.join(sys.argv[1:]))

import jargs
from jargs import argp, args, spaced_args

# sys.path.append(Path(__file__).parent.as_posix())
# sys.path.append((Path(__file__).parent / 'src').as_posix())

logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)

os.chdir(Path(__file__).parent)

DEFAULT_ACTION = 'r'
VENV_DIR = "venv"
python_exec = "python3"
if jargs.args.remote:
    python_exec = sys.executable

# The actual script when launched as standalone
# ----------------------------------------
# python_exec = sys.executable
# print(python_exec)

if not args.run and __name__ == '__main__':
    # Disallow running as root on linux
    if os.name == 'posix' and os.geteuid() == 0:
        print("You are warning as root, proceed at your own risks")

    # Check that we're running python 3.9 or higher
    if sys.version_info < (3, 9):
        print(f"Warning your python version {sys.version_info} is detected as lower than 3.9, you may be fucked, proceed at your own risks")

    # Check that we have git installed
    from src.lib import corelib
    has_git = corelib.has_exe('git')
    if not has_git:
        print("Please install git")
        exit(1)

    # Create a virtual environment if it doesn't exist
    if not args.no_venv:
        if not os.path.exists(VENV_DIR) or args.recreate_venv:
            if args.recreate_venv:
                shutil.rmtree(VENV_DIR)

            os.system(f"{python_exec} -m venv {VENV_DIR}")
            argp.upgrade = True

        # Run bash shell with commands to activate the virtual environment and run the launch script
        # if os.name == 'posix':
        os.system(f"bash -c 'source {VENV_DIR}/bin/activate'")
        # elif os.name == 'nt':


    if args.upgrade:
        # Install requirements with venv pip
        if args.no_venv:
            os.system(f"{python_exec} -m pip install -r requirements.txt")
        else:
            os.system(f"{VENV_DIR}/bin/pip install -r requirements.txt")
        print('----------------------------------------')
        print("\n\n")
        exit(0)

    if args.no_venv:
        os.system(f"bash -c '{python_exec} {__file__} {spaced_args} --upgrade --run'")
    else:
        os.system(f"bash -c 'source {VENV_DIR}/bin/activate && {python_exec} {__file__} {spaced_args} --upgrade --run'")

    exit(0)

# ----------------------------------------

# from src import renderer
# from src.renderer import get_script_path, parse_action_script, render_frame, render_init, render_loop
# from src import core
# from src.classes import paths
# from src.classes.logs import loglaunch, loglaunch_err
# from src.classes.Session import Session

from src.classes import paths

def install_core():
    """
Install all core requirements
"""
    from src.installer import check_import
    from src.installer import python
    from src.installer import pipargs
    from src.lib.printlib import printerr

    torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113")
    clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
    requirements_file = os.environ.get('REQS_FILE', "../requirements_versions.txt")

    paths.plug_repos.mkdir(exist_ok=True)
    paths.plug_logs.mkdir(exist_ok=True)
    paths.plug_res.mkdir(exist_ok=True)
    paths.plugins.mkdir(exist_ok=True)
    paths.sessions.mkdir(exist_ok=True)

    if not check_import("torch") or not check_import("torchvision"):
        from src import installer
        installer.run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch")

    try:
        import torch
        assert torch.cuda.is_available()
    except:
        printerr('Torch is not able to use GPU')
        sys.exit(1)

    if not check_import("clip"):
        pipargs(f"install {clip_package}", "clip")

def on_ctrl_c():
    from src.classes.logs import logdiscore
    from src import renderer

    logdiscore("Exiting because of Ctrl+C.")
    renderer.requests.stop = True
    exit(0)


def plugin_wizard():
    import shutil
    from src import installer
    import re
    from src.classes import paths
    from art import text2art

    PLUGIN_TEMPLATE_PREFIX = ".template"

    print(text2art("Plugin Wizard"))

    # Find template directories (start with .template)
    templates = []
    for d in paths.code_plugins.iterdir():
        if d.name.startswith(PLUGIN_TEMPLATE_PREFIX):
            templates.append(d)

    template = None

    if len(templates) == 0:
        print("No templates found.")
        exit(1)
    elif len(templates) == 1:
        template = templates[0]
    else:
        for i, path in enumerate(templates):
            s = path.name[len(PLUGIN_TEMPLATE_PREFIX):]
            while not s[0].isdigit() and not s[0].isalpha():
                s = s[1:]
            print(f"{i + 1}. {s}")

        print()
        while template is None:
            try:
                v = int(input("Select a template: ")) - 1
                if v >= 0:
                    template = templates[v]
            except:
                pass

    pid = input("ID name: ")

    clsdefault = f"{pid.capitalize()}Plugin"
    cls = input(f"Class name (default={clsdefault}): ")
    if not cls:
        cls = clsdefault

    plugdir = paths.code_plugins / f"{pid}_plugin"
    clsfile = plugdir / f"{cls}.py"

    shutil.copytree(template.as_posix(), plugdir)
    shutil.move(plugdir / "TemplatePlugin.py", clsfile)

    # Find all {{word}} with regex and ask for a replacement
    regex = r'__(\w+)__'
    with open(clsfile, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        matches = re.findall(regex, line)
        if matches:
            vname = matches[0]

            # Default values
            vdefault = ''
            if vname == 'classname': vdefault = cls
            if vname == 'title': vdefault = pid

            # Ask for a value
            if vdefault:
                value = input(f"{vname} (default={vdefault}): ")
            else:
                value = input(f"{vname}: ")

            if not value and vdefault:
                value = vdefault

            # Apply the value
            lines[i] = re.sub(regex, value, line)

    # Write lines back to file
    with open(clsfile, "w") as f:
        f.writelines(lines)

    # Open plugdir in the file explorer
    installer.open_explorer(plugdir)
    print("Done!")
    input()
    exit(0)

def main():
    from src.classes.logs import logdiscore_err
    from src.classes.logs import logdiscore
    from yachalk import chalk
    from src.lib.printlib import print_existing_sessions, print_possible_scripts

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

            print("")

            print("Scripts:")
            print_possible_scripts()
            exit(0)

        from src.classes.paths import parse_action_script
        a, sc = parse_action_script(args.action, DEFAULT_ACTION)

        logdiscore(chalk.green(f"action: {a}"))
        logdiscore(chalk.green(f"script: {sc}"))

        if a is not None:
            if a == 'help':
                print_possible_scripts()
                exit(0)


            # Run an action script
            # ----------------------------------------
            # Get the path, check if it exists
            apath = paths.get_script_file_path(a)
            if not apath.is_file():
                logdiscore_err(f"Unknown action '{args.action}' (searched at {apath})")
                print_possible_scripts()
                exit(1)

            # Load the action script module
            amod = importlib.import_module(paths.get_script_module_path(a), package=a)
            if amod is None:
                logdiscore_err(f"Couldn't load '{args.action}'")
                print_possible_scripts()
                exit(1)

            # By specifying this attribute, we can skip session loading when it's unnecessary to gain speed

            # t = threading.Thread(target=src.renderer.start_mainloop)
            # t.start()
            amod.action(args)

            print("Action done.")


            # threading.Thread(target=amod.action, args=tuple([args])).start()
            # renderer.start_mainloop()
        else:
            # Nothing is specified
            # ----------------------------------------
            # Dry run, only install and exit.
            if args.dry:
                logdiscore("Exiting because of -dry argument")
                exit(0)

            # Start server
            from old import server
            server.run()



# with cpuprofile():
#     import pytorch_lightning as pl

# from src.classes import printlib
# import userconf
#
# printlib.print_timing = userconf.print_timing
# printlib.print_trace = userconf.print_trace
# printlib.print_gputrace = userconf.print_gputrace
#
# if userconf.print_extended_init:
#     from src import installer
#     installer.print_info()
#     print()

if __name__ == '__main__':
    from src import core
    core.setup_annoying_logging()

    main()
