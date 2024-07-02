import os
import subprocess
import sys

from yachalk import chalk

from src.classes import paths


def run_vast_command(command):
    vastpath = paths.root / 'vast'
    return subprocess.check_output([sys.executable, vastpath.as_posix()] + command).decode('utf-8')


def get_os():
    if sys.platform.startswith('win'):
        return 'windows'
    elif sys.platform.startswith('darwin'):
        return 'macos'
    else:
        return 'linux'


def open_terminal(ssh_cmd):
    if sys.platform.startswith('win'):
        subprocess.Popen(f'start cmd /k {ssh_cmd}', shell=True)
    elif sys.platform.startswith('darwin'):
        subprocess.Popen(['open', '-a', 'Terminal', ssh_cmd])
    else:
        subprocess.Popen(['xterm', '-e', ssh_cmd])


def open_shell(ssh):
    import interactive
    print_header("user --shell")
    # Start a ssh shell for the user
    channel = ssh.invoke_shell()
    interactive.interactive_shell(channel)


def print_header(string):
    print("")
    print("----------------------------------------")
    print(chalk.green(string))
    print("----------------------------------------")
    print("")


def download_vast_script():
    os_type = get_os()
    vastpath = paths.root / 'vast'
    if not vastpath.is_file():
        if os_type == 'windows':
            subprocess.run(["powershell", "-Command", f"Invoke-WebRequest -Uri https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -OutFile {vastpath}"], check=True)
        else:
            os.system(f"wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O {vastpath}; chmod +x {vastpath};")
