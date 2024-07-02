import logging
import subprocess
import sys
from pathlib import Path

import paramiko

import deploy_utils
from src.lib.loglib import print_cmd

log = logging.getLogger(__name__)

class DiscoreSSHClient(paramiko.SSHClient):
    """
    Custom SSH client for Discore deployments.

    This class extends paramiko's SSHClient with additional functionality
    specific to Discore deployments, such as command execution, file existence
    checks, and remote mounting.

    Methods:
        run(cm, cwd, log_output): Executes a command on the remote instance.
        file_exists(path): Checks if a file exists on the remote instance.
        mount(local_path, remote_path, ip, port): Mounts a remote directory locally.
    """

    def run(self, cm, cwd=None, *, log_output=False):
        """
        Execute a command on the remote instance.
        """
        cm = cm.replace("'", '"')
        if cwd is not None:
            cm = f"cd {cwd}; {cm}"

        # if sys.platform.startswith('win'):
        #     cm = f'powershell -Command "{cm}"'
        # else:
        cm = f"/bin/bash -c '{cm}'"

        print_cmd(cm, log)
        stdin, stdout, stderr = self.exec_command(cm, get_pty=True)
        stdout.channel.set_combine_stderr(True)
        ret = ''
        for line in stdout:
            ret += line
            if log_output:
                log.info(line, end='')

        return ret

    def file_exists(self, path):
        """
        Check if a file exists on the remote instance.
        """
        if sys.platform.startswith('win'):
            cmd = f'Test-Path "{path}"'
        else:
            cmd = f"test -e '{path}'"

        ret = self.run(cmd)
        return ret.strip().lower() == 'true' if sys.platform.startswith('win') else ret == ''

    def mount(self, local_path, remote_path, ip, port):
        """
        Mount a remote directory locally
        """
        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        if sys.platform.startswith('win'):
            # For Windows, we'll use SSHFS-Win
            # not_mounted = subprocess.run(f'net use {local_path} 2>nul', shell=True).returncode != 0
            # if not_mounted:
            #     mount_cmd = f'sshfs.exe root@{ip}:{remote_path} {local_path} -p {port}'
            log.info("Windows not supported yet.")
            return
        elif sys.platform.startswith('darwin'):
            # For macOS
            not_mounted = subprocess.run(f'mount | grep -q "{local_path}"', shell=True).returncode != 0
            if not_mounted:
                mount_cmd = f'sshfs root@{ip}:{remote_path} {local_path} -p {port} -o volname=Discore'
        else:
            # For Linux
            not_mounted = subprocess.run(f'mountpoint -q {local_path}', shell=True).returncode != 0
            if not_mounted:
                mount_cmd = f'sshfs root@{ip}:{remote_path} {local_path} -p {port}'

        if not_mounted:
            deploy_utils.print_header("Mounting with sshfs...")
            result = subprocess.run(mount_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                log.info("Mounted successfully!")
            else:
                log.info(f"Failed to mount. Error: {result.stderr}")
        else:
            log.info("Already mounted.")

    def print_cmd(cmd):
        # Implementation of print_cmd function
        log.info(f"> {cmd}")

    def print_header(header):
        # Implementation of print_header function
        log.info(f"\n{'=' * 20}\n{header}\n{'=' * 20}")
