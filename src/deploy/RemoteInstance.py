import concurrent
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, List, Tuple

import paramiko
from desktop_notifier import DesktopNotifier
from paramiko import SSHException
from paramiko.ssh_exception import NoValidConnectionsError
from yachalk import chalk

import jargs
import userconf
from src.deploy.DiscoreSSHClient import DiscoreSSHClient
from deploy_utils import open_terminal, open_shell, print_header
from jargs import args
from src import renderer
from src.classes import Session, paths
from src.deploy.VastInstance import VastInstance
from src.deploy.watch import Watcher
import src.deploy.constants as const

logger = logging.getLogger(__name__)

class RemoteInstance:
    """
 Represents and manages a remote instance on Vast.ai.

 This class handles all operations related to a single remote instance,
 including deployment, file transfers, and job management.

 Attributes:
     data (VastInstance.VastInstance): Data about the Vast.ai instance.
     id (str): Unique identifier of the instance.
     ip (str): IP address of the instance.
     port (str): SSH port of the instance.
     session (Session): Associated Discore session.
     ssh (DiscoreSSHClient.DiscoreSSHClient): SSH client for the instance.
     sftp (SFTPClient): SFTP client for file transfers.
     continue_work (bool): Flag to control job execution.
     notifier (DesktopNotifier): Notifier for balance updates.
     last_balance (float): Last known account balance.
     watcher (Watcher): File watcher for detecting local changes.

 Methods:
     connect(): Establishes SSH and SFTP connections to the instance.
     set_data(instance_data): Updates instance data.
     wait_for_ready(timeout): Waits for the instance to be in 'running' state.
     deploy(session, ...): Deploys Discore to the remote instance.
     send_files(src, dst, ...): Sends files to the remote instance.
     run_git_clones(dst): Clones necessary Git repositories on the remote instance.
     discore_job(): Runs the main Discore job on the remote instance.
     balance_job(): Monitors and notifies about account balance changes.
     upload_job(): Watches for local file changes and uploads them.
     download_job(): Downloads session data from the remote instance.
     start_jobs(): Starts all background jobs.
     stop_jobs(): Stops all background jobs.
    """

    def __init__(self, instance_data: VastInstance, session: Session):
        self.data: Optional[VastInstance] = None
        self.id = None
        self.ip = None
        self.port = None
        self.session: Session = session
        self.ssh = None
        self.sftp = None

        self.continue_work = True
        self.notifier = DesktopNotifier()
        self.last_balance = None
        self.watcher = None

        self.set_data(instance_data)

    def connect(self):
        self.ssh = DiscoreSSHClient()
        self.ssh.load_host_keys(os.path.expanduser(os.path.join('~', '.ssh', 'known_hosts')))
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        num_connection_failure = 0
        while True:
            try:
                self.ssh.connect(self.ip, port=int(self.port), username='root')
                break
            except NoValidConnectionsError as e:
                logger.warning(f"Failed to connect ({e}), retrying...")
                time.sleep(3)
                num_connection_failure += 1
            except SSHException:
                open_terminal(f"ssh -p {self.port} root@{self.ip}")

        logger.info(f"Successfully connected through SSH after {num_connection_failure} retries.")

        from src.deploy.sftpclient import SFTPClient
        self.sftp = SFTPClient.from_transport(self.ssh.get_transport())
        self.sftp.max_size = 10 * 1024 * 1024
        self.sftp.urls = getattr(userconf, 'deploy_urls', [])
        self.sftp.ssh = self.ssh
        self.sftp.enable_urls = not args.vastai_no_download
        self.sftp.ip = self.ip
        self.sftp.port = self.port

    def set_data(self, instance_data: VastInstance):
        self.data = instance_data
        self.id = instance_data.id
        self.ip = instance_data.sshaddr
        self.port = instance_data.sshport

    def wait_for_ready(self, timeout=None):
        start_time = time.time()
        while True:
            instances = vast_manager.fetch_instances()
            instance_data = next((i for i in instances if i.id == self.id), None)

            if instance_data:
                self.set_data(instance_data)
                if self.data.status == 'running':
                    logger.info(f"Instance {self.id} is ready")
                    return True

            logger.info(f"Waiting for instance {self.id} to be ready (status: {self.data.status if instance_data else 'unknown'})")

            if timeout and time.time() - start_time > timeout:
                logger.warning(f"Timeout waiting for instance {self.id} to be ready")
                return False

            time.sleep(3)

    def deploy(self,
               session,
               b_clone=False,
               b_shell=False,
               b_sshfs=userconf.vastai_sshfs,
               b_copy_workdir=False,
               b_pip_upgrades=False):
        self.wait_for_ready()
        self.connect()

        if session:
            self.session = session
        else:
            session = self.session

        src = paths.root
        dst = Path("/workspace/discore_deploy")
        src_session = session.dirpath
        dst_session = dst / 'sessions' / session.dirpath.stem
        install_checkpath = "/root/.discore_installed"

        is_fresh_install = not self.sftp.exists(install_checkpath)
        if is_fresh_install:
            logger.info(chalk.green("Installing system packages ..."))
            for apt_package in const.APT_PACKAGES:
                logger.info(f"Installing {apt_package}...")
                self.ssh.run(f"apt-get install {apt_package} -y")

        clone = b_clone or not self.ssh.file_exists(dst)
        if clone:
            self.run_git_clones(dst)
            self.ssh.run(f"chmod +x {dst / 'discore.py'}")

        if b_sshfs:
            local_path = Path(userconf.vastai_sshfs_path).expanduser() / 'discore_deploy'
            thread = threading.Thread(target=self.ssh.mount, args=(local_path,
                                                                   '/workspace/discore_deploy',
                                                                   self.ip,
                                                                   self.port))
            thread.start()

        if b_copy_workdir:
            # Send all text files
            self.send_files(src, dst, const.FAST_UPLOADS, rclone_includes=[f'*{v}' for v in paths.text_exts])

        if is_fresh_install or args.vastai_copy:
            self.send_files(src, dst, const.SLOW_UPLOADS, is_ftp=True)

        if b_shell:
            open_shell(self.ssh)

        if is_fresh_install or b_pip_upgrades:
            logger.info(chalk.green("Discore pip refresh"))
            discore_cmd = f"cd /workspace/discore_deploy/; {const.VAST_AI_PYTHON_BIN} {dst / 'discore.py'} {self.session.name} --cli --remote --unsafe --no_venv"
            subprocess.run(f"ssh -p {self.port} root@{self.ip} '{discore_cmd}' --upgrade", shell=True)

        self.ssh.run(f"touch {install_checkpath}")

        if not args.vastai_quick:
            copy_session(self.session, dst_session, self.sftp)

    def send_files(self,
                   src: Path,
                   dst: Path,
                   file_list: List[str | Tuple[str, str]],
                   is_ftp: bool = False,
                   rclone_includes: List[str] = None):
        """
        Send files to the remote machine.

        Args:
        - src (Path): Source directory
        - dst (Path): Destination directory
        - sftp: SFTP client object
        - file_list (List[Union[str, Tuple[str, str]]]): List of files to send
        - is_ftp (bool): Whether to use FTP mode (forbids rclone)
        - rclone_includes (List[str]): List of patterns to include for rclone
        """
        if not is_ftp:
            print_header("File copies...")

        for file in file_list:
            source = src / (file[0] if isinstance(file, tuple) else file)
            destination = dst / (file[1] if isinstance(file, tuple) else file)

            kwargs = {'forbid_rclone': True} if is_ftp else {'rclone_includes': rclone_includes}
            self.sftp.put_any(source, destination, **kwargs)

    def run_git_clones(self, dst: Path):
        if self.ssh.file_exists(dst):
            logger.info(chalk.red("Removing old deploy..."))
            time.sleep(3)
            self.ssh.run(f"rm -rf {dst}")
        logger.info(chalk.green("Cloning repositories..."))

        # TODO move to constant or config idk
        repos = {
            dst: ['https://github.com/distable/discore'],
            dst / 'ComfyUI': ['https://github.com/comfyanonymous/ComfyUI'],
            (dst / 'ComfyUI' / 'custom_nodes'): [
                'https://github.com/ltdrdata/ComfyUI-Manager',
                'https://github.com/chrisgoringe/cg-use-everywhere',
                'https://github.com/giriss/comfy-image-saver',
                'https://github.com/ethansmith2000/comfy-todo',
                'https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes',
                'https://github.com/Fannovel16/comfyui_controlnet_aux',
                'https://github.com/cubiq/ComfyUI_IPAdapter_plus',
                'https://github.com/shiimizu/ComfyUI_smZNodes',
                'https://github.com/crystian/ComfyUI-Crystools',
                'https://github.com/pythongosssss/ComfyUI-Custom-Scripts',
                'https://github.com/styler00dollar/ComfyUI-deepcache',
                'https://github.com/adieyal/comfyui-dynamicprompts',
                'https://github.com/kijai/ComfyUI-KJNodes',
                'https://github.com/shiimizu/ComfyUI-TiledDiffusion'
            ]
        }

        def clone_repo(repo: str, target_dir: Path):
            recursive = '--recursive' if 'discore' in repo or 'ComfyUI' in repo else ''
            cmd = f"git clone {recursive} {repo} {target_dir}"
            self.ssh.run(cmd)
            logger.info(f"Cloned {repo} to {target_dir}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_repo = {}
            for target_dir, repo_list in repos.items():
                for repo in repo_list:
                    future = executor.submit(clone_repo, repo, target_dir)
                    future_to_repo[future] = repo

            for future in concurrent.futures.as_completed(future_to_repo):
                repo = future_to_repo[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error(f"{repo} generated an exception: {exc}")

        self.ssh.run(f"git -C {dst} submodule update --init --recursive")

    def discore_job(self):
        """
        Run the main Discore job on the remote instance
        """
        dst = Path("/workspace/discore_deploy")
        cmd = f"cd {dst}; {const.VAST_AI_PYTHON_BIN} {dst / 'discore.py'}"

        oargs = jargs.argv
        jargs.remove_deploy_args(oargs)
        cmd += f' {" ".join(oargs)}'
        cmd += " --run -cli --remote --unsafe --no_venv"

        logger.info("Launching discore for work ...")

        if args.vastai_upgrade or args.vastai_install:
            cmd += " --install"

        if args.vastai_trace:
            cmd += " --trace"

        while self.continue_work:
            logger.info(f"> {cmd}")
            self.ssh.run(cmd, log_output=True)

        renderer.request_stop = True

    def balance_job(self):
        """
        Monitor and notify about account balance changes.
        """
        threshold = 0.25
        elapsed = 0
        while self.continue_work:
            if elapsed > 5:
                elapsed = 0
                balance = vast_manager.fetch_balance()
                if self.last_balance is None or balance - self.last_balance > threshold:
                    self.last_balance = balance
                    self.notifier.send_sync(title='Vast.ai balance', message=f'{balance:.02f}$')
            time.sleep(0.1)
            elapsed += 0.1

    def upload_job(self):
        """
        Watch for local file changes and upload them.
        """
        src = paths.root
        dst = Path("/workspace/discore_deploy")

        def execute(f):
            f = Path(f)
            logger.info(chalk.blue_bright("Changed", f.relative_to(src)))
            relative = f.relative_to(src)
            src2 = src / relative
            dst2 = dst / relative
            self.sftp.put_rclone(src2, dst2, False, [], [])

        self.watcher = Watcher([*const.RCLONE_JOB_UPLOAD_PATHS, self.session.dirpath / 'script.py'], [execute])

        while self.continue_work:
            self.watcher.monitor_once()
            time.sleep(1)

    def download_job(self):
        """
        Download session data from the remote instance.
        """
        src = paths.root
        dst = Path("/workspace/discore_deploy")

        while self.continue_work:
            src2 = src / paths.sessions_name
            dst2 = dst / paths.sessions_name / self.session.name

            self.sftp.get_rclone(dst2, src2, False, const.RCLONE_JOB_DOWNLOAD_EXCLUSION, [])

            time.sleep(3)

    def start_jobs(self):
        jobs = [
            threading.Thread(target=self.discore_job),
            threading.Thread(target=self.balance_job),
            threading.Thread(target=self.upload_job),
            threading.Thread(target=self.download_job),
        ]
        for job in jobs:
            job.start()
        return jobs

    def stop_jobs(self):
        self.continue_work = False
        if self.watcher:
            self.watcher.stop()
