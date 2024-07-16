import asyncio
import enum
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple, Union

import asyncssh
from desktop_notifier import DesktopNotifier
from yachalk import chalk

import jargs
import src.deploy.deploy_constants as const
import userconf
from jargs import args
from src.classes import paths
from src.classes.Session import Session
from src.deploy.DiscoSSH import DiscoSSH
from src.deploy.VastInstance import VastInstance
from src.deploy.deploy_utils import open_terminal, get_git_remote_urls, invalidate, make_header_text
from src.deploy.watch import Watcher



class DeploymentInstallationStep(Enum):
    """
    Refers to the progress of a deployment installation. (cloning, apt packages, pip installs)
    """
    ZERO = 0
    APT = 1
    GIT = 2
    DONE = 3


class SSHConnectionState(Enum):
    """
    Refers to the state of the SSH connection.
    """
    NONE = 0
    CONNECTING = 1
    CONNECTED = 2
    CONNECTION_ERROR = 3
    CONNECTION_LOST = 4
    PERMISSION_DENIED = 5
    HOST_KEY_NOT_VERIFIABLE = 6


@dataclass
class DeploymentStatus:
    """
    A data class to return the overall status of a deployed instance.
    """
    connection: SSHConnectionState = None
    """The state of the SSH connection."""

    installation_step = DeploymentInstallationStep.ZERO
    """Where are we at with this deployment installation?"""

    is_discore_running = False
    """Is the discore.py script running on the remote machine? (the python process will be tagged as discore)"""

    is_comfy_running = False

    """Is the ComfyUI main.py script running on the remote machine? (the python process will be tagged as ComfyUI)"""
    is_mounted = False

    """What session is this remote currently rendering?"""
    work_session: Optional[Session] = ""


class DiscoRemote:
    install_checkpath = "/root/.discore_installed"
    apt_checkpath = "/root/.packages_installed"

    def __init__(self, vdata: VastInstance, session: Session):
        self.vdata: Optional[VastInstance] = vdata
        self.session: Session = session
        self.ssh: Optional[DiscoSSH] = DiscoSSH(self)
        self.connection_state = SSHConnectionState.NONE

        self.disco_jobs = SyncJobs(self)
        self.comfy_jobs = ComfyJobs(self)
        self.notifier = DesktopNotifier()
        self.last_balance = None
        self.watcher = None

        self._log = logging.getLogger("Disc")

    @property
    def log(self):
        self._log.name = 'Disc'
        if self.vdata:
            self._log.name = f"Disc#{self.vdata.id}"

        return self._log

    async def to_view(self):
        return await DiscoRemoteView.from_remote(self)

    def info(self, message):
        if self.ssh:
            self.ssh.logs.info(message)
        self.log.info(message)
        invalidate()

    def warning(self, message):
        if self.ssh:
            self.ssh.warning(message)
        self.log.warning(message)
        invalidate()

    def error(self, message):
        if self.ssh:
            self.ssh.logs.error(message)
        self.log.error(message)
        invalidate()

    async def connect(self):
        self.connection_state = SSHConnectionState.CONNECTING
        self.ssh.reset()

        try:
            # self.info(make_header_text("Connecting through SSH ..."))
            await self.ssh.connect(self.ip, int(self.port), username='root')
        except asyncssh.ConnectionLost as e:
            self.log.error(f"Failed to connect ({e}).")
            self.connection_state = SSHConnectionState.CONNECTION_LOST
            return
        except asyncssh.PermissionDenied:
            self.connection_state = SSHConnectionState.PERMISSION_DENIED
            return
        except asyncssh.HostKeyNotVerifiable:
            self.warning("Host key not verifiable.")
            self.connection_state = SSHConnectionState.HOST_KEY_NOT_VERIFIABLE
            return

        self.log.info(f"Successfully connected through SSH.")
        self.connection_state = SSHConnectionState.CONNECTED
        # TODO Custom attributes for SFTP client need to be handled differently
        # as asyncssh's SFTPClient doesn't support custom attributes
        # You may need to create a wrapper class for the SFTP client

    async def disconnect(self):
        if self.ssh:
            await self.ssh.disconnect()
            self.connection_state = SSHConnectionState.NONE

    @property
    def src(self) -> Path:
        return paths.root

    @property
    def dst(self) -> Path:
        return Path("/workspace/discore_deploy")

    @property
    def id(self):
        return self.vdata.id

    @property
    def ip(self):
        return self.vdata.sshaddr

    @property
    def port(self):
        return self.vdata.sshport

    def set_data(self, vdata: VastInstance):
        self.vdata = vdata

    def refresh_data(self):
        from src.deploy.ui_vars import vast

        vdata = vast.fetch_instance(self.vdata.id)
        self.vdata = vdata

        if self.connection_state == SSHConnectionState.CONNECTED and \
                self.vdata.status != 'running':
            raise Exception("Instance has crashed while running for some reason - we need to handle this.")

    async def is_discore_running(self):
        return self.ssh and self.ssh.connection and await self.ssh.is_process_running('discore.py')

    async def is_comfy_running(self):
        return self.ssh and self.ssh.connection and await self.ssh.is_process_running('ComfyUI')

    async def probe_worksession(self) -> str:
        if not self.ssh:
            return ''

        work_session = await self.ssh.read_file(self.dst / '.work_session')
        return work_session or ''

    async def probe_deployment_status(self) -> DeploymentStatus:
        status = DeploymentStatus(connection=self.connection_state)

        if status.connection == SSHConnectionState.CONNECTED:
            # Use gather to run async checks concurrently
            is_discore_running, is_comfy_running, work_session = (
                await self.ssh.is_process_running('discore.py'),
                await self.ssh.is_process_running('comfy.py'),
                await self.probe_worksession()
            )

            status.is_discore_running = is_discore_running
            status.is_comfy_running = is_comfy_running
            status.work_session = work_session

            # Uncomment if needed:
            # status.is_mounted = await self.ssh.is_mounted()

        return status

    async def wait_for_ready(self, timeout: Optional[float] = None) -> bool:
        from src.deploy.DeployUI import vast

        if self.vdata.status == 'running':
            return True

        start_time = asyncio.get_running_loop().time()

        while True:
            try:
                instances = await vast.fetch_instances()
                vdata = next((i for i in instances if i.id == self.id), None)

                if vdata:
                    self.vdata = vdata
                    if self.vdata.status == 'running':
                        self.info(f"Instance {self.id} is ready")
                        return True

                self.info(f"Waiting for instance {self.id} to be ready (status: {self.vdata.status if vdata else 'unknown'})")

                if timeout and (asyncio.get_running_loop().time() - start_time > timeout):
                    self.warning(f"Timeout waiting for instance {self.id} to be ready")
                    return False

                await asyncio.sleep(3)

            except Exception as e:
                self.error(f"Error while waiting for instance to be ready: {str(e)}")
                return False

    async def is_ready(self) -> bool:
        from src.deploy.DeployUI import vast

        try:
            instances = await vast.fetch_instances()
            vdata = next((i for i in instances if i.id == self.id), None)

            if vdata:
                self.vdata = vdata
                return self.vdata.status == 'running'

            return False

        except Exception as e:
            self.error(f"Error while checking instance readiness: {str(e)}")
            return False

    def can_deploy(self, session):
        return (self.vdata
                and self.vdata.status == 'running'
                and session is not None)

    async def deploy(self,
                     session_or_name,
                     *,
                     b_redeploy=False,
                     b_clone=False,
                     b_pip_upgrades=False,
                     b_send_fast=True,
                     b_send_slow=False):
        session = (session_or_name or self.session) if isinstance(session_or_name, Session) else Session(session_or_name)
        if session is None:
            self.error("No session to deploy")
            return

        if not self.can_deploy(session):
            self.error("Cannot deploy")
            return

        if not self.ssh.is_connected():
            self.error("Not connected")
            return

        self.session = session

        await self.wait_for_ready()

        src, dst = self.src, self.dst
        src_comfy, src_comfy_nodes = src / 'ComfyUI', src / 'ComfyUI' / 'custom_nodes'
        dst_comfy, dst_comfy_nodes = dst / 'ComfyUI', dst / 'ComfyUI' / 'custom_nodes'
        src_session = self.session.dirpath

        step = await self.probe_installation_step()  # zero, apt, git, optional, done
        is0 = step.value == DeploymentInstallationStep.ZERO.value
        is1 = step.value <= DeploymentInstallationStep.APT.value

        if b_redeploy:
            is1 = True
            step = DeploymentInstallationStep.ZERO
            await self.ssh.run(f"rm -rf {dst.as_posix()}")

        if is0:
            self.info(make_header_text(f"""Deploying session '{session.name}' to {self.ip} ...
Detected installation step: {step.name} ({step.value} / {DeploymentInstallationStep.DONE.value})"""))
            await self.apt_install()

        if b_clone or is1: await self.run_git_clones(dst)
        if b_send_fast or is1: await self.send_fast_uploads(dst, src)
        if b_send_slow or is1: await self.send_slow_uploads(dst, src)

        # Installing custom_nodes each subirectory requirements.txt (in python, based on the )
        if b_pip_upgrades or is1:
            # Fix a potential annoying warning ("There was an error checking the latest version of pip")
            await self.ssh.run("rm -r ~/.cache/pip/selfcheck/")

            # Update pip
            await self.ssh.run(f"{const.VAST_PYTHON_BIN} -m pip install --upgrade pip --force-reinstall", log_output=False)

            # Run pip install -r requirements
            await self.pip_upgrade()
            self.info(make_header_text("f) Installing custom nodes ..."))

            # Install the requirements.txt for each custom node
            repos = self.get_git_clones(dst / 'ComfyUI' / 'custom_nodes')
            for target_dir, repo_list in repos.items():
                for repo in repo_list:
                    reqfile = dst_comfy_nodes / Path(repo).stem / 'requirements.txt'
                    await self.ssh.run(f"{const.VAST_PIP_BIN} install -r {reqfile.as_posix()}", log_output=False)

            # Install the requirements.txt for ComfyUI
            reqfile = dst_comfy / 'requirements.txt'
            await self.ssh.run(f"{const.VAST_PIP_BIN} install -r {reqfile.as_posix()}", log_output=False)

            # Install the demucs package
            await self.ssh.run(f"{const.VAST_PIP_BIN} install --no-deps demucs", log_output=False)

        # Mark the installation
        if is1:
            self.info(make_header_text("DONE) Marking installation as done ..."))
            self.info("\n")
            await self.ssh.run_safe(f"touch {DiscoRemote.install_checkpath}")

        if not args.vastai_quick:
            await self.send_session(session)

    async def apt_install(self):
        self.info(make_header_text("a) Installing system packages ..."))
        for apt_package in const.APT_PACKAGES:
            # self.info(f"Installing {apt_package}...")
            try:
                await self.ssh.run(f"apt-get install {apt_package} -y")
            except Exception as e:
                self.error(f"Failed to install {apt_package}, skipping ...")
                self.error(e)


        # Mark the installation
        await self.ssh.run(f"touch {DiscoRemote.apt_checkpath}")

    async def probe_installation_step(self) -> DeploymentInstallationStep:
        if await self.ssh.file_exists(DiscoRemote.install_checkpath):
            return DeploymentInstallationStep.DONE
        if await self.ssh.file_exists(self.dst):
            return DeploymentInstallationStep.GIT
        if await self.ssh.file_exists(self.apt_checkpath):
            return DeploymentInstallationStep.APT

        return DeploymentInstallationStep.ZERO

    async def pip_upgrade(self, dst=None):
        dst = dst or self.dst
        self.info(make_header_text("e) Discore pip refresh"))
        discore_cmd = self.get_discore_command(False)  # TODO remove this when the refactor TODO comment in discore.py is cleared
        discore_cmd += " --upgrade"
        await self.ssh.run(discore_cmd, log_output=True)
        # await self.ssh.run(f"cd {dst}; pip install -r requirements.txt", log_output=True)

    async def mount(self):
        local_path = Path(userconf.vastai_sshfs_path).expanduser() / 'discore_deploy'
        # Note: This still uses a synchronous call. Consider using an async FUSE library if available.
        await asyncio.to_thread(self.ssh.mount, local_path, '/workspace/discore_deploy', self.ip, self.port)
        invalidate()

    async def shell(self):
        await open_terminal(f"ssh -p {self.port} root@{self.ip}")
        invalidate()

    async def send_session(self, src_session: Optional[Session] = None):
        dst_path = self.dst / 'sessions' / src_session.dirpath.stem

        if src_session is not None and src_session.dirpath.exists():
            self.info(make_header_text(f"Syncing session '{src_session.dirpath.stem}' to remote"))
            await self.ssh.mkdir(str(dst_path))
            await self.ssh.put_any(src_session.dirpath, dst_path, forbid_recursive=True, rclone_excludes=[*['*' + e for e in paths.image_exts]])

            # Send the latest image (no need to know everything before)
            # We could send the last few images as well, that could be useful
            src_file = self.session.f_last_path
            dst_file = dst_path / paths.sessions_name / self.session.dirpath.stem / src_file.name
            await self.ssh.put_any(src_file, dst_file)

        invalidate()

    async def send_slow_uploads(self, dst=None, src=None):
        logging.info(make_header_text("d) Sending slow uploads ..."))

        src = src or self.src
        dst = dst or self.dst
        await self.send_files(src, dst, const.SLOW_UPLOADS, is_ftp=False)
        invalidate()

    async def send_fast_uploads(self, dst=None, src=None):
        logging.info(make_header_text("c) Sending fast uploads ..."))

        src = src or self.src
        dst = dst or self.dst
        await self.send_files(src,
                              dst,
                              const.UPLOADS_ON_CONNECT,
                              rclone_includes=[f'*{v}' for v in paths.text_exts],
                              is_ftp=False)
        invalidate()

    async def send_files(self,
                         src: Path,
                         dst: Path,
                         file_list: List[str | Tuple[str, str]],
                         is_ftp: bool = False,
                         rclone_includes: List[str] = None):
        def process_file_paths(file_item: Union[str, Tuple[str, str]]) -> Tuple[Path, Path]:
            if isinstance(file_item, tuple):
                source_path, dest_path = file_item
            else:
                source_path = dest_path = file_item

            return normalize_path(src, source_path), normalize_path(dst, dest_path)

        def normalize_path(base_path: Path, file_path: str) -> Path:
            path = Path(file_path)
            if path.is_absolute():
                return path.relative_to(paths.root)
            return base_path / path

        async def transfer_file(source: Path, destination: Path):
            kwargs = {'forbid_rclone': True} if is_ftp else {'rclone_includes': rclone_includes}
            await self.ssh.put_any(source, destination, **kwargs)

        for it in file_list:
            a, b = process_file_paths(it)
            await transfer_file(a, b)

        invalidate()

    async def run_git_clones(self, dst: Path):
        if await self.ssh.file_exists(dst):
            self.info(chalk.red(f"b) Removing old deploy ({dst.as_posix()})..."))
            await asyncio.sleep(3)
            await self.ssh.run(f"rm -rf {dst.as_posix()}")

        # Clone the repositories
        # ----------------------------------------
        repos = self.get_git_clones(dst)

        self.info(make_header_text("Cloning repositories ..."))
        self.info(str(repos))

        tasks = []
        for target_dir, repo_list in repos.items():
            for repo in repo_list:
                if 'custom_nodes' in target_dir.as_posix():
                    await self.ssh.clone_repo(repo, target_dir / Path(repo).stem)
                else:
                    await self.ssh.clone_repo(repo, target_dir)
                # task = asyncio.create_task()
                # tasks.append(task)

        # await asyncio.gather(*tasks)
        await self.ssh.run(f"git -C {dst} submodule update --init --recursive")

        # Make discore.py and comfy executable
        discore_py = dst / 'discore.py'
        comfy_py = dst / 'ComfyUI' / 'main.py'
        await self.ssh.run_safe(f"chmod +x {discore_py.as_posix()}")
        await self.ssh.run_safe(f"chmod +x {comfy_py.as_posix()}")

        # Rename ComfyUI/main.py to ComfyUI/comfy.py
        # py_1 = dst / 'ComfyUI' / 'main.py'
        # py_2 = dst / 'ComfyUI' / 'comfy.py'
        # await self.ssh.run_safe(f"mv {py_1.as_posix()} {py_2.as_posix()}")

        invalidate()

    def get_git_clones(self, dst):
        src_nodes = self.src / 'ComfyUI' / 'custom_nodes'
        node_urls = get_git_remote_urls(src_nodes)
        repos = {
            dst: ['https://github.com/distable/discore'],
            dst / 'ComfyUI': ['https://github.com/comfyanonymous/ComfyUI'],
            (dst / 'ComfyUI' / 'custom_nodes'): [node['URL'] for node in node_urls]
        }
        return repos

    async def _discore_job(self, upgrade=False, install=False, trace=False):
        cmd = self.get_discore_command()
        if upgrade or install:
            cmd += " --install"
        if trace:
            cmd += " --trace"

        self.info(make_header_text("Launching discore for work ..."))
        await self.ssh.run(cmd, log_output=True)
        self.info(make_header_text("Discore job has ended."))

    def get_discore_command(self, run=False):
        dst = Path("/workspace/discore_deploy")
        dst_main_py = dst / 'discore.py'
        cmd = f"cd {dst.as_posix()}; {const.VAST_PYTHON_BIN} {dst_main_py.as_posix()}"
        oargs = jargs.remove_deploy_args(jargs.argv)
        cmd += f' {" ".join(oargs)}'
        cmd += " -cli --remote --unsafe --no_venv"
        if run:
            cmd += " --run"
        return cmd

    async def start_discore(self):
        await self.disco_jobs.start()
        await self._discore_job()

    async def stop_discore(self):
        await self.disco_jobs.stop()
        await self.ssh.kill_process('discore.py')

    async def start_comfy(self):
        await self.comfy_jobs.start()

    async def stop_comfy(self):
        await self.comfy_jobs.stop()


class ExecutionState(enum.Enum):
    OFF = 0
    RUNNING = 1

    @classmethod
    def from_bool(cls, b):
        if not b:
            return ExecutionState.OFF
        else:
            return ExecutionState.RUNNING


@dataclass
class DiscoRemoteView:
    """
    Provides a UI view for the DiscoRemote class with only
    the necessary information for the UI.
    """
    id: int
    ip: str
    port: int
    machine: str
    ssh: SSHConnectionState
    mounted: bool
    discore: ExecutionState
    comfy: ExecutionState
    _remote: DiscoRemote = None

    @property
    def vdata(self):
        return self._remote.vdata

    @classmethod
    async def from_remote(cls, remote: DiscoRemote) -> 'DiscoRemoteView':
        """

        @rtype: object
        """
        status = await remote.probe_deployment_status()

        # TODO verify what we're getting back here
        return cls(
            id=int(remote.id),
            ip=remote.ip,
            port=int(remote.port),
            machine=remote.vdata.status,
            ssh=remote.connection_state,
            mounted=await remote.ssh.is_mounted(),
            discore=ExecutionState.from_bool(await remote.is_discore_running()),
            comfy=ExecutionState.from_bool(await remote.is_comfy_running()),
            _remote=remote
        )


class ComfyJobs:
    def __init__(self, remote: DiscoRemote):
        self.remote = remote
        self.task = None
        self.continue_work = False

    async def start(self):
        self.remote.info(make_header_text("Starting ComfyUI..."))
        await self.kill()

        if self.task is None or self.task.done():
            self.continue_work = True
            self.task = asyncio.create_task(self.comfy_job())

    async def stop(self, kill=False):
        if self.task and not self.task.done():
            self.continue_work = False
            self.task.cancel()
            await self.task
            self.task = None
        elif kill:
            await self.kill()

    async def kill(self):
        await self.remote.ssh.run("pgrep -f '8188' | xargs -r kill -9")

    async def comfy_job(self):
        dst = Path("/workspace/discore_deploy/ComfyUI")
        cmd = f"cd {dst.as_posix()}; {const.VAST_PYTHON_BIN} main.py --listen 0.0.0.0 --port 8188"

        self.remote.info("Launching ComfyUI...")

        while self.continue_work:
            self.remote.info(f"> {cmd}")
            await self.remote.ssh.run(cmd, log_output=True)
            if self.continue_work:
                self.remote.info("ComfyUI stopped unexpectedly. Restarting in 5 seconds...")
                await asyncio.sleep(5)


class SyncJobs:
    def __init__(self, remote: DiscoRemote):
        self.remote = remote
        self.jobs = {
            # 'upload': self._upload_job,
            'download': self._download_job,
        }
        self._tasks = {}
        self._continue_work = False
        self._watcher = None

    async def start(self):
        self.remote.info(make_header_text("Starting Sync Jobs..."))
        self._continue_work = True
        for job_name, job_func in self.jobs.items():
            self._tasks[job_name] = asyncio.create_task(job_func())

    async def stop(self):
        self.remote.info(make_header_text("Stopping Discore..."))
        self._continue_work = False
        for task in self._tasks.values():
            task.cancel()
        if self._watcher:
            self._watcher.stop()
        # Wait for all tasks to complete their cancellation
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

    async def _upload_job(self):
        src = self.remote.src
        dst = self.remote.dst

        async def execute(changed_file):
            changed_file = Path(changed_file)
            self.remote.info(chalk.blue_bright("Changed", changed_file.relative_to(src)))
            relative = changed_file.relative_to(src)
            src2 = src / relative
            dst2 = dst / relative
            await self.remote.ssh.put_rclone(src2, dst2, False, [], [], print_cmd=False, print_output=False)

        watched_files = [
            *const.UPLOAD_JOB_PATHS,
            self.remote.session.dirpath / 'script.py'
        ]
        self._watcher = Watcher(watched_files, [execute])

        while self._continue_work:
            await self._watcher.monitor_once()
            await asyncio.sleep(1)

    async def _download_job(self):
        src = self.remote.src
        dst = self.remote.dst

        while self._continue_work:
            src2 = src / paths.sessions_name / self.remote.session.name
            dst2 = dst / paths.sessions_name / self.remote.session.name

            await self.remote.ssh.get_rclone(dst2, src2, False, const.DOWNLOAD_JOB_EXCLUSIONS, [], print_cmd=False, print_output=False)
            await asyncio.sleep(3)
