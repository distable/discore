"""
Discore Deployment Script for Vast.ai

This script manages the deployment of Discore to Vast.ai instances, supporting
multiple remote deployments for parallel rendering of animations.

Note: In this context, 'src' always refers to the local machine, 'dst' refers to the remote machine.
"""
import concurrent
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import paramiko
from paramiko.ssh_exception import NoValidConnectionsError, SSHException
from yachalk import chalk

import jargs
import userconf
from jargs import args
from src import renderer
from src.classes import Session
from src.classes import paths
from src.lib.loglib import print_cmd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

ssh = None


# region Utilities
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


def download_vast_script():
    os_type = get_os()
    vastpath = paths.root / 'vast'
    if not vastpath.is_file():
        if os_type == 'windows':
            subprocess.run(["powershell", "-Command", f"Invoke-WebRequest -Uri https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -OutFile {vastpath}"], check=True)
        else:
            os.system(f"wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O {vastpath}; chmod +x {vastpath};")


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


# endregion


@dataclass
class VastOffer:
    """
    A vast.ai offer data class.
    An offer is a machine that can be rented by the user.
    """
    index: int
    id: str
    cuda: str
    num: str
    model: str
    pcie: str
    cpu_ghz: str
    vcpus: str
    ram: str
    disk: str
    price: str
    dlp: str
    dlp_per_dollar: str
    score: str
    nv_driver: str
    net_up: str
    net_down: str
    r: str
    max_days: str
    mach_id: str
    status: str
    ports: str
    country: str

    def __str__(self):
        return self.tostring(self.index)

    def tostring(self, i):
        return f'[{i + 1:02}] - {self.model} - {self.num} - {self.dlp} - {self.net_down} Mbps - {self.price} $/hr - {self.dlp_per_dollar} DLP/HR'


@dataclass
class VastInstance:
    """
    A vast.ai instance data class.
    An instance is a machine rente by the user. (running, stopped, launching, etc.)
    It contains private information for launching the data.
    """
    index: int
    id: str
    machine: str
    status: str
    num: str
    model: str
    util: str
    vcpus: str
    ram: str
    storage: str
    sshaddr: str
    sshport: str
    price: str
    image: str
    netup: str
    netdown: str
    r: str
    label: str
    age: str

    def __str__(self):
        return f'{self.index + 1} - {self.model} - {self.num} - {self.netdown} Mbps - {self.price} $/hr - {self.status}'


class RemoteInstance:
    def __init__(self, instance_data: VastInstance, session: Session):
        self.data: Optional[VastInstance] = None
        self.id = None
        self.ip = None
        self.port = None
        self.session: Session = session
        self.ssh = None
        self.sftp = None

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
               b_shell=args.shell,
               b_discore_pip_upgrades=args.vastai_upgrade,
               b_copy_workdir=not args.vastai_quick,
               b_clone=not jargs.is_vastai_continue,
               b_sshfs=userconf.vastai_sshfs):
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
            for apt_package in APT_PACKAGES:
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
            self.send_files(src, dst, FAST_UPLOADS, rclone_includes=[f'*{v}' for v in paths.text_exts])

        if is_fresh_install or args.vastai_copy:
            self.send_files(src, dst, SLOW_UPLOADS, is_ftp=True)

        if b_shell:
            open_shell(self.ssh)

        if is_fresh_install or b_discore_pip_upgrades:
            logger.info(chalk.green("Discore pip refresh"))
            discore_cmd = f"cd /workspace/discore_deploy/; {VAST_AI_PYTHON_BIN} {dst / 'discore.py'} {self.session.name} --cli --remote --unsafe --no_venv"
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

    def start_jobs(self):
        jobs = [
            threading.Thread(target=self.vastai_job),
            threading.Thread(target=self.balance_job),
            threading.Thread(target=self.upload_job),
            threading.Thread(target=self.download_job),
        ]
        for job in jobs:
            job.start()
        return jobs

    def vastai_job(self):
        # Implementation of vastai_job
        pass

    def balance_job(self):
        # Implementation of balance_job
        pass

    def upload_job(self):
        # Implementation of upload_job
        pass

    def download_job(self):
        # Implementation of download_job
        pass


class VastAIManager:
    def __init__(self):
        self.vastpath = paths.root / 'vast'
        self.remotes = dict()

    def download_vast_script(self):
        if not self.vastpath.is_file():
            if sys.platform.startswith('win'):
                subprocess.run(["powershell", "-Command", f"Invoke-WebRequest -Uri https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -OutFile {self.vastpath}"], check=True)
            else:
                os.system(f"wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O {self.vastpath}; chmod +x {self.vastpath};")

    def run_command(self, command):
        return subprocess.check_output([sys.executable, self.vastpath.as_posix()] + command).decode('utf-8')

    def fetch_balance(self):
        out = self.run_command(['show', 'invoices'])
        return json.loads(out.splitlines()[-1].replace('Current:  ', '').replace("'", '"'))['credit']

    def fetch_offers(self) -> List[VastOffer]:
        search_query = f"{args.vastai_search or userconf.vastai_default_search} disk_space>{DISK_SPACE} verified=true"
        out = self.run_command(['search', 'offers', search_query])
        return [
            VastOffer(index, *fields[:22])
            for index, line in enumerate(out.splitlines()[1:], start=1)
            if len(fields := line.split()) >= 22
        ]

    def fetch_instances(self) -> List[VastInstance]:
        """
        Retrieve all instances rented by the user.
        """
        out = self.run_command(['show', 'instances'])
        return [
            VastInstance(index, *fields[:18])
            for index, line in enumerate(out.splitlines()[1:], start=1)
            if len(fields := line.split()) >= 18
        ]

    def get_instance(self, instance_id) -> RemoteInstance:
        """
        Retrieve a specific instance by its ID.
        """
        if instance_id in self.remotes:
            return self.remotes[instance_id]

        instances = self.fetch_instances()
        instance_data = next((i for i in instances if i.id == instance_id), None)
        if instance_data:
            self.remotes[instance_id] = RemoteInstance(instance_data, None)
            return RemoteInstance(instance_data, None)

        raise ValueError(f"Instance {instance_id} not found")

    def create_instance(self, offer_id):
        return self.run_command(['create', 'instance', offer_id, '--image', DOCKER_IMAGE, '--disk', DISK_SPACE, '--env', '-p 8188:8188', '--ssh'])

    def destroy_instance(self, instance_id):
        self.remotes.pop(instance_id, None)
        return self.run_command(['destroy', 'instance', str(instance_id)])

    def reboot_instance(self, instance_id):
        return self.run_command(['reboot', 'instance', str(instance_id)])

    def stop_instance(self, instance_id):
        return self.run_command(['stop', str(instance_id)])


class RemoteManager:
    def __init__(self):
        self.remotes = []
        self.vast_manager = VastAIManager()

    def add_remote(self, instance_data, session):
        remote = RemoteInstance(instance_data, session)
        self.remotes.append(remote)
        return remote

    def deploy_all(self):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(remote.deploy) for remote in self.remotes]
            for future in futures:
                future.result()

    def start_all_jobs(self):
        all_jobs = []
        for remote in self.remotes:
            all_jobs.extend(remote.start_jobs())
        return all_jobs


vast_manager: VastAIManager = VastAIManager()


def deploy_vastai():
    vast_manager.download_vast_script()

    session = jargs.get_discore_session()

    instances = vast_manager.fetch_instances()
    if args.vastai_list:
        for instance in instances:
            print(chalk.green_bright(instance))
        return

    selected_id = instances[0].id if instances else None
    if selected_id is None:
        offers = vast_manager.fetch_offers()
        offers.sort(key=lambda e: float(e.price), reverse=True)
        for i, offer in enumerate(offers):
            print(chalk.green(offer.tostring(i)))

        choice = int(input("Enter the number of the machine you want to use: ")) - 1
        print(f"Creating instance with offer {offers[choice].id} ...")

        selected_id = offers[choice].id
        vast_manager.create_instance(selected_id)

        time.sleep(3)
        new_instances = [i for i in vast_manager.fetch_instances() if i.id not in [e.id for e in instances]]
        if len(new_instances) != 1:
            logger.error("Failed to create instance, couldn't find the new instance. Check that you have spare credits.")
            return
        selected_id = new_instances[0].id

    if args.vastai_delete:
        logger.info(f"Deleting Vast.ai instance {selected_id} in 5 seconds ...")
        time.sleep(5)
        vast_manager.destroy_instance(selected_id)
        logger.info("All done!")
        return

    if args.vastai_reboot:
        logger.info(f"Rebooting Vast.ai instance {selected_id} ...")
        vast_manager.reboot_instance(selected_id)

    instance = vast_manager.get_instance(selected_id)
    instance.deploy(session)

    jobs = instance.start_jobs()

    for job in jobs:
        job.join()

    logger.info(f"Remaining balance: {vast_manager.fetch_balance():.02f}$")

    if args.vastai_stop:
        logger.info(f"Stopping Vast.ai instance {selected_id} in 5 seconds ...")
        time.sleep(5)
        vast_manager.stop_instance(selected_id)
        logger.info("All done!")


def copy_session(src_session, dst_path, sftp):
    if src_session is not None and src_session.dirpath.exists():
        logger.info(chalk.green(f"Copying session '{src_session.dirpath.stem}'"))
        sftp.mkdir(str(dst_path))
        sftp.put_any(src_session.dirpath, dst_path, forbid_recursive=True)


def renderer_job():
    renderer.enable_readonly = True
    renderer.enable_dev = True
    renderer.unsafe = False
    renderer.init()
    renderer.run()


def multi_deploy_vastai():
    manager = RemoteManager()
    manager.vast_manager.download_vast_script()

    sessions = jargs.get_discore_sessions()  # Assume this function returns multiple sessions

    instances = manager.vast_manager.fetch_instances()
    if args.vastai_list:
        for instance in instances:
            print(str(instance))
        return

    num_instances = len(sessions)
    selected_ids = []

    # Use existing instances or create new ones
    for _ in range(num_instances):
        if instances:
            selected_ids.append(instances.pop(0).id)
        else:
            offers = manager.vast_manager.fetch_offers()
            offers.sort(key=lambda e: float(e['price']), reverse=True)
            for offer in offers:
                print(str(offer))

            choice = int(input("Enter the number of the machine you want to use: ")) - 1
            offer_id = offers[choice].id
            manager.vast_manager.create_instance(offer_id)

            time.sleep(3)
            new_instance = manager.vast_manager.fetch_instances()[-1]
            selected_ids.append(new_instance.id)

    # Wait for all instances to be ready
    ready_instances = []
    for instance_id in selected_ids:
        instance = manager.vast_manager.get_instance(instance_id)
        ready_instances.append(instance.wait_for_ready())

    # Create RemoteInstance objects and connect
    for instance, session in zip(ready_instances, sessions):
        remote = manager.add_remote(instance, session)
        remote.connect()

    manager.deploy_all()  # Deploy to all instances
    all_jobs = manager.start_all_jobs()  # Start jobs on all instances

    # Wait for all jobs to complete
    for job in all_jobs:
        job.join()

    logger.info(f"Remaining balance: {manager.vast_manager.fetch_balance():.02f}$")

    if args.vastai_stop:
        logger.info(f"Stopping all Vast.ai instances in 5 seconds ...")
        time.sleep(5)
        for remote in manager.remotes:
            manager.vast_manager.stop_instance(remote.id)
        logger.info("All instances stopped.")


class DiscoreSSHClient(paramiko.SSHClient):
    def run(self, cm, cwd=None, *, log_output=False):
        cm = cm.replace("'", '"')
        if cwd is not None:
            cm = f"cd {cwd}; {cm}"

        # if sys.platform.startswith('win'):
        #     cm = f'powershell -Command "{cm}"'
        # else:
        cm = f"/bin/bash -c '{cm}'"

        print_cmd(cm)
        stdin, stdout, stderr = self.exec_command(cm, get_pty=True)
        stdout.channel.set_combine_stderr(True)
        ret = ''
        for line in stdout:
            ret += line
            if log_output:
                print(line, end='')

        return ret

    def file_exists(self, path):
        if sys.platform.startswith('win'):
            cmd = f'Test-Path "{path}"'
        else:
            cmd = f"test -e '{path}'"

        ret = self.run(cmd)
        return ret.strip().lower() == 'true' if sys.platform.startswith('win') else ret == ''

    def mount(self, local_path, remote_path, ip, port):
        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        if sys.platform.startswith('win'):
            # For Windows, we'll use SSHFS-Win
            # not_mounted = subprocess.run(f'net use {local_path} 2>nul', shell=True).returncode != 0
            # if not_mounted:
            #     mount_cmd = f'sshfs.exe root@{ip}:{remote_path} {local_path} -p {port}'
            print("Windows not supported yet.")
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
            print_header("Mounting with sshfs...")
            result = subprocess.run(mount_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("Mounted successfully!")
            else:
                print(f"Failed to mount. Error: {result.stderr}")
        else:
            print("Already mounted.")

    def print_cmd(cmd):
        # Implementation of print_cmd function
        print(f"> {cmd}")

    def print_header(header):
        # Implementation of print_header function
        print(f"\n{'=' * 20}\n{header}\n{'=' * 20}")


def deploy_local():
    import shutil
    import platform

    # A test 'provider' which attempts to do a clean clone of the current installation
    # 1. Clone the repo to ~/discore_deploy/
    src = paths.root
    dst = Path.home() / "discore_deploy"

    shutil.rmtree(dst.as_posix())

    clonepath = dst.as_posix()
    cmds = [
        ['git', 'clone', '--recursive', 'https://github.com/distable/discore', clonepath],
        ['git', '-C', clonepath, 'submodule', 'update', '--init', '--recursive'],
    ]
    for cmd in cmds:
        subprocess.run(cmd)

    for file in FAST_UPLOADS:
        if isinstance(file, tuple):
            shutil.copyfile(src / file[0], dst / file[1])
        if isinstance(file, str):
            shutil.copyfile(src / file, dst / file)

    # 3. Run ~/discore_deploy/discore.py
    if platform.system() == "Linux":
        subprocess.run(['chmod', '+x', dst / 'discore.py'])

    subprocess.run([dst / 'discore.py', '--upgrade'])