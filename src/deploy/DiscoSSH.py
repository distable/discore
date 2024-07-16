import asyncio
import io
import logging
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import aiofiles
import asyncssh
from asyncssh import SSHClientConnection
from yachalk import chalk

from src.deploy.deploy_utils import make_header_text, invalidate


class NoTermSSHClientConnection(asyncssh.SSHClientConnection):
    def _process_pty_req(self, packet):
        return False


class LogBuffer:
    def __init__(self):
        self.text = ""
        self.handlers = []

    def clear(self):
        self.text = ""

    def info(self, text):
        self.text += text + '\n'
        for handler in self.handlers:
            handler(text)

    def warning(self, text):
        self.info(chalk.yellow(text))
        for handler in self.handlers:
            handler(chalk.yellow(text))

    def error(self, text):
        self.info(chalk.red(text))
        for handler in self.handlers:
            handler(chalk.red(text))


class DiscoSSH:
    def __init__(self, remote):
        self.remote = remote
        self.connection: SSHClientConnection = None
        self.sftp = None
        self.max_size = 1000000000
        self.urls = {
            'sd-v1-5.ckpt': '',
            'vae.vae.pt': '',
        }
        self.enable_urls = True
        self.enable_print_upload = False
        self.enable_print_download = True
        self.ip = None
        self.port = None
        self.print_rsync = True
        self.logs = LogBuffer()

        self._log = logging.getLogger("DiscSSH")

    @property
    def log(self):
        self._log.name = "DiscSSH"
        if self.remote is not None and self.remote.vdata is not None:
            self._log = logging.getLogger(f"DiscSSH#{self.remote.vdata.id}")
        return self._log

    def is_connected(self):
        return self.connection is not None

    def reset(self):
        self.logs.clear()
        if self.connection:
            self.connection.close()
            self.connection = None
            self.sftp = None

    def debug(self, msg):
        self.log.debug(msg)

    def info(self, msg):
        self.log.info(msg)
        self.logs.info(msg)
        invalidate()

    def warning(self, msg):
        self.log.warning(msg)
        self.logs.warning(msg)
        invalidate()

    def error(self, msg):
        self.log.error(msg)
        self.logs.error(msg)
        invalidate()

    async def connect(self, hostname, port, username, password=None, key_filename=None):
        """Establish an SSH connection."""
        self.ip = hostname
        self.port = port
        self.connection = await asyncssh.connect(hostname, port=port, username=username)
        self.sftp = await self.connection.start_sftp_client()
        self.info(make_header_text(f"Connected to {hostname} on port {port} as {username}"))

    async def disconnect(self):
        """Close the SSH connection."""
        self.connection.close()
        self.connection = None
        self.sftp = None
        self.info(make_header_text("Disconnected"))

    async def run_safe(self, cmd, cwd=None, *, log_output=False, handle_output=None):
        """Execute a command on the remote instance with improved output handling for both stdout and stderr."""
        try:
            return await self.run(cmd, cwd, log_output=log_output, handle_output=handle_output)
        except asyncssh.ProcessError as e:
            self.error(f"Error running command: {e}")
            return e.exit_status

    async def run(self, cmd, cwd=None, *, log_output=False, handle_output=None, return_output=False):
        """Execute a command on the remote instance with improved output handling for both stdout and stderr."""
        cmd = cmd.replace("'", '"')
        if cwd is not None:
            cmd = f"cd {cwd}; {cmd}"

        cmd = f"/bin/bash -c '{cmd}'"
        # self.info(f"\n")
        self.info(f"> {cmd}")

        asyncssh.logging.set_log_level(logging.WARNING)
        async with self.connection.create_process(cmd) as process:
            async def read_stream(stream, prefix, is_error):
                output = []
                async for line in stream:
                    line = line.strip()
                    if not line: continue  # there were some empty lines...

                    full_line = f"{prefix}: {line}"
                    full_line = chalk.red(full_line) if is_error else chalk.gray(full_line)

                    output.append(full_line)
                    if log_output:
                        self.info(full_line)
                        if handle_output:
                            await handle_output(full_line)
                    else:
                        self.debug(full_line)
                return '\n'.join(output)

            # Create tasks for reading stdout and stderr
            stdout_task = asyncio.create_task(read_stream(process.stdout, "STDOUT", False))
            stderr_task = asyncio.create_task(read_stream(process.stderr, "STDERR", True))

            # Wait for both streams to be fully read
            stdout_output, stderr_output = await asyncio.gather(stdout_task, stderr_task)

            # Wait for the process to complete and get the exit code
            ret = await process.wait()

            asyncssh.logging.set_log_level(logging.INFO)

            if return_output:
                return stdout_output.strip()
            else:
                return ret

    async def run_detached(self, cmd, cwd=None, output_file=None):
        """
        Execute a command on the remote instance in a detached state,
        allowing it to continue running after disconnection.
        """
        cmd = cmd.replace("'", '"')
        if cwd is not None:
            cmd = f"cd {cwd} && {cmd}"

        if output_file is None:
            output_file = "/dev/null"

        # Use nohup to run the command in the background, immune to hangups
        detached_cmd = f"nohup {cmd} > {output_file} 2>&1 &"

        # Use 'disown' to remove the process from the shell's job control
        full_cmd = f"/bin/bash -c '{detached_cmd}; disown'"

        self.info(f"Starting detached process: {full_cmd}")

        result = await self.connection.run(full_cmd)

        if result.exit_status == 0:
            self.info("Detached process started successfully")
        else:
            self.log.error(f"Failed to start detached process. Exit status: {result.exit_status}")
            self.log.error(f"Error output: {result.stderr}")

        return result.exit_status

    async def check_process(self, process_name):
        """Check if a process is running on the remote instance."""
        cmd = f"pgrep -f {process_name}"
        result = await self.connection.run(cmd)
        return result.exit_status == 0

    async def kill_process(self, process_name):
        """Kill a process on the remote instance."""
        cmd = f"pkill -f {process_name}"
        result = await self.run(cmd)
        return result.exit_status == 0

    async def clone_repo(self, repo: str, target_dir: Path):
        recursive = '--recursive' if 'discore' in repo or 'ComfyUI' in repo else ''
        cmd = f"git clone {recursive} {repo} {target_dir.as_posix()}"
        await self.run(cmd, log_output=True)
        self.log.info(f"Cloned {repo} to {target_dir.as_posix()}")

    async def read_file(self, path):
        """Read a text file from the remote instance and return it."""
        result = await self.run(f'cat {path}')
        if result.exit_status == 0:
            return result.stdout
        else:
            self.error(f"Failed to read file {path}: {result.stderr}")
            return ""

    async def file_exists(self, path):
        """Check if a file exists on the remote instance."""
        result = await self.run(f'test -e {path}')
        return result.exit_status == 0

    async def mount(self, local_path, remote_path, ip, port):
        """Mount a remote directory locally"""
        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        if sys.platform.startswith('win'):
            self.info("Windows not supported yet.")
            return
        elif sys.platform.startswith('darwin'):
            not_mounted = await asyncio.to_thread(subprocess.run, f'mount | grep -q "{local_path}"', shell=True, check=False)
            not_mounted = not_mounted.returncode != 0
            if not_mounted:
                mount_cmd = f'sshfs root@{ip}:{remote_path} {local_path} -p {port} -o volname=Discore'
        else:
            not_mounted = await asyncio.to_thread(subprocess.run, f'mountpoint -q {local_path}', shell=True, check=False)
            not_mounted = not_mounted.returncode != 0
            if not_mounted:
                mount_cmd = f'sshfs root@{ip}:{remote_path} {local_path} -p {port}'

        if not_mounted:
            result = await asyncio.to_thread(subprocess.run, mount_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                self.info("Mounted successfully!")
            else:
                self.info(f"Failed to mount. Error: {result.stderr}")
        else:
            self.info("Already mounted.")

    async def is_mounted(self):
        return False

        # # TODO
        # local_path = None
        # if sys.platform.startswith('win'):
        #     self.info("Windows not supported yet.")
        #     return False
        # elif sys.platform.startswith('darwin'):
        #     return await asyncio.to_thread(subprocess.run, f'mount | grep -q "{local_path}"', shell=True, check=False).returncode == 0
        # else:
        #     return await asyncio.to_thread(subprocess.run, f'mountpoint -q {local_path}', shell=True, check=False).returncode == 0

    async def is_process_running(self, process_name):
        return await self.run(f"ps aux | grep {process_name} | grep -v grep", return_output=True)

    async def put_any(self, source, target,
                      forbid_rclone=False,
                      force_rclone=False,
                      forbid_recursive=False,
                      rclone_excludes=None,
                      rclone_includes=None,
                      force=False):
        if rclone_excludes is None: rclone_excludes = []
        if rclone_includes is None: rclone_includes = []

        source = Path(source)
        target = Path(target)

        if force_rclone or source.is_dir():
            if force_rclone or not forbid_rclone:
                await self.put_rclone(source, target, forbid_recursive, rclone_excludes, rclone_includes)
            else:
                await self.put_dir(source.as_posix(), target.as_posix())
        else:
            await self.put_file(source.as_posix(), target.as_posix())

    async def put_dir(self, src, dst):
        """Uploads the contents of the source directory to the target path."""
        src = Path(src)
        dst = Path(dst)

        self.info(f"Starting directory upload: {src} -> {dst}")

        # Skip certain directories
        if src.stem in ["__pycache__", ".idea"]:
            self.info(f"Skipping directory: {src}")
            return

        # Ensure the destination directory exists
        self.info(f"Creating destination directory: {dst}")
        await self.run(f'mkdir -p {dst.as_posix()}')

        # Gather all files and their info
        files_to_check = []
        for item in os.listdir(src):
            if '.git' in item:
                self.info(f"Skipping .git item: {item}")
                continue

            src_path = src / item
            dst_path = dst / item

            if src_path.is_file():
                files_to_check.append((src_path, dst_path))
                self.info(f"Adding file to check: {src_path}")

        # Batch check files
        if files_to_check:
            self.info("Performing batch file check")
            check_cmd = " && ".join([f'stat -c "%s %Y" "{p[1].as_posix()}" 2>/dev/null || echo "NOT_FOUND"' for p in files_to_check])
            self.info(f"Batch check command: {check_cmd}")
            result = await self.run(check_cmd)
            remote_stats = result.stdout.strip().split('\n')
            self.info(f"Received {len(remote_stats)} remote file stats")

        # Prepare files for upload
        files_to_upload = []
        large_files = []
        for (src_path, dst_path), remote_stat in zip(files_to_check, remote_stats):
            file_size = src_path.stat().st_size
            local_mtime = int(src_path.stat().st_mtime)

            self.info(f"Checking file: {src_path}")
            self.info(f"  Local size: {file_size}, Local mtime: {local_mtime}")

            if remote_stat != "NOT_FOUND":
                remote_size, remote_mtime = map(int, remote_stat.split())
                self.info(f"  Remote size: {remote_size}, Remote mtime: {remote_mtime}")
                if file_size == remote_size and local_mtime <= remote_mtime:
                    self.info(chalk.gray(f"{src_path} -> {dst_path} (unchanged)"))
                    continue
            else:
                self.info("  File not found on remote")

            if file_size > self.max_size:
                self.info(f"  Large file, will handle separately: {src_path}")
                large_files.append((src_path, dst_path))
            else:
                self.info(f"  Adding to upload list: {src_path}")
                files_to_upload.append((src_path, dst_path))

        # Handle large files
        self.info(f"Processing {len(large_files)} large files")
        for src_path, dst_path in large_files:
            if self.enable_urls and src_path.name in self.urls:
                url = self.urls[src_path.name]
                if url:
                    self.info(f"Downloading large file from URL: {url} -> {dst_path}")
                    await self.run(f"wget {url} -O {dst_path.as_posix()}", log_output=True)
                    if self.enable_print_download:
                        self.print_download(src_path.name, url, dst_path, url)
                else:
                    self.info(chalk.red(f"<!> Invalid URL '{url}' for {src_path.name}"))
            else:
                self.info(chalk.red(f"<!> File too big, skipping: {src_path.name}"))

        # Batch upload small files
        if files_to_upload:
            self.info(f"Preparing to upload {len(files_to_upload)} files in batch")
            with io.BytesIO() as tar_buffer:
                with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
                    for src_path, _ in files_to_upload:
                        # self.info(f"Adding to tar: {src_path}")
                        tar.add(src_path, arcname=src_path.name)
                tar_buffer.seek(0)

                self.info(f"Uploading tar archive to {dst}")
                await self.run_safe(f'mkdir -p {dst.as_posix()}')
                path = dst / 'temp.tar.gz'
                await asyncssh.scp(tar_buffer, (self.connection, f'{path.as_posix()}'))

                self.info("Extracting tar archive on remote")
                await self.connection.run(f'cd {dst.as_posix()} && tar xzf temp.tar.gz && rm temp.tar.gz')

            for src_path, dst_path in files_to_upload:
                self.info(chalk.green(f"{src_path} -> {dst_path}"))

        # Recursively handle subdirectories
        for item in os.listdir(src):
            src_path = src / item
            if src_path.is_dir():
                self.info(f"Recursing into subdirectory: {src_path}")
                await self.put_dir(src_path, dst / item)

        self.info(f"Completed directory upload: {src} -> {dst} [put_dir / scp-tar]")

    async def put_file(self, src, dst):
        """Uploads a file to the target path."""
        src = Path(src)
        dst = Path(dst)

        file_size = src.stat().st_size

        # Check if file exists and compare sizes
        result = await self.connection.run(f'stat -c "%s %Y" {dst.as_posix()}')
        if result.exit_status == 0:
            remote_size, remote_mtime = map(int, result.stdout.split())
            local_mtime = int(src.stat().st_mtime)

            if file_size == remote_size and local_mtime <= remote_mtime:
                self.info(chalk.gray(f"{src} -> {dst} (unchanged)"))
                return

        if file_size > self.max_size:
            if self.enable_urls and src.name in self.urls:
                url = self.urls[src.name]
                if url:
                    await self.run(f"wget {url} -O {dst}", log_output=True)
                    if self.enable_print_download:
                        self.print_download(src.name, url, dst, url)
                    return
                else:
                    self.info(chalk.red("<!> Invalid URL '{url}' for "), src.name)

            self.info(chalk.red("<!> File too big, skipping"), src.name)
            return

        if self.enable_print_upload:
            self.print_upload(src.name, src.parent, dst.parent)

        try:
            await self.connection.run(f'mkdir -p {dst.parent.as_posix()}')
            await asyncssh.scp(src, (self.connection, dst.as_posix()))
            self.info(chalk.green(f"{src} -> {dst} [put-file / scp]"))
        except asyncssh.Error as e:
            self.error(f"SCP upload failed: {str(e)}")

    async def get_file(self, src, dst):
        """Downloads a file from the target path using SCP."""
        src = Path(src)
        dst = Path(dst)
        try:
            await asyncssh.scp((self.connection, src.as_posix()), dst.as_posix())
            self.info(chalk.green(f"{src} -> {dst}"))
        except asyncssh.Error as e:
            self.error(f"SCP download failed: {str(e)}")

    async def put_rclone(self, source, target, forbid_recursive, rclone_excludes, rclone_includes, print_cmd=True, print_output=True):
        start_time = time.time()

        source = Path(source)
        target = Path(target)

        ssh_key_path = os.path.expanduser('~/.ssh/id_rsa')
        # self.info(f"Using SSH key: {ssh_key_path}")

        # Create temporary rclone config file
        async with aiofiles.tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.conf') as temp_config:
            config_content = f"""[sftp]
    type = sftp
    host = {self.ip}
    port = {self.port}
    user = root
    key_file = {ssh_key_path}
    """
            await temp_config.write(config_content)
            temp_config_path = temp_config.name
        # self.info(f"Created temporary rclone config at: {temp_config_path}")

        # Prepare rclone flags
        flags = [
            # '--use-mmap',
            # '--delete-excluded',
            '--checkers', '50',  # Increase number of checkers for potentially faster processing
            '--transfers', '64',  # Increase number of concurrent transfers
            # '--sftp-set-modtime',
            # '--fast-list',  # Use recursive list if possible, can speed up large transfers
            '--buffer-size', '2M',  # Increase buffer size for potentially better performance
            '--size-only',  # Only check file size, not modtime
            # '-v',
            '-q',
            '--stats-one-line'
        ]

        if forbid_recursive:
            flags.extend(['--max-depth', '1'])
            # self.info("Recursive transfer forbidden, max depth set to 1")

        for exclude in rclone_excludes: flags.extend(['--exclude', exclude])
        for include in rclone_includes: flags.extend(['--include', include])

        # Prepare rclone command
        rclone_cmd = [
                         "rclone",
                         "--config", temp_config_path,
                         "copy",
                         source.as_posix(),
                         f"sftp:{target.as_posix()}",
                     ] + flags

        if print_cmd:
            self.info("\n")
            self.info(f"{source.as_posix()} -> {target.as_posix()} (rclone)")

        if rclone_excludes: self.info(f"Exclude patterns: {' '.join(rclone_excludes)}")
        if rclone_includes: self.info(f"Include patterns: {' '.join(rclone_includes)}")
        # self.info_cmd(" ".join(rclone_cmd), False)

        try:
            process = await asyncio.create_subprocess_exec(
                *rclone_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            if print_output:
                # Process output in real-time
                async def log_output(stream, prefix):
                    async for line in stream:
                        self.info(f"{prefix}: {line.decode().strip()}")

                await asyncio.gather(
                    log_output(process.stdout, "STDOUT"),
                    log_output(process.stderr, "STDERR")
                )

            await process.wait()

            if process.returncode != 0:
                self.info(f"Error running rclone. Exit code: {process.returncode}")

        finally:
            os.unlink(temp_config_path)
            # self.info(f"Deleted temporary rclone config: {temp_config_path}")

        end_time = time.time()
        duration = end_time - start_time
        # self.info(f"rclone transfer finished. Duration: {duration:.2f} seconds")

    async def get_rclone(self, source, target, forbid_recursive, rclone_excludes, rclone_includes, print_cmd=True, print_output=True):
        source = Path(source)
        target = Path(target)

        ssh_key_path = os.path.expanduser('~/.ssh/id_rsa')

        async with aiofiles.tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.conf') as temp_config:
            await temp_config.write(f"""[sftp]
    type = sftp
    host = {self.ip}
    port = {self.port}
    user = root
    key_file = {ssh_key_path}
    """)
            temp_config_path = temp_config.name

        flags = ['--progress']
        if forbid_recursive:
            flags.extend(['--max-depth', '1'])

        for exclude in rclone_excludes: flags.extend(['--exclude', exclude])
        for include in rclone_includes: flags.extend(['--include', include])

        flags.extend([
            '--update',
            '--use-mmap',
            '--delete-excluded',
            '--checkers', '50',
            '--transfers', '64',
            '--sftp-set-modtime',
        ])

        rclone_cmd = [
                         "rclone",
                         "--config", temp_config_path,
                         "copy",
                         "--update",
                         f"sftp:{source.as_posix()}",
                         target.as_posix(),
                         "-v"
                     ] + flags

        if print_cmd:
            self.info_cmd(" ".join(rclone_cmd))
            self.info(f'{source} -> {target}')

        try:
            process = await asyncio.create_subprocess_exec(
                *rclone_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                if print_output:
                    self.info(stdout.decode())
            else:
                self.info(f"Error running rclone. Exit code: {process.returncode}")
                self.info("Standard output:")
                self.info(stdout.decode())
                self.info("Standard error:")
                self.info(stderr.decode())
        finally:
            os.unlink(temp_config_path)

    def print_upload(self, item, source, target):
        self.info(f"Uploading {os.path.join(source, item)} to {target}")

    def print_download(self, item, source, target, url):
        self.info(f"Downloading {item} from {source} to {target}")

    async def exists(self, path):
        try:
            if isinstance(path, Path):
                path = path.as_posix()
            await self.sftp.lstat(path)
            return True
        except asyncssh.SFTPError:
            return False

    async def mkdir(self, path, mode=511, ignore_existing=False):
        """Augments mkdir by adding an option to not fail if the folder exists"""
        if isinstance(path, Path):
            path = path.as_posix()

        try:
            await self.run(f'mkdir -p {path}')
        except asyncssh.SFTPError:
            if ignore_existing:
                pass
            else:
                raise

    # @staticmethod
    # def print_cmd(cmd):
    #     log.info(f"> {cmd}")

    # @staticmethod
    # def print_header(header):
    #     log.info(f"\n{'=' * 20}\n{header}\n{'=' * 20}")
    def info_cmd(self, text, newline=True):
        if newline:
            self.info("\n")
        self.info(f"> {text}")

# import logging
# import subprocess
# import sys
# from pathlib import Path
#
# import asyncssh
#
# import deploy_utils
# from src.lib.loglib import print_cmd
#
# log = logging.getLogger(__name__)
#
#
# class SSHClient:
#     """
#     Custom asynchronous SSH client for Discore deployments.
#
#     This class provides functionality similar to the original DiscoreSSHClient,
#     but uses asyncssh for asynchronous SSH operations.
#
#     Methods:
#         connect: Establishes an SSH connection.
#         run: Executes a command on the remote instance.
#         file_exists: Checks if a file exists on the remote instance.
#         mount: Mounts a remote directory locally.
#     """
#
#     def __init__(self):
#         self.connection = None
#         self.sftp = None
#
#     async def connect(self, hostname, port, username, password=None, key_filename=None):
#         """Establish an SSH connection."""
#         self.connection = await asyncssh.connect(
#             hostname, port=port,
#             username=username, password=password,
#             client_keys=key_filename,
#             known_hosts=None  # Note: Consider using known_hosts in production
#         )
#         self.sftp = await self.connection.start_sftp_client()
#
#     async def run(self, cm, cwd=None, *, log_output=False):
#         """
#         Execute a command on the remote instance.
#         """
#         cm = cm.replace("'", '"')
#         if cwd is not None:
#             cm = f"cd {cwd}; {cm}"
#
#         cm = f"/bin/bash -c '{cm}'"
#
#         print_cmd(cm, log)
#         result = await self.connection.run(cm, check=True)
#
#         if log_output:
#             log.info(result.stdout)
#
#         return result.stdout
#
#     async def file_exists(self, path):
#         """
#         Check if a file exists on the remote instance.
#         """
#         try:
#             await self.sftp.stat(path)
#             return True
#         except asyncssh.SFTPError:
#             return False
#
#     async def mount(self, local_path, remote_path, ip, port):
#         """
#         Mount a remote directory locally
#         """
#         local_path = Path(local_path)
#         local_path.mkdir(parents=True, exist_ok=True)
#
#         if sys.platform.startswith('win'):
#             log.info("Windows not supported yet.")
#             return
#         elif sys.platform.startswith('darwin'):
#             not_mounted = subprocess.run(f'mount | grep -q "{local_path}"', shell=True).returncode != 0
#             if not_mounted:
#                 mount_cmd = f'sshfs root@{ip}:{remote_path} {local_path} -p {port} -o volname=Discore'
#         else:
#             not_mounted = subprocess.run(f'mountpoint -q {local_path}', shell=True).returncode != 0
#             if not_mounted:
#                 mount_cmd = f'sshfs root@{ip}:{remote_path} {local_path} -p {port}'
#
#         if not_mounted:
#             deploy_utils.print_header("Mounting with sshfs...")
#             result = subprocess.run(mount_cmd, shell=True, capture_output=True, text=True)
#             if result.returncode == 0:
#                 log.info("Mounted successfully!")
#             else:
#                 log.info(f"Failed to mount. Error: {result.stderr}")
#         else:
#             log.info("Already mounted.")
#
#     @staticmethod
#     def print_cmd(cmd):
#         log.info(f"> {cmd}")
#
#     @staticmethod
#     def print_header(header):
#         log.info(f"\n{'=' * 20}\n{header}\n{'=' * 20}")
