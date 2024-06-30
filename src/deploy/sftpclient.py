import subprocess
import tempfile
from pathlib import Path

import paramiko
import os

from yachalk import chalk

from src.lib.loglib import print_cmd


def sshexec(ssh, cmd, with_printing=True):
    if with_printing:
        print(f'$ {cmd}')
    stdin, stdout, stderr = ssh.exec_command(cmd)
    stdout.channel.set_combine_stderr(True)
    ret = []
    for line in iter(stdout.readline, ""):
        # print(line, end="")
        ret.append(line)
    if with_printing:
        from yachalk import chalk
        print(chalk.dim(''.join(ret)))

    # return ret


class SFTPClient(paramiko.SFTPClient):
    def __init__(self, *args, **kwargs):
        super(SFTPClient, self).__init__(*args, **kwargs)
        self.max_size = 1000000000
        self.urls = {
            'sd-v1-5.ckpt': '',
            'vae.vae.pt': '',
        }
        self.ssh = None
        self.enable_urls = True
        self.enable_print_upload = False
        self.enable_print_download = True
        self.rclone = True
        self.ip = None
        self.port = None
        self.print_rsync = True

    def put_any(self, source, target,
                forbid_rclone=False,
                force_rclone=False,
                forbid_recursive=False,
                rclone_excludes=None,
                rclone_includes=None,
                force=False):
        if rclone_excludes is None:
            rclone_excludes = []
        if rclone_includes is None:
            rclone_includes = []

        source = Path(source)
        target = Path(target)

        if force_rclone or source.is_dir():
            if force_rclone or not forbid_rclone:
                self.put_rclone(source, target, forbid_recursive, rclone_excludes, rclone_includes)
            else:
                self.put_dir(source.as_posix(), target.as_posix())
        else:
            self.put_file(source.as_posix(), target.as_posix())

    def put_dir(self, src, dst):
        """
        Uploads the contents of the source directory to the target path, individually one by one.
        The target directory needs to exists. All subdirectories in source are created under target.
        """
        if Path(src).stem in ["__pycache__", ".idea"]:
            return

        self.mkdir(dst, ignore_existing=True)
        for item in os.listdir(src):
            if '.git' in item:
                continue

            if os.path.isfile(os.path.join(src, item)):
                if self.exists(os.path.join(dst, item)):
                    continue

                # If size is above self.max_size (in bytes)
                if os.path.getsize(os.path.join(src, item)) > self.max_size:
                    if self.enable_urls and item in self.urls:
                        url = self.urls[item]
                        if url:
                            sshexec(self.ssh, f"wget {url} -O {os.path.join(dst, item)}")
                            if self.enable_print_download:
                                self.print_download(item, url, dst, url)
                            continue
                        else:
                            print(chalk.red("<!> Invalid URL '{url}' for "), item)

                    print(chalk.red("<!> File too big, skipping"), item)
                    continue

                if self.enable_print_upload:
                    self.print_upload(item, src, dst)

                print(chalk.green(f"{src}/{item} -> {dst}/{item}"))
                self.put(os.path.join(src, item), '%s/%s' % (dst, item))
            else:
                print(chalk.green(f"Creating directory {dst}/{item}"))
                self.mkdir('%s/%s' % (dst, item), ignore_existing=True)

                print(chalk.green(f"{src} -> {dst}"))
                self.put_dir(os.path.join(src, item), '%s/%s' % (dst, item))

    def put_file(self, src, dst):
        """
        Uploads a file to the target path. The target path needs to include
        the target filename.
        Args:
            src: The path to the local file.
            dst: The path to the remote file.
        """
        src = Path(src)
        dst = Path(dst)
        if not self.ssh.file_exists(dst):
            print(chalk.green(f"{src} -> {dst}"))
            self.put(src.as_posix(), dst.as_posix().replace('\\', '/'))
        else:
            # Check mtime to see if src is newer
            try:
                stat = self.stat(dst.as_posix())
                newer = int(src.stat().st_mtime) > int(stat.st_mtime)
            except FileNotFoundError:
                newer = True

            if newer:
                print(chalk.yellow(f"{src} -> {dst}"))
                self.mkdir(dst.parent.as_posix(), ignore_existing=True)
                self.put(src.as_posix(), dst.as_posix())
            else:
                print(chalk.dim(f"{src} -> {dst}"))

    def get_file(self, src, dst):
        """
        Downloads a file from the target path. The target path needs to include
        the target filename.

        Args:
            src: The source path on the remote host.
            dst: The destination path on the local host.
        """
        src = Path(src)
        dst = Path(dst)
        if not self.ssh.file_exists(src):
            print(chalk.red(f"{src} -> {dst}"))
            self.get(src.as_posix(), dst.as_posix())
        else:
            # Check mtime to see if src is newer
            try:
                stat = self.stat(src.as_posix())
                newer = int(dst.stat().st_mtime) < int(stat.st_mtime)
            except FileNotFoundError:
                newer = True

            if newer:
                print(chalk.yellow(f"{src} -> {dst}"))
                self.mkdir(dst.parent.as_posix(), ignore_existing=True)
                self.get(src.as_posix(), dst.as_posix())
            else:
                print(chalk.dim(f"{src} -> {dst}"))


    def put_rclone(self, source, target, forbid_recursive, rclone_excludes, rclone_includes):
        source = Path(source)
        target = Path(target)

        # Assume the SSH private key is stored in the default location
        ssh_key_path = os.path.expanduser('~/.ssh/id_rsa')

        # Create a temporary rclone config file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.conf') as temp_config:
            temp_config.write(f"""[sftp]
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

        for exclude in rclone_excludes:
            flags.extend(['--exclude', exclude])
        for include in rclone_includes:
            flags.extend(['--include', include])

        flags.extend([
            '--update',
            '--use-mmap',
            '--delete-excluded',
            '--checkers', '8',
            '--transfers', '4',
            '--sftp-set-modtime',
        ])

        # Construct the rclone command using the temporary config
        rclone_cmd = [
                         "rclone",
                         "--config", temp_config_path,
                         "copy",
                         source.as_posix(),
                         f"sftp:{target.as_posix()}",
                         "-v"  # Add verbosity
                     ] + flags

        print_cmd(" ".join(rclone_cmd))
        print(f'{source} -> {target}')

        try:
            result = subprocess.run(
                rclone_cmd,
                check=True,
                text=True,
                capture_output=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running rclone. Exit code: {e.returncode}")
            print("Standard output:")
            print(e.stdout)
            print("Standard error:")
            print(e.stderr)
        finally:
            # Clean up the temporary config file
            os.unlink(temp_config_path)

    def print_upload(self, item, source, target):
        print(f"Uploading {os.path.join(source, item)} to {target}")

    def print_download(self, item, source, target, url):
        print(f"Downloading {item} from {source} to {target}")

    def exists(self, path):
        try:
            if isinstance(path, Path):
                path = path.as_posix()
            # print(f'check if {path} exists', self.stat(path))
            if self.lstat(path) is not None:
                return True
            return False
        except:
            return False

    def mkdir(self, path, mode=511, ignore_existing=False):
        """
        Augments mkdir by adding an option to not fail if the folder exists
        """
        try:
            self.ssh.run('mkdir -p %s' % path)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise
