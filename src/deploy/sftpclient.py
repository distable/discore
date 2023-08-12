import subprocess
from pathlib import Path

import paramiko
import os

from yachalk import chalk

from src.lib.printlib import print_cmd


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
            'vae.vae.pt'  : '',
        }
        self.ssh = None
        self.enable_urls = True
        self.enable_print_upload = False
        self.enable_print_download = True
        self.rsync = True
        self.ip = None
        self.port = None
        self.print_rsync = True

    def put_any(self, source, target, forbid_rsync=False, force_rsync=False, forbid_recursive=False, rsync_excludes=None, rsync_includes=None, force=False):
        if rsync_excludes is None:
            rsync_excludes = []
        if rsync_includes is None:
            rsync_includes = []
        # if source.is_file():
        #     self.put_file(source, target)
        #     return

        source = Path(source)
        target = Path(target)

        # Rsync automatically for directories, otherwise the rsync overhead is too much
        if source.is_dir() or force_rsync:
            self.put_rsync(source, target, forbid_recursive, rsync_excludes, rsync_includes)
            # self.mkdir(target.as_posix(), ignore_existing=True)
            # self.put_dir(source.as_posix(), target.as_posix())
        else:
            self.put_file(source.as_posix(), target.as_posix())

    def put_dir(self, src, dst, *, forbid_rsync=False):
        """
        Uploads the contents of the source directory to the target path. The
        target directory needs to exists. All subdirectories in source are
        created under target.
        """
        import yachalk as chalk
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

                    from yachalk import chalk
                    print(chalk.red("<!> File too big, skipping"), item)
                    continue

                if self.enable_print_upload:
                    self.print_upload(item, src, dst)

                self.put(os.path.join(src, item), '%s/%s' % (dst, item))
            else:
                self.mkdir('%s/%s' % (dst, item), ignore_existing=True)
                self.put_dir(os.path.join(src, item), '%s/%s' % (dst, item), forbid_rsync=forbid_rsync)

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
            self.put(src.as_posix(), dst.as_posix())
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

    def put_rsync(self, source, target, forbid_recursive, rsync_excludes, rsync_includes):
        source = Path(source)
        target = Path(target)

        flags = ''
        if self.print_rsync:
            flags = 'v'
        if not forbid_recursive:
            flags += 'r'
        flags2 = ""
        for exclude in rsync_excludes:
            flags2 += f" --exclude='{exclude}'"
        for include in rsync_includes:
            flags2 += f" --include='{include}'"
        # if len(rsync_includes) > 0 and len(rsync_excludes) == 0:
        #     flags2 += " --exclude='*'"

        target = target.parent
        source = source.as_posix()
        target = target.as_posix()
        if target[-1] != '/':
            target += '/'  # very important for rsync

        cm = f"rsync -rlptgoDz{flags} --exclude '*/.*' -e 'ssh -p {self.port}' {source} root@{self.ip}:{target} {flags2}"
        print_cmd(cm)
        # self.ssh.run(cm)
        print(f'{source} -> {target}')
        os.system(cm)
        # subprocess.run(cm, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)

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
