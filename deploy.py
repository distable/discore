# """
# Note: in this context src always refer to local machine, dst always refer to remote machine
# """
#
#
# import os
# import subprocess
# import sys
# import time
# from pathlib import Path
#
# from paramiko.ssh_exception import NoValidConnectionsError, SSHException
# from yachalk import chalk
#
# import jargs
# from src.classes.common import setup_ctrl_c
# from src.lib.corelib import shlexrun
# from src.lib.loglib import print_cmd
# from jargs import args, argv
# from src.classes import paths
# from src import renderer
# import paramiko
# import userconf
# import json
# import threading
#
# disk_space = '32'
# docker_image = 'pytorch/pytorch'
#
# # logging.basicConfig()
# # logging.getLogger("paramiko").setLevel(logging.WARNING) # for example
#
# ssh = None
#
# apt_packages = [
#     'python3-venv',
#     'libgl1',
#     'zip',
#     'ffmpeg',
#     'gcc'
# ]
#
# # User files/directories to copy at the start of a deployment
# deploy_upload_paths = [('requirements-vastai.txt', 'requirements.txt'),
#                        'discore.py',
#                        'deploy.py',
#                        'jargs.py',
#                        paths.userconf_name,
#                        paths.scripts_name,
#                        paths.src_plugins_name,
#                        paths.src_name,
#                        # paths.plug_repos_name
#                        ]
# deploy_upload_paths_sftp = [paths.plug_res_name]
# deploy_upload_paths_sftp = []
#
# deploy_upload_blacklist_paths = [
#     "video.mp4",
#     "video__*.mp4",
#     "*.jpg",
#     "__pycache__"
#     # "*.npy",
# ]
#
# rsync_job_download_exclusion = [
#     "video.mp4",
#     "video__*.mp4",
#     "script.py",
#     "*.npy",
#     "__pycache__/*"
# ]
#
# rsync_job_upload_paths = [
#     paths.scripts,
#     paths.code_core / 'party',
# ]
#
# vastai_python_bin = '/opt/conda/bin/python3'
#
#
# # Commands to run in order to setup a deployment
# def get_deploy_commands(clonepath):
#     return [
#         ['git', 'clone', '--recursive', 'https://github.com/distable/discore', clonepath],
#         ['git', '-C', clonepath, 'submodule', 'update', '--init', '--recursive'],
#     ]
#
#
# def deploy_local():
#     import shutil
#     import platform
#
#
#     # A test 'provider' which attempts to do a clean clone of the current installation
#     # 1. Clone the repo to ~/discore_deploy/
#     src = paths.root
#     dst = Path.home() / "discore_deploy"
#
#     shutil.rmtree(dst.as_posix())
#
#     cmds = get_deploy_commands(dst.as_posix())
#     for cmd in cmds:
#         subprocess.run(cmd)
#
#     for file in deploy_upload_paths:
#         if isinstance(file, tuple):
#             shutil.copyfile(src / file[0], dst / file[1])
#         if isinstance(file, str):
#             shutil.copyfile(src / file, dst / file)
#
#     # 3. Run ~/discore_deploy/discore.py
#     if platform.system() == "Linux":
#         subprocess.run(['chmod', '+x', dst / 'discore.py'])
#
#     subprocess.run([dst / 'discore.py', '--upgrade'])
#
# def print_header(string):
#     print("")
#     print("----------------------------------------")
#     print(chalk.green(string))
#     print("----------------------------------------")
#     print("")
#
# def deploy_vastai():
#     """
#     Deploy onto cloud.
#     """
#
#     global ssh
#
#     session = jargs.get_discore_session()
#
#     is_fresh_install = False
#
#     # 1. List the available machines with vastai api
#     # 2. Prompt the user to choose one
#     # 3. Connect to it with SSH
#     # 4. Git clone the core repository
#     # 5. Upload our userconf
#     # 6. Launch discore
#     # 7. Connect our core to it
#     # ----------------------------------------
#     vastpath = paths.root / 'vast'
#     if not vastpath.is_file():
#         if sys.platform == 'linux':
#             os.system("wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;")
#         else:
#             print("Vastai deployment is only supported on Linux, may be broken on windows. it will probably break, good luck")
#
#     # Run vast command and parse its output
#
#     def fetch_offers():
#         import userconf
#         out = subprocess.check_output(['python3', vastpath.as_posix(), 'search', 'offers', f"{args.vastai_search or userconf.vastai_default_search} disk_space>{disk_space} verified=true"]).decode('utf-8')
#         # Example output:
#         # ID       CUDA  Num  Model     PCIE  vCPUs    RAM  Disk   $/hr    DLP    DLP/$  NV Driver   Net_up  Net_down  R     Max_Days  mach_id  verification
#         # 5563966  11.8  14x  RTX_3090  12.7  64.0   257.7  1672   5.8520  341.8  58.4   520.56.06   23.1    605.7     99.5  22.4      6294     verified
#         # 5412452  12.0   4x  RTX_3090  24.3  32.0   257.8  751    1.6800  52.0   30.9   525.60.13   714.7   792.1     99.8  7.8       1520     verified
#         # 5412460  12.0   2x  RTX_3090  24.3  16.0   257.8  376    0.8400  26.0   30.9   525.60.13   714.7   792.1     99.8  7.8       1520     verified
#         # 5412453  12.0   1x  RTX_3090  24.3  8.0    257.8  188    0.4200  13.0   30.9   525.60.13   714.7   792.1     99.8  7.8       1520     verified
#         # 5412458  12.0   8x  RTX_3090  24.3  64.0   257.8  1502   3.3600  103.9  30.9   525.60.13   714.7   792.1     99.8  7.8       1520     verified
#
#         lines = out.splitlines()
#         offers = list()  # list of dict
#         for i, line in enumerate(lines):
#             line = line.split()
#             if len(line) == 0: continue
#             if i == 0: continue
#             offers.append(dict(id=line[0], cuda=line[1], num=line[2], model=line[3], pcie=line[4], vcpus=line[5], ram=line[6], disk=line[7], price=line[8], dlp=line[9], dlpprice=line[10], nvdriver=line[11], netup=line[12], netdown=line[13], r=line[14], maxdays=line[15], machid=line[16], verification=line[17]))
#
#         return offers
#
#     def fetch_balance():
#         out = subprocess.check_output(['python3', vastpath.as_posix(), 'show', 'invoices']).decode('utf-8')
#         # Example output (only the last line)
#         # Current:  {'charges': 0, 'service_fee': 0, 'total': 0, 'credit': 6.176303554744997}
#
#         lines = out.splitlines()
#         s = lines[-1]
#         s = s.replace('Current:  ', '')
#         s = s.replace("'", '"')
#         o = json.loads(s)
#         return float(o['credit'])
#
#
#     def fetch_instances():
#         out = subprocess.check_output(['python3', vastpath.as_posix(), 'show', 'instances']).decode('utf-8')
#         # Example output:
#         # ID       Machine  Status  Num  Model     Util. %  vCPUs    RAM  Storage  SSH Addr      SSH Port  $/hr    Image            Net up  Net down  R     Label
#         # 5717760  5721     -        1x  RTX_3090  -        6.9    128.7  17       ssh4.vast.ai  37760     0.2436  pytorch/pytorch  75.5    75.1      98.5  -
#
#         lines = out.splitlines()
#         instances = list()
#         for i, line in enumerate(lines):
#             line = line.split()
#             if len(line) == 0: continue
#             if i == 0: continue
#             instances.append(dict(id=line[0], machine=line[1], status=line[2], num=line[3], model=line[4], util=line[5], vcpus=line[6], ram=line[7], storage=line[8], sshaddr=line[9], sshport=line[10], price=line[11], image=line[12], netup=line[13], netdown=line[14], r=line[15], label=line[16]))
#         return instances
#
#     from yachalk import chalk
#
#     def print_offer(e, i):
#         print(chalk.green(f'{i + 1} - {e["model"]} - {e["num"]} - {e["dlp"]} - {e["netdown"]} Mbps - {e["price"]} $/hr - {e["dlpprice"]} DLP/HR'))
#
#     def print_instance(e):
#         print(chalk.green_bright(f'{i + 1} - {e["model"]} - {e["num"]} - {e["netdown"]} Mbps -  {e["price"]} $/hr - {e["status"]}'))
#
#     print("Deployed instances:")
#     instances = fetch_instances()
#     selected_id = None  # The instance to boot up
#     for i, e in enumerate(instances):
#         print_instance(e)
#     print("")
#
#     if args.vastai_list:
#         return
#
#     # 1. Choose or create instance
#     # ----------------------------------------
#     if len(instances) >= 1:
#         selected_id = instances[0]['id']
#     # while len(instances) >= 1:
#     #     try:
#     #         s = input("Choose an instance or type 'n' to create a new one: ")
#     #         if s == 'n':
#     #             break;
#     #         selected_id = instances[int(s) - 1]['id']
#     #         break
#     #     except:
#     #         print("Invalid choice")
#
#     if selected_id is None:
#         # Create new instance
#         # ----------------------------------------
#         while True:
#             offers = fetch_offers()
#
#             # Sort the offers by price descending
#             offers = sorted(offers, key=lambda e: float(e['price']), reverse=True)
#
#             # Print the list of machines
#             for i, e in enumerate(offers):
#                 print_offer(e, i)
#
#             # Ask user to choose a machine, keep asking until valid choice
#             print("")
#             try:
#                 choice = input("Enter the number of the machine you want to use: ")
#                 choice = int(choice)
#                 if 1 <= choice <= len(offers):
#                     print_offer(offers[choice - 1], choice - 1)
#                     print()
#                     break
#             except:
#                 print("Invalid choice. Try again, or type r to refresh the list and see again.")
#
#         # Create the machine
#         # Example command: ./vast create instance 36842 --image vastai/tensorflow --disk 32
#         selected_id = offers[choice - 1]['id']
#         out = subprocess.check_output(['python3', vastpath.as_posix(), 'create', 'instance', selected_id, '--image', docker_image, '--disk', disk_space]).decode('utf-8')
#         if 'Started.' not in out:
#             print("Failed to create instance:")
#             print(out)
#             return
#
#         time.sleep(3)
#
#         new_instances = fetch_instances()
#         # Diff between old and new instances
#         new_instances = [e for e in new_instances if e['id'] not in [e['id'] for e in instances]]
#
#         if len(new_instances) != 1:
#             print("Failed to create instance, couldn't find the new instance by diffing.")
#             return
#
#         selected_id = new_instances[0]['id']
#         is_fresh_install = True
#
#         print(f"Successfully created instance {selected_id}!")
#
#     def wait_for_instance(id):
#         printed_loading = False
#         ins = None
#         while ins is None or ins['status'] != 'running':
#             all_ins = [i for i in fetch_instances() if i['id'] == id]
#             if len(all_ins) > 0:
#                 ins = all_ins[0]
#
#                 status = ins['status']
#                 if ins is not None and status == 'running':
#                     return ins
#
#                 if ins is not None and status != 'running':
#                     if not printed_loading:
#                         print("")
#                         print(f"Waiting for {id} to finish loading (status={status})...")
#                         printed_loading = True
#
#             time.sleep(3)
#
#         time.sleep(3)
#
#     if args.vastai_delete:
#         print(f"Deleting Vast.ai instance {selected_id} in 5 seconds ...")
#         time.sleep(5)
#         subprocess.check_output(['python3', vastpath.as_posix(), 'destroy instance', str(selected_id)]).decode('utf-8')
#         print("All done!")
#         return
#
#     if args.vastai_reboot:
#         print(f"Rebooting Vast.ai instance {selected_id} ...")
#         subprocess.check_output(['python3', vastpath.as_posix(), 'reboot instance', str(selected_id)]).decode('utf-8')
#
#     # 2. Wait for instance to be ready
#     # ----------------------------------------
#     instance = wait_for_instance(selected_id)
#
#     # 3. Connections
#     # ----------------------------------------
#     user = 'root'
#     ip = instance['sshaddr']
#     port = instance['sshport']
#
#     src = paths.root
#     dst = Path("/workspace/discore_deploy")
#     src_session = session.dirpath
#     dst_session = dst / 'sessions' / session.dirpath.stem
#     install_checkpath = "/root/.discore_installed"
#     ssh_cmd = f"ssh -p {port} {user}@{ip}"
#     kitty_cmd = f"kitty +kitten {ssh_cmd}"
#     discore_cmd = f"cd /workspace/discore_deploy/; {vastai_python_bin} {dst / 'discore.py'} {session.name} --cli --remote --unsafe --no_venv"
#
#     print("")
#     print('----------------------------------------')
#     print(chalk.green(f"Establishing connections {user}@{ip}:{port}..."))
#     print(ssh_cmd)
#     print(kitty_cmd)
#     print('----------------------------------------')
#     print("")
#
#     # SSH connection
#     # ----------------------------------------
#
#     ssh = DiscoreSSHClient()
#     ssh.load_host_keys(os.path.expanduser(os.path.join('~', '.ssh', 'known_hosts')))
#     ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#
#     num_connection_failure = 0
#     while True:
#         try:
#             # This can fail when the instance just launched
#             ssh.connect(ip, port=int(port), username='root')
#             break
#         except NoValidConnectionsError as e:
#             print(f"Failed to connect ({e}), retrying...")
#             time.sleep(3)
#             num_connection_failure += 1
#         except SSHException as e:
#             os.system(ssh_cmd)
#
#     if num_connection_failure > 0:
#         print(f"Successfully connected through SSH after {num_connection_failure} retries.")
#         if num_connection_failure > 10:
#             print("Jeez, why did that take so many tries?")
#     elif num_connection_failure == 0:
#         print("Successfully connected through SSH.")
#
#     # SFTP connection
#     # ----------------------------------------
#
#     from src.deploy.sftpclient import SFTPClient
#     sftp = SFTPClient.from_transport(ssh.get_transport())
#     sftp.max_size = 10 * 1024 * 1024
#     sftp.urls = []
#     if hasattr(userconf, 'deploy_urls'):
#         sftp.urls = userconf.deploy_urls
#     sftp.ssh = ssh
#     sftp.enable_urls = not args.vastai_no_download
#     sftp.ip = ip
#     sftp.port = port
#
#     if not is_fresh_install:
#         is_fresh_install = not sftp.exists(install_checkpath)
#
#     print("Is fresh install:", is_fresh_install)
#
#     # rm -rf existing & Clone
#     # ----------------------------------------
#     if not ssh.file_exists(dst) and not jargs.is_vastai_continue:
#         if ssh.file_exists(dst):
#             print(chalk.red("Removing old deploy..."))
#             time.sleep(3)
#             ssh.run(f"rm -rf {dst}")
#
#         print_header("Cloning")
#         cmds = get_deploy_commands(dst.as_posix())
#         for cmd in cmds:
#             ssh.run(' '.join(cmd))
#
#     if userconf.vastai_sshfs:
#         local_path = Path(userconf.vastai_sshfs_path).expanduser() / 'discore_deploy'
#         thread = threading.Thread(target=ssh.mount, args=(local_path, '/workspace/discore_deploy', ip, port))
#         thread.start()
#         # mount()
#
#     # Install system packages
#     # ----------------------------------------
#
#     if is_fresh_install:
#         print_header("Installing system packages ...")
#         for apt_package in apt_packages:
#             print(f"Installing {apt_package}...")
#             ssh.run(f"apt-get install {apt_package} -y")
#
#         ssh.run(f"chmod +x {dst / 'discore.py'}")
#         # ssh.run(ssh, f"rm -rf {dst / 'venv'}")
#
#     # Copy files
#     # ----------------------------------------
#     # if is_fresh_install or args.vastai_copy:
#     if not args.vastai_quick:
#         sync_files(src, dst, sftp)
#
#     # Copy res
#     # ----------------------------------------
#     # This is slow and usually only needed for the first time
#     if is_fresh_install or args.vastai_copy:
#         sync_res(src, dst, sftp)
#
#     # --shell
#     # ----------------------------------------
#     if args.shell:
#         open_shell(ssh)
#
#     # pip & plugin install
#     # ----------------------------------------
#     if is_fresh_install or args.vastai_upgrade:
#         print_header("Discore pip refresh")
#         subprocess.run(f"{ssh_cmd} '{discore_cmd}' --upgrade", shell=True)
#         # ssh.run(f"{discore_cmd} --upgrade")
#
#     # Uninstall numba
#     # subprocess.run(f"{ssh_cmd} {vastai_python_bin} -m pip uninstall numba -y", shell=True)
#
#
#     ssh.run(f"touch {install_checkpath}")
#
#     # Session state copies
#     # ----------------------------------------
#     if not args.vastai_quick:
#         copy_session(session, dst_session, sftp)
#
#     continue_work = True
#
#     # ----------------------------------------
#     # It's pizza time
#     # ----------------------------------------
#
#     def vastai_job():
#         # For some reason abs path to python3 doesnt work here with os.system
#         cmd = f"cd /workspace/discore_deploy/; {vastai_python_bin} {dst / 'discore.py'}"
#         # cmd = f"{vastai_python_bin} {dst / 'discore.py'}"
#
#         oargs = argv
#         jargs.remove_deploy_args(oargs)
#         cmd += f' {" ".join(oargs)}'
#         cmd += " --run -cli --remote --unsafe --no_venv"
#
#         print_header("Launching discore for work ...")
#         print("")
#
#         if is_fresh_install or args.vastai_upgrade or args.vastai_install:
#             cmd += " --install"
#
#         if args.vastai_trace:
#             cmd += " --trace"
#
#         nonlocal continue_work
#         while continue_work:
#             print("> " + cmd)
#             # ssh.run(cmd, log_output=True)
#             os.system(f"{ssh_cmd} '{cmd}'")
#             # sync_files(src, dst, sftp)
#         continue_work = False
#         renderer.request_stop = True
#
#
#     def balance_job():
#         """
#         Notify the user how much credit is left, every 0.25$
#         """
#         from desktop_notifier import DesktopNotifier
#
#         threshold = 0.25
#
#         notifier = DesktopNotifier()
#         last_balance = None
#         elapsed = 0
#         while continue_work:
#             if elapsed > 5:
#                 elapsed = 0
#                 balance = fetch_balance()
#                 if last_balance is None or balance - last_balance > threshold:
#                     last_balance = balance
#                     notifier.send_sync(title='Vast.ai balance', message=f'{balance:.02f}$')
#
#             time.sleep(0.1)
#             elapsed += 0.1
#
#     def upload_job():
#         """
#         Detect changes to the code (in src/scripts) and copy them up to the server (to dst/scripts)
#         """
#         from src.deploy.watch import Watcher
#
#         changed_files = []
#
#         def execute(f):
#             nonlocal changed_files
#             changed_files.append(f)
#
#
#         watch = Watcher([*rsync_job_upload_paths, session.dirpath / 'script.py'], [execute])
#         elapsed = 999
#         while continue_work:
#             if elapsed > 1:
#                 elapsed = 0
#                 watch.monitor_once()
#
#                 for f in changed_files:
#                     f = Path(f)
#                     print(chalk.blue_bright("Changed", f.relative_to(src)))
#                     relative = Path(f).relative_to(src)
#
#                     src2 = src / relative
#                     dst2 = dst / relative
#                     subprocess.run(f"rsync -avz -e 'ssh -p {port}' {src2} root@{ip}:{dst2} --exclude '.*/'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#
#                 changed_files.clear()
#
#             time.sleep(0.25)
#             elapsed += 0.25
#     def download_job():
#         """
#         Detect changes to the code (in src/scripts) and copy them up to the server (to dst/scripts)
#         """
#         from src.deploy.watch import Watcher
#
#         elapsed = 999
#         while continue_work:
#             # Download the latest frames every 1.5 second
#             if elapsed > 3.0:
#                 elapsed = 0
#                 src2 = src / paths.sessions_name
#                 dst2 = dst / paths.sessions_name / session.name
#
#                 # while True:
#                 #     src_file = session.det_frame_path(session.f_last + 1)
#                 #     dst_file = dst2 / src_file.name
#                 #     exists = sftp.exists(dst_file)
#                 #     print(src_file, dst_file, exists)
#                 #     if exists:
#                 #         sftp.get_file(dst_file, src_file)
#                 #         session.f_last += 1
#                 #         session.f_last_path = session.det_frame_path(session.f_last)
#                 #     else:
#                 #         break
#
#                 # Exclude video.mp4 and video__*.mp4
#                 cmd = f"rsync -az -e 'ssh -p {port}' root@{ip}:{dst2} {src2}"
#                 for fname in rsync_job_download_exclusion:
#                     cmd += f" --exclude '{fname}'"
#                 cmd += " --exclude '.*/'"
#
#                 # os.system(cmd)
#                 subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
#
#             time.sleep(0.25)
#             elapsed += 0.25
#
#     # TODO readonly renderer job
#     def renderer_job():
#         """
#         Render the video in the background
#         """
#         # renderer.is_dev = True
#         renderer.enable_readonly = True
#         renderer.enable_dev = True
#         renderer.unsafe = False
#         renderer.init()
#         renderer.run()
#
#     t1 = threading.Thread(target=vastai_job)
#     t2 = threading.Thread(target=balance_job)
#     t3 = threading.Thread(target=upload_job)
#     t4 = threading.Thread(target=download_job)
#     # t5 = threading.Thread(target=renderer_job)
#
#     t1.start()
#     t2.start()
#     t3.start()
#     t4.start()
#
#     # vastai_job()
#     renderer_job()
#     continue_work = False
#     renderer.stop()
#
#     t1.join()
#     t2.join()
#     t3.join()
#     t4.join()
#
#     def on_ctrl_c():
#         nonlocal continue_work
#         continue_work = False
#         renderer.stop()
#         print("Stopping ...")
#         time.sleep(1)
#         print("Stopped")
#         sys.exit(0)
#
#
#     setup_ctrl_c(on_ctrl_c)
#
#     print(f"Remaining balance: {fetch_balance():.02f}$")
#
#     if args.vastai_stop:
#         print(f"Stopping Vast.ai instance {selected_id} in 5 seconds ...")
#         time.sleep(5)
#         subprocess.check_output(['python3', vastpath.as_posix(), 'stop', str(selected_id)]).decode('utf-8')
#         print("All done!")
#
#
#     # interactive.interactive_shell(channel)
# def copy_session(src_session, dst_session, sftp):
#     if src_session is not None and src_session.dirpath.exists():
#         print_header(f"Copying session '{src_session.dirpath.stem}'")
#         sftp.mkdir(str(dst_session))
#         sftp.put_any(src_session.dirpath, dst_session, forbid_recursive=True)
#     # MANUAL SOLUTION BELOW
#     #     upload_list = []
#     #     for file in session.dirpath.iterdir():
#     #         if file.stem.isnumeric() and paths.is_image(file):
#     #             num = int(file.stem)
#     #             # print(num, session.f_last)
#     #             if num < session.f_last:
#     #                 continue
#     #         if file.name in deploy_upload_blacklist_paths: continue
#     #         if file.is_dir(): continue
#     #         upload_list.append(file)
#     #
#     #     for v in upload_list:
#     #         sftp.put_any(v, dst_session / v.name, force_rsync=True)
#     # sftp.put_any(src_session, dst_session, force_rsync=False, rsync_includes=[v.name for v in upload_list])
# def open_shell(ssh):
#     import interactive
#     print_header("user --shell")
#     # Start a ssh shell for the user
#     channel = ssh.invoke_shell()
#     interactive.interactive_shell(channel)
# def sync_res(src, dst, sftp):
#     for file in deploy_upload_paths_sftp:
#         if isinstance(file, str):
#             sftp.put_any(src / file, dst / file, forbid_rsync=True)
#         if isinstance(file, tuple):
#             sftp.put_any(src / file[0], dst / file[1], forbid_rsync=True)
# def sync_files(src, dst, sftp):
#     print_header("File copies...")
#     for file in deploy_upload_paths:
#         if isinstance(file, str):
#             sftp.put_any(src / file, dst / file, rsync_includes=[f'*{v}' for v in paths.text_exts])
#         if isinstance(file, tuple):
#             sftp.put_any(src / file[0], dst / file[1])
#
#
# class DiscoreSSHClient(paramiko.SSHClient):
#     def run(self, cm, cwd=None, *, log_output=False):
#         cm = cm.replace("'", '"')
#         if cwd is not None:
#             cm = f"cd {cwd}; {cm}"
#
#         # cm = f"{ssh_cmd} '{cm}'"
#         # print(f'> {cm}')
#         # ret = os.system(cm)
#
#         cm = f"/bin/bash -c '{cm}'"
#         print_cmd(cm)
#         stdin, stdout, stderr = ssh.exec_command(cm, get_pty=True)
#         stdout.channel.set_combine_stderr(True)
#         ret = ''
#         for line in stdout:
#             ret += line
#             if log_output:
#                 print(line, end='')
#
#         # print(ret)
#
#         return ret
#
#     def file_exists(self, path):
#         ret = self.run(f"stat '{path}'")
#         return ret != ''
#
#     def mount(self, local_path, remote_path, ip, port):
#         # Use sshfs to mount the machine
#         local_path.mkdir(parents=True, exist_ok=True)
#
#         not_mounted = subprocess.Popen(f"mountpoint -q {local_path}", shell=True).wait()
#         if not_mounted > 0:
#             # print_header("Mounting with sshfs...")
#             os.system(f"sshfs root@{ip}:{remote_path} -p {port} {local_path}")
#             print("mounted!")
