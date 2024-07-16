"""
Discore Deployment Script for Vast.ai

This script manages the deployment of Discore to Vast.ai instances, supporting
multiple remote deployments for parallel rendering of animations.

Note: In this context, 'src' always refers to the local machine, 'dst' refers to the remote machine.
"""
import asyncio
import os
import sys

from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, FloatContainer, Window, Float, HSplit, Dimension
from prompt_toolkit.widgets import Frame

import jargs
from src.deploy.DeployUI import OfferList
from src.deploy.DiscoRemote import SSHConnectionState
from src.deploy.VastOffer import VastOffer
from src.deploy.deploy_utils import make_header_text, forget
from src.lib import loglib

os.environ['SSH_TTY'] = ''
os.environ['TERM'] = 'dumb'

import logging
import subprocess
from pathlib import Path

# from src.deploy.RemoteInstance import RemoteInstance
# from src.deploy.RemoteManager import RemoteManager
# from src.deploy.VastAIManager import VastAIManager
import src.deploy.deploy_constants as const
from src import renderer
from src.classes import paths
from src.deploy import VastAIManager, RemoteManager, DeployUI
from jargs import args

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('deploy')

logger = logging.getLogger(__name__)

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# )

loglib.use_beeprint = False

vast = VastAIManager.instance
rman = RemoteManager.instance


def create_offers_dialog(offer_list):
    return Layout(
        FloatContainer(
            content=Window(),
            floats=[
                Float(
                    Frame(HSplit([offer_list, ],
                                 width=Dimension(preferred=99999999999)),
                          title="Select an offer"),
                    top=2, left=2, right=2, bottom=2
                )
            ]
        )
    )


async def run_interactive_picker() -> VastOffer|None:
    selected_offer = None

    def on_offer_confirm(offer):
        nonlocal selected_offer
        selected_offer = offer

        app.exit()

    offer_list = OfferList(on_offer_confirm)
    layout = create_offers_dialog(offer_list)
    kb = KeyBindings()

    @kb.add('q')
    def k_q(event):
        app.exit()


    app = Application(
        layout=layout,
        full_screen=True,
        key_bindings=kb
    )

    forget(offer_list.fetch_offers())
    await app.run_async()
    return selected_offer


async def deploy_vastai():
    setup_logging()

    if args.vastai_gui:
        from src.deploy import DeployUI
        loglib.use_logging_lib = True
        await DeployUI.main()
        return

    async def create_instance():
        while True:
            # offers = await vast.fetch_offers()
            # offers.sort(key=lambda e: float(e.price), reverse=True)
            #
            # # Print column titles
            # for i,offer in enumerate(offers):
            #     log.info(offer.tostring(i))
            #
            # choice = int(input("Enter the number of the machine you want to use: ")) - 1
            # if choice < 0 or choice >= len(offers):
            #     print(f"Invalid choice ({choice}, {len(offers)})")
            #     continue

            offer = await run_interactive_picker()

            # print(f"Creating instance with offer {choice+1} ...")
            # await asyncio.sleep(3)

            offer_id = offer.id
            result = await vast.create_instance(offer_id)
            print("CREATED RESULT")
            print(result)
            if result:
                new_id = result['new_contract']

                # wait til new instance is present (it's not always instant)
                while True:
                    instances = await vast.fetch_instances()
                    if any(instance.id == new_id for instance in instances):
                        break
                    await asyncio.sleep(1)

                return new_id

    instances = await vast.fetch_instances()
    if not instances:
        await create_instance()

    if instances[0].status == "expired":
        vast.destroy_instance(instances[0].id)
        await create_instance()

    remote = await rman.get_remote(instances[0])
    await remote.wait_for_ready()
    await remote.connect()

    if args.vastai_shell or remote.connection_state == SSHConnectionState.HOST_KEY_NOT_VERIFIABLE:
        task = remote.shell()

        if args.vastai_shell:
            log.info(make_header_text("Entering shell mode"))
            await task
            return

        await task
        await remote.connect()


    await remote.deploy(jargs.get_discore_session(),
                        b_pip_upgrades=args.vastai_upgrade,
                        b_send_fast=not args.vastai_quick,
                        b_send_slow=False,
                        b_redeploy=args.vastai_redeploy)

    if args.vastai_comfy:
        await remote.start_comfy()
    else:
        if not await remote.is_comfy_running():
            forget(remote.start_comfy())
            await asyncio.sleep(10)
        else:
            log.info("Comfy already running")

        await(remote.start_discore())
        # await remote._discore_job()

        while True:
            await asyncio.sleep(1)


class LogFilter(logging.Filter):
    def filter(self, rec):
        """Filtering internal logs of aiosmtpd, asyncssh"""
        if rec.name == "asyncssh" and rec.levelno < logging.WARNING: return False
        if rec.name == "asyncssh.sftp" and rec.levelno < logging.WARNING: return False

        return True


class EmptyLineFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)

    def format(self, record):
        # Check if the log message is empty or just a newline
        if record.msg.strip() == "":
            return ""

        # Check if the log message is a header
        if record.msg.startswith("\n"):
            return record.msg

        if 'There was an error checking the latest version of pip' in record.msg:
            return record.msg  # I dunno if this is a good idea but it works

        # For non-empty messages, use the standard formatting
        return super().format(record)


def setup_logging():
    formatter = EmptyLineFormatter('%(asctime)s %(levelname)s [%(name)s] %(message)s')

    file_hnd = logging.FileHandler("deploy.log", mode="a", encoding='utf-8')
    file_hnd.setLevel(logging.INFO)
    # file_hnd.addFilter(LogFilter())
    file_hnd.setFormatter(formatter)

    handlers = [file_hnd, ]
    if not args.vastai_gui:
        stream_hnd = logging.StreamHandler(sys.stdout)
        stream_hnd.setLevel(logging.INFO)
        stream_hnd.addFilter(LogFilter())
        stream_hnd.setFormatter(formatter)
        handlers.append(stream_hnd)

    logging.basicConfig(handlers=handlers, level=logging.INFO)


def renderer_job():
    renderer.enable_readonly = True
    renderer.enable_dev = True
    renderer.unsafe = False
    renderer.init()
    renderer.run()


# def multi_deploy_vastai():
#     deploy_utils.download_vast_script()
#
#     sessions = jargs.get_discore_sessions()
#
#     instances = vast.fetch_instances()
#     if args.vastai_list:
#         for instance in instances:
#             log.info(str(instance))
#         return
#
#     num_instances = len(sessions)
#     selected_ids = []
#
#     # Use existing instances or create new ones
#     for _ in range(num_instances):
#         if instances:
#             selected_ids.append(instances.pop(0).id)
#         else:
#             offers = vast.fetch_offers()
#             offers.sort(key=lambda e: float(e['price']), reverse=True)
#             for offer in offers:
#                 log.info(str(offer))
#
#             choice = int(input("Enter the number of the machine you want to use: ")) - 1
#             offer_id = offers[choice].id
#             vast.create_instance(offer_id)
#
#             time.sleep(3)
#             new_instance = vast.fetch_instances()[-1]
#             selected_ids.append(new_instance.id)
#
#     # Wait for all instances to be ready
#     ready_instances = []
#     for instance_id in selected_ids:
#         instance = vast.get_instance(instance_id)
#         ready_instances.append(instance.wait_for_ready())
#
#     # Create RemoteInstance objects and connect
#     for instance, session in zip(ready_instances, sessions):
#         remote = DiscoRemote(instance, session)
#         remote.connect()
#
#         manager.add(remote)
#
#     manager.deploy_all()
#
#     all_jobs = manager.start_all_jobs()
#
#     # Start balance monitoring in a separate thread
#     balance_thread = threading.Thread(target=manager.monitor_balance)
#     balance_thread.start()
#
#     try:
#         manager.wait_for_all_jobs(all_jobs)
#     except KeyboardInterrupt:
#         log.info("Received interrupt, stopping jobs...")
#         manager.stop_all_jobs()
#
#     balance_thread.join()
#
#     log.info(f"Remaining balance: {vast.fetch_balance():.02f}$")
#
#     if args.vastai_stop:
#         log.info(f"Stopping all Vast.ai instances in 5 seconds ...")
#         time.sleep(5)
#         for remote in manager.remotes:
#             vast.stop_instance(remote.id)
#         log.info("All instances stopped.")


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

    for file in const.UPLOADS_ON_CONNECT:
        if file.is_absolute():
            file = file.relative_to(paths.root)

        if isinstance(file, tuple):
            shutil.copyfile(src / file[0], dst / file[1])
        if isinstance(file, str):
            shutil.copyfile(src / file, dst / file)

    # 3. Run ~/discore_deploy/discore.py
    if platform.system() == "Linux":
        subprocess.run(['chmod', '+x', dst / 'discore.py'])

    subprocess.run([dst / 'discore.py', '--upgrade'])
