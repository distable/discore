"""
Discore Deployment Script for Vast.ai

This script manages the deployment of Discore to Vast.ai instances, supporting
multiple remote deployments for parallel rendering of animations.

Note: In this context, 'src' always refers to the local machine, 'dst' refers to the remote machine.
"""
import asyncio
import logging
import subprocess
import threading
import time
from pathlib import Path

from yachalk import chalk

import deploy_utils
import jargs
import userconf
from jargs import args
from src import renderer
from src.classes import paths
from src.deploy import VastAIManager, RemoteManager
# from src.deploy.RemoteInstance import RemoteInstance
# from src.deploy.RemoteManager import RemoteManager
# from src.deploy.VastAIManager import VastAIManager
import src.deploy.constants as const
from src.deploy.RemoteInstance import RemoteInstance

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

logging.basicConfig(filename="deploy.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(levelname)s [%(name)s] %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

vast_manager = VastAIManager.instance
manager = RemoteManager.instance
manager.log = log


def deploy_vastai():
    """
Main function to deploy Discore on a single Vast.ai instance.

This function handles the entire deployment process, including:
- Downloading the Vast.ai script
- Fetching available instances or creating a new one
- Connecting to the selected instance
- Deploying Discore
- Starting background jobs
- Monitoring balance
- Handling user interrupts
- Stopping the instance if requested

The function interacts with the user to select an instance if necessary
and provides feedback throughout the deployment process.
    """
    from src.deploy import DeployUI
    asyncio.run(DeployUI.main())
    return

    deploy_utils.download_vast_script()

    session = jargs.get_discore_session()

    instances = manager.vast_manager.fetch_instances()
    if args.vastai_list:
        for instance in instances:
            log.info(chalk.green_bright(str(instance)))
        return

    selected_id = instances[0].id if instances else None
    if selected_id is None:
        offers = manager.vast_manager.fetch_offers()
        offers.sort(key=lambda e: float(e.price), reverse=True)
        for i, offer in enumerate(offers):
            log.info(chalk.green(offer.tostring(i)))

        choice = int(input("Enter the number of the machine you want to use: ")) - 1
        log.info(f"Creating instance with offer {offers[choice].id} ...")

        selected_id = offers[choice].id
        manager.vast_manager.create_instance(selected_id)

        time.sleep(3)
        new_instances = [i for i in manager.vast_manager.fetch_instances() if i.id not in [e.id for e in instances]]
        if len(new_instances) != 1:
            log.error("Failed to create instance, couldn't find the new instance. Check that you have spare credits.")
            return
        selected_id = new_instances[0].id

    if args.vastai_delete:
        log.info(f"Deleting Vast.ai instance {selected_id} in 5 seconds ...")
        time.sleep(5)
        manager.vast_manager.destroy_instance(selected_id)
        log.info("All done!")
        return

    if args.vastai_reboot:
        log.info(f"Rebooting Vast.ai instance {selected_id} ...")
        manager.vast_manager.reboot_instance(selected_id)

    remote = manager.vast_manager.get_instance(selected_id, session)
    remote.connect()
    remote.deploy(session,
                  b_clone=not jargs.is_vastai_continue,
                  b_shell=args.shell,
                  b_sshfs=userconf.vastai_sshfs,
                  b_copy_workdir=not
                  args.vastai_quick,
                  b_pip_upgrades=args.vastai_upgrade)

    manager.add(remote)

    all_jobs = manager.start_all_jobs()

    # Start balance monitoring in a separate thread
    balance_thread = threading.Thread(target=manager.monitor_balance)
    balance_thread.start()

    try:
        manager.wait_for_all_jobs(all_jobs)
    except KeyboardInterrupt:
        log.info("Received interrupt, stopping jobs...")
        manager.stop_all_jobs()

    balance_thread.join()

    log.info(f"Remaining balance: {manager.vast_manager.fetch_balance():.02f}$")

    if args.vastai_stop:
        log.info(f"Stopping Vast.ai instance {selected_id} in 5 seconds ...")
        time.sleep(5)
        manager.vast_manager.stop_instance(selected_id)
        log.info("All done!")


def copy_session(src_session, dst_path, sftp):
    if src_session is not None and src_session.dirpath.exists():
        log.info(chalk.green(f"Copying session '{src_session.dirpath.stem}'"))
        sftp.mkdir(str(dst_path))
        sftp.put_any(src_session.dirpath, dst_path, forbid_recursive=True)


def renderer_job():
    renderer.enable_readonly = True
    renderer.enable_dev = True
    renderer.unsafe = False
    renderer.init()
    renderer.run()


def multi_deploy_vastai():
    deploy_utils.download_vast_script()

    sessions = jargs.get_discore_sessions()

    instances = manager.vast_manager.fetch_instances()
    if args.vastai_list:
        for instance in instances:
            log.info(str(instance))
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
                log.info(str(offer))

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
        remote = RemoteInstance(instance, session)
        remote.connect()

        manager.add(remote)

    manager.deploy_all()

    all_jobs = manager.start_all_jobs()

    # Start balance monitoring in a separate thread
    balance_thread = threading.Thread(target=manager.monitor_balance)
    balance_thread.start()

    try:
        manager.wait_for_all_jobs(all_jobs)
    except KeyboardInterrupt:
        log.info("Received interrupt, stopping jobs...")
        manager.stop_all_jobs()

    balance_thread.join()

    log.info(f"Remaining balance: {manager.vast_manager.fetch_balance():.02f}$")

    if args.vastai_stop:
        log.info(f"Stopping all Vast.ai instances in 5 seconds ...")
        time.sleep(5)
        for remote in manager.remotes:
            manager.vast_manager.stop_instance(remote.id)
        log.info("All instances stopped.")


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

    for file in const.FAST_UPLOADS:
        if isinstance(file, tuple):
            shutil.copyfile(src / file[0], dst / file[1])
        if isinstance(file, str):
            shutil.copyfile(src / file, dst / file)

    # 3. Run ~/discore_deploy/discore.py
    if platform.system() == "Linux":
        subprocess.run(['chmod', '+x', dst / 'discore.py'])

    subprocess.run([dst / 'discore.py', '--upgrade'])
