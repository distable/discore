import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import aiohttp

from src.deploy import VastAIManager
from src.deploy.DiscoRemote import DiscoRemote
from src.deploy.VastInstance import VastInstance

vast = VastAIManager.instance

log = logging.getLogger('rman')


class RemoteManager:
    """
Manages multiple RemoteInstance objects.

This class orchestrates operations across multiple remote instances,
handling deployment, job management, and balance monitoring.

Attributes:
    remotes (list): List of RemoteInstance objects.
    vast (VastAIManager): Manager for Vast.ai operations.

Methods:
    deploy_all(): Deploys Discore to all managed remote instances.
    start_all_jobs(): Starts jobs on all managed remote instances.
    stop_all_jobs(): Stops jobs on all managed remote instances.
    wait_for_all_jobs(jobs): Waits for all specified jobs to complete.
    monitor_balance(): Continuously monitors the Vast.ai account balance.
    add(remote): Adds a new RemoteInstance to be managed.
    """

    def __init__(self):
        self.remotes: Dict[int, DiscoRemote] = {}

    async def get_remote(self, iid_or_vdata, session: Optional[aiohttp.ClientSession] = None) -> DiscoRemote:
        match iid_or_vdata:
            case DiscoRemote():
                return iid_or_vdata
            case VastInstance() as vdata:
                self.remotes[vdata.id] = DiscoRemote(vdata, None)
                return self.remotes[vdata.id]
            case int() as iid if iid in self.remotes:
                return self.remotes[iid]
            case int() as iid:
                instances = await vast.fetch_instances()  # TODO maybe we can pass in existing?
                if vdata := next((i for i in instances if i.id == iid), None):
                    s = DiscoRemote(vdata, session)
                    self.remotes[iid] = s
                    return s
                raise ValueError(f"Instance {iid} not found")
            case _:
                raise TypeError(f"Unexpected type for iid_or_vdata: {type(iid_or_vdata)}")

    def deploy_all(self):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(remote.deploy) for remote in self.remotes.values()]
            for future in futures:
                future.result()

    def start_all_jobs(self):
        all_jobs = []
        for remote in self.remotes.values():
            all_jobs.extend(remote.start_jobs())
        return all_jobs

    def stop_all_jobs(self):
        for remote in self.remotes.values():
            remote.stop_jobs()

    def wait_for_all_jobs(self, jobs):
        for job in jobs:
            job.join()

    def monitor_balance(self):
        while any(remote.continue_work for remote in self.remotes.values()):
            balance = vast.fetch_balance()
            log.info(f"Current balance: ${balance:.2f}")
            time.sleep(60)  # Check balance every minute

    def add(self, remote):
        self.remotes[remote.id] = remote


instance = RemoteManager()
