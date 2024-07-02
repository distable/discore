import time
from concurrent.futures import ThreadPoolExecutor

from src.deploy.VastAIManager import VastAIManager


class RemoteManager:
    """
Manages multiple RemoteInstance objects.

This class orchestrates operations across multiple remote instances,
handling deployment, job management, and balance monitoring.

Attributes:
    remotes (list): List of RemoteInstance objects.
    vast_manager (VastAIManager): Manager for Vast.ai operations.

Methods:
    deploy_all(): Deploys Discore to all managed remote instances.
    start_all_jobs(): Starts jobs on all managed remote instances.
    stop_all_jobs(): Stops jobs on all managed remote instances.
    wait_for_all_jobs(jobs): Waits for all specified jobs to complete.
    monitor_balance(): Continuously monitors the Vast.ai account balance.
    add(remote): Adds a new RemoteInstance to be managed.
    """

    def __init__(self):
        self.remotes = []
        self.vast_manager = VastAIManager()
        self.logger = None

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

    def stop_all_jobs(self):
        for remote in self.remotes:
            remote.stop_jobs()

    def wait_for_all_jobs(self, jobs):
        for job in jobs:
            job.join()

    def monitor_balance(self):
        while any(remote.continue_work for remote in self.remotes):
            balance = self.vast_manager.fetch_balance()
            self.logger.info(f"Current balance: ${balance:.2f}")
            time.sleep(60)  # Check balance every minute

    def add(self, remote):
        self.remotes.append(remote)

instance = RemoteManager()