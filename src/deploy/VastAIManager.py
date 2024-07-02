import asyncio
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import aiofiles
import aiohttp

import src.deploy.constants as const
import userconf
from jargs import args
from src.classes import paths
from src.deploy.RemoteInstance import RemoteInstance
from src.deploy.VastInstance import VastInstance
from src.deploy.VastOffer import VastOffer
from src.deploy.deploy_utils import fire_and_forget

log = logging.getLogger(__name__)

class VastAIManager:
    def __init__(self):
        self.vastpath = paths.root / 'vast'
        self.remotes = dict()
        self.executor = ThreadPoolExecutor()

    @fire_and_forget
    async def download_vast_script(self):
        if not self.vastpath.is_file():
            async with aiohttp.ClientSession() as session:
                async with session.get("https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py") as response:
                    content = await response.text()
                    async with aiofiles.open(self.vastpath, mode='w') as f:
                        await f.write(content)
            if not sys.platform.startswith('win'):
                await asyncio.to_thread(self.vastpath.chmod, 0o755)

    @fire_and_forget
    async def run_command(self, command):
        log.info(f"Running command: vast {' '.join(command)}")
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            self.vastpath.as_posix(),
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        return stdout.decode('utf-8')

    @fire_and_forget
    async def fetch_balance(self):
        out = await self.run_command(['show', 'invoices'])
        return json.loads(out.splitlines()[-1].replace('Current:  ', '').replace("'", '"'))['credit']

    @fire_and_forget
    async def fetch_offers(self) -> List[VastOffer]:
        search_query = f"{args.vastai_search or userconf.vastai_default_search} disk_space>{const.DISK_SPACE} verified=true"
        out = await self.run_command(['search', 'offers', search_query])
        return [
            VastOffer(index, *fields[:22])
            for index, line in enumerate(out.splitlines()[1:], start=1)
            if len(fields := line.split()) >= 22
        ]

    @fire_and_forget
    async def fetch_instances(self) -> List[VastInstance]:
        out = await self.run_command(['show', 'instances'])
        return [
            VastInstance(index, *fields[:18])
            for index, line in enumerate(out.splitlines()[1:], start=1)
            if len(fields := line.split()) >= 18
        ]

    @fire_and_forget
    async def get_instance(self, instance_id, session: Optional[aiohttp.ClientSession] = None) -> RemoteInstance:
        if instance_id in self.remotes:
            return self.remotes[instance_id]

        instances = await self.fetch_instances()
        instance_data = next((i for i in instances if i.id == instance_id), None)
        if instance_data:
            self.remotes[instance_id] = RemoteInstance(instance_data, None)
            return RemoteInstance(instance_data, session)

        raise ValueError(f"Instance {instance_id} not found")

    @fire_and_forget
    async def create_instance(self, offer_id):
        return await self.run_command(['create', 'instance', offer_id, '--image', const.DOCKER_IMAGE, '--disk', const.DISK_SPACE, '--env', '-p 8188:8188', '--ssh'])

    @fire_and_forget
    async def destroy_instance(self, instance_id):
        self.remotes.pop(instance_id, None)
        return await self.run_command(['destroy', 'instance', str(instance_id)])

    @fire_and_forget
    async def reboot_instance(self, instance_id):
        return await self.run_command(['reboot', 'instance', str(instance_id)])

    @fire_and_forget
    async def stop_instance(self, instance_id):
        return await self.run_command(['stop', str(instance_id)])

    # Synchronous versions of the methods for compatibility
    def download_vast_script_sync(self):
        asyncio.run(self.download_vast_script())

    def run_command_sync(self, command):
        return asyncio.run(self.run_command(command))

    def fetch_balance_sync(self):
        return asyncio.run(self.fetch_balance())

    def fetch_offers_sync(self) -> List[VastOffer]:
        return asyncio.run(self.fetch_offers())

    def fetch_instances_sync(self) -> List[VastInstance]:
        return asyncio.run(self.fetch_instances())

    def get_instance_sync(self, instance_id, session: Optional[aiohttp.ClientSession] = None) -> RemoteInstance:
        return asyncio.run(self.get_instance(instance_id, session))

    def create_instance_sync(self, offer_id):
        return asyncio.run(self.create_instance(offer_id))

    def destroy_instance_sync(self, instance_id):
        return asyncio.run(self.destroy_instance(instance_id))

    def reboot_instance_sync(self, instance_id):
        return asyncio.run(self.reboot_instance(instance_id))

    def stop_instance_sync(self, instance_id):
        return asyncio.run(self.stop_instance(instance_id))


instance = VastAIManager()
