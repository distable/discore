import asyncio
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import aiofiles
import aiohttp

import src.deploy.deploy_constants as const
import userconf
from jargs import args
from src.classes import paths
from src.deploy.DiscoRemote import DiscoRemote
from src.deploy.VastInstance import VastInstance
from src.deploy.VastOffer import VastOffer

log = logging.getLogger('vast')

class VastAIManager:
    def __init__(self):
        self.vastpath = paths.root / 'vast'
        self.remotes = dict()
        self.executor = ThreadPoolExecutor()
        self.destroyed_instances = set()

    async def download_vast_script(self):
        if not self.vastpath.is_file():
            async with aiohttp.ClientSession() as session:
                async with session.get("https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py") as response:
                    content = await response.text()
                    async with aiofiles.open(self.vastpath, mode='w') as f:
                        await f.write(content)
            if not sys.platform.startswith('win'):
                await asyncio.to_thread(self.vastpath.chmod, 0o755)

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

    async def fetch_balance(self)->float:
        out = await self.run_command(['show', 'invoices'])
        ret = json.loads(out.splitlines()[-1].replace('Current:  ', '').replace("'", '"'))['credit']
        return ret

    async def fetch_offers(self) -> List[VastOffer]:
        search_query = f"{args.vastai_search or userconf.vastai_default_search} disk_space>{const.VASTAI_DISK_SPACE} verified=true"
        out = await self.run_command(['search', 'offers', search_query])
        return [
            VastOffer(index, *fields[:22])
            for index, line in enumerate(out.splitlines()[1:], start=1)
            if len(fields := line.split()) >= 22
        ]

    async def fetch_instances(self) -> List[VastInstance]:
        out = await self.run_command(['show', 'instances'])
        ret = [
            VastInstance(index, *fields[:18])
            for index, line in enumerate(out.splitlines()[1:], start=1)
            if len(fields := line.split()) >= 18
        ]

        # Remove destroyed instances (the server is not updated immediately) TODO this does not work for forsaken reasons
        ret = [instance for instance in ret if instance.id not in self.destroyed_instances]

        return ret

    async def create_instance(self, offer_id):
        return await self.run_command(['create', 'instance', offer_id, '--image', const.VASTAI_DOCKER_IMAGE, '--disk', const.VASTAI_DISK_SPACE, '--env', '-p 8188:8188', '--ssh'])

    async def destroy_instance(self, instance_id):
        self.remotes.pop(instance_id, None)
        self.destroyed_instances.add(instance_id)
        return await self.run_command(['destroy', 'instance', str(instance_id)])

    async def reboot_instance(self, instance_id):
        return await self.run_command(['reboot', 'instance', str(instance_id)])

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

    def create_instance_sync(self, offer_id):
        return asyncio.run(self.create_instance(offer_id))

    def destroy_instance_sync(self, instance_id):
        return asyncio.run(self.destroy_instance(instance_id))

    def reboot_instance_sync(self, instance_id):
        return asyncio.run(self.reboot_instance(instance_id))

    def stop_instance_sync(self, instance_id):
        return asyncio.run(self.stop_instance(instance_id))


instance = VastAIManager()

