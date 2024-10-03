import nats
import asyncio
import docker
import platform
import os
import importlib
from tabulate import tabulate

from datetime import datetime, timezone
from magnet.utils.data_classes import MagnetConfig, IndexConfig, Status
from magnet.utils.globals import _f
from magnet.ic.field import Charge, Resonator

milvus_server = None  # Set to None initially
os_name = platform.system()

if os_name == 'Darwin' and not os.getenv('DOCKER_ENV'):
    try:
        milvus_module = importlib.import_module('milvus')
        # Initialize milvus_server only if imported
        milvus_server = milvus_module.default_server
    except ImportError:
        _f("warn", "milvus module not found on macOS without DOCKER_ENV")

auto_config = {
    "host": "127.0.0.1",
    "credentials": None,
    "domain": None,
    "stream_name": "my_stream",
    "category": "my_category",
    "kv_name": "my_kv",
    "session": "my_session",
    "os_name": "my_object_store",
    "index": {
        "milvus_uri": "127.0.0.1",
        "milvus_port": 19530,
        "milvus_user": "test",
        "milvus_password": "test",
        "dimension": 1024,
        "model": "BAAI/bge-large-en-v1.5",
        "name": "test",
        "options": {
            'metric_type': 'COSINE',
            'index_type': 'HNSW',
            'params': {
                "efConstruction": 40,
                "M": 48
            }
        }
    }
}


class Magnet:
    def __init__(self, config: MagnetConfig | dict, status_callback):
        try:
            if isinstance(config, dict):
                config = MagnetConfig(**config)
            if isinstance(config.index, dict):
                config.index = IndexConfig(**config.index)
            else:
                status_callback(Status(datetime.now(
                ), "fatal", "config must be a MagnetConfig instance or a dictionary"))
                raise ValueError(
                    "config must be a MagnetConfig instance or a dictionary")
        except Exception as e:
            raise e
        self.config = config
        self.stream = None
        self.nc = None
        self.js = None
        self.kv = None
        self.os = None
        self.fabric = None
        self.jobs_kv = None
        self.runs_kv = None
        self.status_callback = status_callback
        self.charge = Charge(self)
        self.resonator = Resonator(self)

    async def _cre_handler(self, cre):
        self.status_callback(Status(datetime.now(), "warn", cre))

    async def align(self, backoff_strategy: str = 'eul'):
        """Connect to NATS and setup configurations with customizable backoff based on a ratio."""

        attempt = 0
        eulers_number = 2.71828
        golden_ratio = 1.61803398875

        while True:
            try:
                _server = f'{"nats://" if not self.config.credentials else "tls://"}{self.config.host}:4222'
                self.nc = await nats.connect(
                    _server,
                    user_credentials=self.config.credentials,
                    error_cb=self._cre_handler
                )
                self.status_callback(
                    Status(datetime.now(), "success", f"🧲 connected to \n💎 {_server} "))

                try:
                    self.js = self.nc.jetstream(domain=self.config.domain)
                    self.status_callback(
                        Status(datetime.now(), "info", "🧲 initialized client "))
                except Exception as e:
                    self.status_callback(Status(
                        datetime.now(), "fatal", f"could not find domain {self.config.domain}\n{e}"))
                    raise e

                if self.config.stream_name:
                    self.stream = await self._setup_stream()
                if self.config.kv_name:
                    self.kv = await self._setup_kv()
                    self.jobs_kv = await self._setup_kv_substore("jobs")
                    self.fabric_kv = await self._setup_kv_substore("fabric")
                    self.runs_kv = await self._setup_kv_substore("runs")
                if self.config.os_name:
                    self.os = await self._setup_object_store()
                    self.jobs_os = await self._setup_sub_object_store("jobs")
                    self.fabric_os = await self._setup_sub_object_store("fabric")
                    self.runs_os = await self._setup_sub_object_store("runs")

                self.status_callback(
                    Status(datetime.now(), "success", f"🧲 connected to \n💎 {self.config} "))
                return [self.js, self.stream, self.kv, self.jobs_kv, self.runs_kv, self.fabric_kv, self.os, self.jobs_os, self.runs_os, self.fabric_os]
            except Exception as e:
                self.status_callback(
                    Status(datetime.now(), "fatal", f"could not align {self.config.host}\n{e}"))
                # Determine delay based on ratio type
                if backoff_strategy == 'eul':
                    # Exponential backoff using Euler's number
                    delay = pow(eulers_number, attempt)
                elif backoff_strategy == 'fib':
                    # Linear backoff approximated by multiplying attempt number by the Golden Ratio
                    delay = attempt * golden_ratio

                self.status_callback(Status(datetime.now(
                ), "wait", f"attempt {attempt+1} failed, retrying in {delay:.2f} seconds"))
                await asyncio.sleep(delay)
                attempt += 1

    async def _setup_stream(self):
        """Setup stream."""
        try:
            await self.js.stream_info(self.config.stream_name)
        except Exception as e:
            self.status_callback(Status(
                datetime.now(), "warn", f"Stream {self.config.stream_name} not found, creating"))
            await self.js.add_stream(name=self.config.stream_name, subjects=[self.config.category, 'jobs.>', 'runs.>', 'fabric.>'])
            self.status_callback(Status(datetime.now(
            ), "success", f"created `{self.config.stream_name}` with category `{self.config.category}`"))
        return await self.js.stream_info(self.config.stream_name)

    async def _setup_kv(self):
        """Setup key-value store."""
        try:
            await self.js.key_value(f"{self.config.kv_name}")
        except Exception as e:
            self.status_callback(Status(
                datetime.now(), "warn", f"KV bucket {self.config.kv_name} not found, creating"))
            await self.js.create_key_value(bucket=self.config.kv_name)
            self.status_callback(
                Status(datetime.now(), "success", f"created `{self.config.kv_name}`"))
        return await self.js.key_value(self.config.kv_name)

    async def _setup_kv_substore(self, substore_name):
        """Setup a hierarchical key-value substore under the main kv_name."""
        full_name = f"{substore_name}".lower().replace(
            "_", "-")

        # Ensure the bucket name does not start or end with a hyphen
        full_name = full_name.strip('-')

        if len(full_name) < 2 or len(full_name) > 255:
            self.status_callback(Status(datetime.now(
            ), "fatal", f"Invalid bucket name: {full_name}. Must be 2-255 characters long and conform to JetStream naming rules."))
            raise ValueError(f"Invalid bucket name: {full_name}")

        try:
            await self.js.key_value(full_name)
        except Exception as e:
            self.status_callback(
                Status(datetime.now(), "warn", f"KV substore {full_name} not found, creating"))
            try:
                await self.js.create_key_value(bucket=full_name)
                self.status_callback(
                    Status(datetime.now(), "success", f"created `{full_name}`"))
            except Exception as e:
                self.status_callback(Status(
                    datetime.now(), "fatal", f"Failed to create KV substore {full_name}\n{e}"))
                raise e
        return await self.js.key_value(full_name)
    async def _setup_sub_object_store(self, substore_name):
        try:
            await self.js.object_store(substore_name)
        except Exception as e:
            self.status_callback(Status(datetime.now(
            ), "warn", f"Object Store {substore_name} not found, creating"))
            await self.js.create_object_store(bucket=substore_name)
            self.status_callback(
                Status(datetime.now(), "success", f"created `{substore_name}`"))
        return await self.js.object_store(substore_name)
    
    async def _setup_object_store(self):
        """Setup object store."""
        try:
            await self.js.object_store(self.config.os_name)
        except Exception as e:
            self.status_callback(Status(datetime.now(
            ), "warn", f"Object Store {self.config.os_name} not found, creating"))
            await self.js.create_object_store(bucket=self.config.os_name)
            self.status_callback(
                Status(datetime.now(), "success", f"created `{self.config.os_name}`"))
        return await self.js.object_store(self.config.os_name)

    async def _list_streams(self):
        try:
            streams = await self.js.streams_info()
            remote_streams = [x.config.name for x in streams]
            remote_subjects = [x.config.subjects for x in streams]
            data = zip(remote_streams, remote_subjects)
            formatted_data = []

            for stream, subjects in data:
                match_stream_name = stream == self.config.stream_name
                match_subject = self.config.category in subjects

                if match_stream_name or match_subject:
                    formatted_stream = f"\033[92m{stream} \U0001F9F2\033[0m"
                    formatted_subjects = [f"\033[92m{subject} \U0001F9F2\033[0m" if subject ==
                                          self.config.category else subject for subject in subjects]
                else:
                    formatted_stream = stream
                    formatted_subjects = subjects

                formatted_data.append(
                    [formatted_stream, ', '.join(formatted_subjects)])

            table = tabulate(formatted_data, headers=[
                             'Stream Name', 'Subjects'], tablefmt="pretty")

            self.status_callback(
                Status(datetime.now(timezone.utc), "info", f'\n{table}'))
        except TimeoutError:
            self.status_callback(Status(
                datetime.now(timezone.utc), 'fatal', f'could not connect to {self.magnet.config.host}'))
        except Exception as e:
            self.magnet.status_callback(
                Status(datetime.now(timezone.utc), 'fatal', str(e)))

    async def off(self):
        """Disconnects from the NATS server and prints a warning message."""
        await self.nc.drain()
        self.status_callback(
            Status(datetime.now(), 'warn', f'disconnected from {self.config.host}'))


class EmbeddedMagnet:
    def __init__(self, status_callback):
        self.client = None
        self.status_callback = status_callback
        try:
            self.client = docker.from_env()
            self.status_callback(
                Status(datetime.now(), 'success', 'containerization engine connected'))
        except Exception as e:
            self.status_callback(Status(datetime.now(), 'fatal', str(e)))
        self.nats_image = "nats:latest"
        self.client.images.pull(self.nats_image)
        if os_name == 'Darwin':
            pass

    def start(self):
        try:
            nats_container = self.client.containers.run(
                self.nats_image,
                name="magnet-embedded-nats",
                detach=True,
                ports={'4222/tcp': 4222},
                command="-js"
            )
        except Exception as e:
            self.status_callback(Status(datetime.now(), 'fatal', str(e)))
            return
        for container in self.client.containers.list():
            self.status_callback(
                Status(datetime.now(), 'wait', f"{container}"))
            if container.id == nats_container.id:
                self.status_callback(Status(datetime.now(
                ), "wait", f"{container.name} container progressing with id {container.id}"))
                nats_logs = container.logs().decode('utf-8').split('[INF]')

            if container.name == "magnet-embedded-nats" and container.status == "running":
                self.status_callback(
                    Status(datetime.now(), "info", f"nats logs"))
                for log in nats_logs:
                    self.status_callback(
                        Status(datetime.now(), "warn", f"{log}", luxe=True))

            if container.status == "exited":
                self.status_callback(
                    Status(datetime.now(), "fatal", f"{container.name} has exited"))
                break
        try:
            if milvus_server:  # Only try to start if milvus_server is initialized
                milvus_server.start()
                self.status_callback(
                    Status(datetime.now(), "success", "milvus server started"))
        except Exception as e:
            self.status_callback(
                Status(datetime.now(), "fatal", f"milvus failure\n{e}"))

    def stop(self):
        for container in self.client.containers.list():
            if container.name == "magnet-embedded-nats":
                self.status_callback(
                    Status(datetime.now(), "wait", f"stopping {container.name}"))
                container.stop()
                self.status_callback(
                    Status(datetime.now(), "success", f"{container.name} stopped"))
                self.status_callback(
                    Status(datetime.now(), "info", f"removing {container.name}"))
                container.remove()
                self.status_callback(
                    Status(datetime.now(), "success", f"embedded nats removed"))
        try:
            if milvus_server:  # Only try to stop if milvus_server is initialized
                milvus_server.stop()
                self.status_callback(
                    Status(datetime.now(), "success", "embedded milvus server stopped"))
        except Exception as e:
            self.status_callback(
                Status(datetime.now(), "warn", f"embedded milvus can't be stopped\n{e}"))

    def create_magnet(self):
        self.status_callback(
            Status(datetime.now(), 'wait', 'creating magnet with embedded cluster'))
        magnet = Magnet(auto_config, self.status_callback)
        return magnet

    def cleanup(self):
        self.client.images.prune()
        self.client.volumes.prune()
        self.status_callback(
            Status(datetime.now(), "warn", "container engine pruned"))
        try:
            if milvus_server:  # Only try to cleanup if milvus_server is initialized
                milvus_server.cleanup()
                self.status_callback(
                    Status(datetime.now(), "success", "embedded cluster cleaned up"))
        except Exception as e:
            self.status_callback(
                Status(datetime.now(), "fatal", f"error cleaning up milvus\n{e}"))
