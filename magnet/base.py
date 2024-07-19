
import nats, asyncio, docker, platform

import platform

# Check if the OS is macOS
if platform.system() == 'Darwin':
    from milvus import default_server
    
from magnet.utils.globals import _f
from magnet.utils.data_classes import MagnetConfig, IndexConfig

milvus_server = default_server
os_name = platform.system()

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
            'index_type':'HNSW',
            'params': {
                "efConstruction": 40
                , "M": 48
            }
        }
    }
}

class Magnet:
    def __init__(self, config: MagnetConfig | dict = None):
        try:
            if isinstance(config, dict):
                config = MagnetConfig(**config)
            if isinstance(config.index, dict):
                config.index = IndexConfig(**config.index)
            else:
                _f("fatal", "config must be a MagnetConfig instance or a dictionary")
                raise ValueError
        except Exception as e:
            raise e
        self.config = config
        self.nc = None
        self.js = None
        self.kv = None
        self.os = None

    async def _cre_handler(self, cre):
        _f("warn", cre, no_print=True)

    async def align(self, backoff_strategy: str = 'eul'):
        """Connect to NATS and setup configurations with customizable backoff based on a ratio.

        Args:
            ratio_type (str): The type of ratio to use for backoff. 'euler' for Euler's number, 'fibonacci' for the Golden Ratio.
            max_retries (int): Maximum number of retries before giving up.
        """
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
                _f("success", f"🧲 connected to \n💎 {_server} ")
                
                try:
                    self.js = self.nc.jetstream(domain=self.config.domain)
                    _f("info", f"🧲 initialized client ")
                except Exception as e:
                    _f("fatal", f"could not find domain {self.config.domain}\n{e}")
                    raise e
                if self.config.stream_name:
                    await self._setup_stream()
                if self.config.kv_name:
                    self.kv = await self._setup_kv()
                if self.config.os_name:
                    self.os = await self._setup_object_store()
                _f("success", f"🧲 connected to \n💎 {self.config} ")
                return [self.js, self.kv, self.os]
            except Exception as e:
                _f("fatal", f"could not align {self.config.host}\n{e}")
                # Determine delay based on ratio type
                if backoff_strategy == 'eul':
                    delay = pow(eulers_number, attempt)  # Exponential backoff using Euler's number
                elif backoff_strategy == 'fib':
                    delay = attempt * golden_ratio  # Linear backoff approximated by multiplying attempt number by the Golden Ratio

                _f("wait", f"attempt {attempt+1} failed, retrying in {delay:.2f} seconds")
                await asyncio.sleep(delay)
                attempt += 1

    async def _setup_stream(self):
        """Setup stream."""
        try:
            await self.js.stream_info(self.config.stream_name)
        except Exception as e:
            _f("warn", f"Stream {self.config.stream_name} not found, creating")
            await self.js.add_stream(name=self.config.stream_name, subjects=[self.config.category])
            _f("success", f"created `{self.config.stream_name}` with category `{self.config.category}`")
        return await self.js.stream_info(self.config.stream_name)

    async def _setup_kv(self):
        """Setup key-value store."""
        try:
            await self.js.key_value(self.config.kv_name)
        except Exception as e:
            _f("warn", f"KV bucket {self.config.kv_name} not found, creating")
            await self.js.create_key_value(bucket=self.config.kv_name)
            _f("success", f"created `{self.config.kv_name}`")
        return await self.js.key_value(self.config.kv_name)

    async def _setup_object_store(self):
        """Setup object store."""
        try:
            await self.js.object_store(self.config.os_name)
        except Exception as e:
            _f("warn", f"Object Store {self.config.os_name} not found, creating")
            await self.js.create_object_store(bucket=self.config.os_name)
            _f("success", f"created `{self.config.os_name}`")
        return await self.js.object_store(self.config.os_name)

    async def off(self):
        """Disconnects from the NATS server and prints a warning message."""
        await self.nc.drain()
        _f('warn', f'disconnected from {self.config.host}')

class EmbeddedMagnet:
    def __init__(self):
        self.client = None
        try:
            self.client = docker.from_env()
            _f('success', 'containerization engine connected')
        except Exception as e:
            _f('fatal', e)
        self.nats_image = "nats:latest"
        self.client.images.pull(self.nats_image)
        if os_name == 'Darwin':
            pass

    def start(self):
        try:
            nats_container = self.client.containers.run(
                self.nats_image
                , name="magnet-embedded-nats"
                , detach=True
                , ports={'4222/tcp': 4222}
                , command="-js"
            )
        except Exception as e:
            return _f('fatal', e)
        for container in self.client.containers.list():
            _f('wait', f"{container}")
            if container.id == nats_container.id:
                _f("wait", f"{container.name} container progressing with id {container.id}")
                nats_logs = container.logs().decode('utf-8').split('[INF]')
            
            if container.name == "magnet-embedded-nats" and container.status == "running":
                _f("info", f"nats logs")
                for log in nats_logs:
                    _f("warn", f"{log}", luxe=True)
            
            if container.status == "exited":
                _f("fatal", f"{container.name} has exited")
                break
        try:
            milvus_server.start()
            _f("success", "milvus server started")
        except Exception as e:
            _f("fatal", f"milvus failure\n{e}")

    def stop(self):
        for container in self.client.containers.list():
            if container.name == "magnet-embedded-nats":
                _f("wait", f"stopping {container.name}")
                container.stop()
                _f("success", f"{container.name} stopped")
                _f("info", f"removing {container.name}")
                container.remove()
                _f("success", f"embedded nats removed")
        try:
            milvus_server.stop()
            _f("success", "embedded milvus server stopped")
        except Exception as e:
            _f("warn", f"embedded milvus can't be stopped\n{e}")

    def create_magnet(self):
        _f('wait', 'creating magnet with embedded cluster')
        magnet = Magnet(auto_config)
        return magnet

    def cleanup(self):
        self.client.images.prune()
        self.client.volumes.prune()
        _f("warn", "container engine pruned")
        try:
            milvus_server.cleanup()
            _f("success", "embedded cluster cleaned up")
        except Exception as e:
            _f("fatal",f"error cleaning up milvus\n{e}")

# class Electrode:
#     def __init__(self, config: dict = None):
#         self.config = config if config else _f('fatal', 'no config applied')
#     async def auto(self):
#         match self.config['JOB_TYPE']:
#             case 'index':
#                 self.reso = field.Resonator(f"{self.config['NATS_USER']}:{self.config['NATS_PASSWORD']}@{self.config['NATS_URL']}")
#                 self.embedder = memory.Embedder(self.config, create=self.config["CREATE"])
#                 await self.reso.on(category=self.config['NATS_CATEGORY'], session=self.config['NATS_SESSION'], stream=self.config['NATS_STREAM'], job=True)
#                 await self.reso.listen(cb=self.embedder.index, job_n=self.config['JOB_N'])