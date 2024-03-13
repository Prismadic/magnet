
import nats

from magnet.utils.globals import _f
from magnet.utils.data_classes import PrismConfig, IndexConfig

class Prism:
    def __init__(self, config: PrismConfig | dict = None):
        try:
            if isinstance(config, dict):
                config = PrismConfig(**config)
            if isinstance(config.index, dict):
                config.index = IndexConfig(**config.index)
            elif not isinstance(config, PrismConfig):
                _f("fatal", "config must be a PrismConfig instance or a dictionary")
                raise ValueError
        except Exception as e:
            raise e
        self.config = config
        self.nc = None
        self.js = None
        self.kv = None
        self.os = None

    async def align(self):
        """Connect to NATS and setup configurations."""
        try:
            self.nc = await nats.connect(f'{"nats://" if not self.config.credentials else "tls://"}{self.config.host}:4222',user_credentials=self.config.credentials)
            self.js = self.nc.jetstream(
                domain=self.config.domain 
            ) if self.config.domain is not None else self.nc.jetstream()
            if self.config.name:
                await self._setup_stream()
            if self.config.kv_name:
                self.kv = await self._setup_kv()
            if self.config.os_name:
                self.os = await self._setup_object_store()
            _f("success", f"🧲 connected to \n💎 {self.config} ")
            return [self.js, self.kv, self.os]
        except Exception as e:
            _f("fatal", f"could not align {self.config.host}\n{e}")
            return None

    async def _setup_stream(self):
        """Setup stream."""
        try:
            await self.js.stream_info(self.config.name)
        except Exception as e:
            _f("warn", f"Stream {self.config.name} not found, creating")
            await self.js.add_stream(name=self.config.name, subjects=["magnet"])
            _f("success", f"created `{self.config.name}` with default category `magnet`")
        return await self.js.stream_info(self.config.name)

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
            _f("warn", f"Object Store {self.config.kv_name} not found, creating")
            await self.js.create_object_store(bucket=self.config.os_name)
            _f("success", f"created `{self.config.os_name}`")
        return await self.js.object_store(self.config.os_name)

    async def off(self):
        """Disconnects from the NATS server and prints a warning message."""
        await self.nc.drain()
        _f('warn', f'disconnected from {self.config.host}')


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