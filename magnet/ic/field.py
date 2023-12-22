import nats, json
from magnet.utils import _f
from dataclasses import asdict
from nats.errors import TimeoutError
from .utils.data_classes import *
from nats.js.api import StreamConfig

class Charge:
    def __init__(self, server):
        self.server = server

    async def on(self, category: str = 'no_category', stream: str = 'documents', create: bool = False):
        self.category = category
        self.stream = stream
        try:
            nc = await nats.connect(f'nats://{self.server}:4222')
            self.nc = nc
            self.js = self.nc.jetstream()
            self.js.purge_stream
            streams = await self.js.streams_info()
            if self.stream not in [x.config.name for x in streams] or self.category not in [x.config.subjects for x in streams]:
                try:
                    if self.stream not in [x.config.name for x in streams]:
                        _f("wait", f'creating {self.stream}') \
                        , await self.js.add_stream(name=self.stream, subjects=[self.category]) \
                            if create else _f("warn", f"couldn't create {stream} on {self.server}")
                        streams = await self.js.streams_info()
                    if self.category not in [x.config.subjects for x in streams if x.config.name == self.stream][0]:
                        subjects = [x.config.subjects for x in streams if x.config.name == self.stream][0]
                        subjects.append(self.category)
                        await self.js.update_stream(StreamConfig(
                            name = self.stream
                            , subjects = subjects
                        ))
                        _f("success", f'created [{self.category}] on\n🛰️ stream: {self.stream}')
                except:
                    _f('fatal', f"couldn't create {stream} on {self.server}")
        except TimeoutError:
            _f('fatal', f'could not connect to {self.server}')
        _f("success", f'connected to [{self.category}] on\n🛰️ stream: {self.stream}')
    async def off(self):
        await self.nc.drain()
        _f('warn', f'disconnected from {self.server}')
    async def pulse(self, payload):
        try:
            bytes_ = json.dumps(asdict(payload), separators=(', ', ':')).encode('utf-8')
        except Exception as e:
            _f('fatal', f'invalid JSON\n{e}')
        try:
            await self.js.publish(self.category, bytes_)
        except Exception as e:
            _f('fatal', f'could not send data to {self.server}\n{e}')
    async def emp(self, name=None):
        if name and name==self.stream:
            await self.js.delete_stream(name=self.stream)
            _f('warn', f'{self.stream} stream deleted')
        else:
            _f('fatal', "name doesn't match the stream or stream doesn't exist")
    async def reset(self, name=None):
        if name and name==self.category:
            await self.js.purge_stream(name=self.stream, subject=self.category)
            _f('warn', f'{self.category} category deleted')
        else:
            _f('fatal', "name doesn't match the stream category or category doesn't exist")

class Resonator:
    def __init__(self, server):
        self.server = server

    async def on(self, category: str = 'no_category', stream: str = 'documents', cb=print, session='magnet'):
        self.category = category
        self.stream = stream
        self.session = session
        _f('wait',f'connecting to {self.server}')
        try:
            self.nc = await nats.connect(f'nats://{self.server}:4222')
            self.js = self.nc.jetstream()
            self.sub = await self.js.subscribe(self.category, durable=self.session)
            _f("success", f'connected to {self.server}')
        except TimeoutError:
            _f("fatal", f'could not connect to {self.server}')
        _f("info", f'consuming delta from [{self.category}] on\n🛰️ stream: {self.stream}\n🧲 session: "{self.session}"')
        while True:
            try:
                msg = await self.sub.next_msg(timeout=60)
                payload = Payload(**json.loads(msg.data))
                try:
                    await msg.ack()
                    await cb(payload)
                except Exception as e:
                    _f("warn", f'retrying connection to {self.server}')
            except Exception as e:
                _f('fatal','invalid JSON')
    async def info(self, session: str = None):
        jsm = await self.js.consumer_info(stream=self.stream, consumer=session)
        _f('info',json.dumps(jsm.config.__dict__, indent=2))
    async def off(self):
        await self.sub.unsubscribe()
        _f('warn', f'unsubscribed from {self.stream}')
        await self.nc.drain()
        _f('warn', f'disconnected from {self.server}')