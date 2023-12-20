import nats, json
from magnet.utils import _f
from dataclasses import asdict
from nats.errors import TimeoutError, SlowConsumerError
from .data_classes import Payload, GeneratedPayload

class Charge:
    def __init__(self, server):
        self.server = server

    async def on(self, category: str = 'no_category', stream: str = 'documents'):
        self.category = category
        self.stream = stream
        try:
            nc = await nats.connect(f'nats://{self.server}:4222')
            self.nc = nc
            self.js = self.nc.jetstream()
            await self.js.add_stream(name=self.stream, subjects=[self.category])
            _f("success", f'connected to {self.server}')
        except TimeoutError:
            _f('fatal', f'could not connect to {self.server}')
    async def info(self, session: str = None):
        jsm = await self.js.consumer_info(stream=self.stream, consumer=session)
        _f('info',json.dumps(jsm.config.__dict__, indent=2))
    async def off(self):
        await self.sub.unsubscribe()
        _f('warn', f'unsubscribed from {self.stream}')
        await self.nc.drain()
        _f('warn', f'disconnected from {self.server}')
    async def pulse(self, packet, document_id):
        try:
            payload = Payload(text=packet,document=document_id)
            bytes_ = json.dumps(asdict(payload)).encode()
        except:
            _f('fatal', 'invalid JSON')
        try:
            await self.js.publish(self.category, bytes_)
        except Exception as e:
            print(e)
            _f('fatal', f'could not send data to {self.server}')
    async def emp(self, name=None):
        if name and name==self.stream:
            await self.js.delete_stream(name=self.stream)
            _f('warn', f'{self.stream} stream deleted')
        else:
            _f('fatal', "name doesn't match the connection or connection doesn't exist")

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
                msg = await self.sub.next_msg()
                try:
                    payload = Payload(**json.loads(msg.data))
                    await msg.ack()
                    cb(payload)
                except json.decoder.JSONDecodeError or BrokenPipeError or SlowConsumerError:
                    _f('fatal','invalid JSON') if not SlowConsumerError else _f('warn', 'processing speed is slow')
            except TimeoutError or BrokenPipeError or SlowConsumerError:
                _f("warn", f'retrying connection to {self.server}') if not SlowConsumerError else _f('warn', 'processing speed is slow')
    async def off(self):
        await self.sub.unsubscribe()
        _f('warn', f'unsubscribed from {self.stream}')
        await self.nc.drain()
        _f('warn', f'disconnected from {self.server}')