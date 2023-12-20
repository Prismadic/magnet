import nats, json
from magnet.utils import _f
from dataclasses import asdict, dataclass
from nats.errors import TimeoutError

@dataclass
class Payload:
    text: str
    document: str

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
    async def emp(self):
        await self.js.delete_stream(name=self.stream)
        _f('success', f'{self.stream} stream deleted')

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
        _f("info", f'consuming delta from <{self.category}> on 🛰️ <{self.stream}> w/ session 🧲 "{self.session}"')
        while True:
            try:
                msg = await self.sub.next_msg()
                try:
                    payload = Payload(**json.loads(msg.data))
                    await msg.ack()
                    cb(payload)
                except json.decoder.JSONDecodeError:
                    _f('fatal','invalid JSON')
            except TimeoutError or BrokenPipeError:
                _f("warn", f'retrying connection to {self.server}')
    async def off(self):
        await self.sub.unsubscribe()
        _f('warn', f'unsubscribed from {self.stream}')
        await self.nc.drain()
        _f('warn', f'disconnected from {self.server}')