import nats, json
from magnet.utils import _f
from dataclasses import asdict, dataclass

@dataclass
class Payload:
    text: str
    document: str

class Charge:
    def __init__(self, server):
        self.server = server

    async def on(self, frequency: str = 'no_category', stream: str = 'documents'):
        self.frequency = frequency
        self.stream = stream
        try:
            nc = await nats.connect(f'nats://{self.server}:4222')
            self.nc = nc
            self.js = self.nc.jetstream()
            await self.js.add_stream(name=self.stream, subjects=[self.frequency])
            self.sub = await self.js.pull_subscribe(self.frequency, 'magnet-charge')
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
            await self.js.publish(self.frequency, bytes_)
        except Exception as e:
            print(e)
            _f('fatal', f'could not send data to {self.server}')
    async def emp(self):
        await self.js.delete_stream(name=self.stream)
        _f('success', f'{self.frequency} stream deleted')

class Resonator:

    def __init__(self, server):
        self.server = server
    async def on(self, frequency: str = 'no_category', stream: str = 'documents', cb=print):
        self.frequency = frequency
        self.stream = stream
        self.nc = await nats.connect(f'nats://{self.server}:4222')
        self.js = self.nc.jetstream()
        self.sub =  await self.js.pull_subscribe(self.frequency, stream=self.stream)
        while True:
            msgs = await self.sub.fetch(batch=10, timeout=60)
            for msg in msgs:
                try:
                    payload = Payload(**json.loads(msg.data))
                    cb(payload)
                except json.decoder.JSONDecodeError:
                    _f('fatal','invalid JSON')
    async def off(self):
        await self.sub.unsubscribe()
        _f('warn', f'unsubscribed from {self.stream}')
        await self.nc.drain()
        _f('warn', f'disconnected from {self.server}')