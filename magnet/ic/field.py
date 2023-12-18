import nats
from magnet.utils import _f

class Charge:
    def __init__(self, server):
        self.server = server

    async def on(self, frequency: str = 'default'):
        self.frequency = frequency
        try:
            nc = await nats.connect(f'nats://{self.server}:4222')
            self.nc = nc
            self.js = self.nc.jetstream()
            await self.js.add_stream(name=self.frequency, subjects=[self.frequency])
            self.sub = await self.js.pull_subscribe(self.frequency, 'workers')
            _f("success", f'connected to {self.server}')
        except TimeoutError:
            _f('fatal', f'could not connect to {self.server}')
    async def off(self):
        await self.sub.unsubscribe()
        _f('warn', f'unsubscribed from {self.frequency}')
        await self.nc.drain()
        _f('warn', f'disconnected from {self.server}')
    async def pulse(self, packet):
        try:
            await self.js.publish(self.frequency, packet)
        except Exception as e:
            print(e)
            _f('fatal', f'could not send data to {self.server}')
    async def emp(self, name: str = None):
        await self.js.delete_stream(name=name if name is not None else self.frequency)
        _f('success', f'{self.frequency} stream deleted')

class Resonator:

    def __init__(self, server):
        self.server = server
    async def on(self, frequency: str = 'default', cb=print):
        self.frequency = frequency
        self.nc = await nats.connect(f'nats://{self.server}:4222')
        self.js = self.nc.jetstream()
        self.sub =  await self.js.pull_subscribe(self.frequency, stream=self.frequency)
        while True:
            msgs = await self.sub.fetch(batch=10, timeout=60)
            for msg in msgs:
                cb(msg)
    async def off(self):
        await self.js.unsubscribe()
        _f('warn', f'unsubscribed from {self.frequency}')
        await self.nc.drain()
        _f('warn', f'disconnected from {self.server}')