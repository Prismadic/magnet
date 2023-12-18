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
            self.sub = await self.nc.subscribe(self.frequency)
            _f("success", f'connected to {self.server}')
        except TimeoutError:
            _f('fatal', f'could not connect to {self.server}')
    async def off(self):
        await self.nc.drain()
    async def pulse(self, packet):
        try:
            await self.nc.publish(self.frequency, packet)
        except:
            _f('fatal', f'could not send data to {self.server}')

class Resonator:
    def __init__(self, server):
        self.server = server
    async def on(self, frequency: str = 'default', cb=print):
        self.frequency = frequency
        self.nc = await nats.connect(f'nats://{self.server}:4222')
        self.sub = await self.nc.subscribe(self.frequency)
        while True:
            msg = await self.sub.next_msg(timeout=1000)
            cb(msg) # customizable callback
    async def off(self):
        await self.sub.unsubscribe()
        _f('warn', f'unsubscribed from {self.frequency}')
        await self.nc.drain()
        _f('warn', f'disconnected from {self.server}')