import nats
from magnet.utils import _f

class Charge:
    def __init__(self, server):
        self.server = server
    async def on(self):
        nc = await nats.connect(f'nats://{self.server}:4222')
        self.nc = await nc
        _f("success", f'connected to {self.server}')
    async def off(self):
        await self.nc.drain()
    async def pulse(self, frequency, packet):
        await self.nc.publish(frequency, packet)
        _f("info", f'{packet}')
        