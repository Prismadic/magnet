import nats
from magnet.utils import _f

class Charge:
    def __init__(self, server):
        self.server = server
    async def on(self):
        try:
            nc = await nats.connect(f'nats://{self.server}:4222')
            self.nc = nc
            _f("success", f'connected to {self.server}')
        except:
            _f('fatal', f'could not connect to {self.server}')
    async def off(self):
        await self.nc.drain()
    async def pulse(self, frequency, packet):
        await self.nc.publish(frequency, packet)
        _f("info", f'{packet}')
        