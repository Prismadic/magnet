from magnet.ic import field, filings
from magnet.ize import memory
from magnet.ron import llm

from magnet.utils.globals import _f

class Electrode:
    def __init__(self, config: dict = None):
        self.config = config if config else _f('fatal', 'no config applied')
    async def auto(self):
        match self.config['JOB_TYPE']:
            case 'index':
                self.reso = field.Resonator(f"{self.config['NATS_USER']}:{self.config['NATS_PASSWORD']}@{self.config['NATS_URL']}")
                self.embedder = memory.Embedder(self.config, create=self.config["CREATE"])
                await self.reso.on(category=self.config['NATS_CATEGORY'], session=self.config['NATS_SESSION'], stream=self.config['NATS_STREAM'], job=True)
                await self.reso.listen(cb=self.embedder.index, job_n=self.config['JOB_N'])

        

