from magnet.base import Magnet
from magnet.ic.field import Resonator
from magnet.ize.memory import Memory
import os

config = {
    "host": os.environ.get("HOST"),
    "credentials": None,
    "domain": None,
    "stream_name": os.environ.get("STREAM_NAME"),
    "category": os.environ.get("CATEGORY"),
    "kv_name": os.environ.get("KV_NAME"),
    "session": os.environ.get("SESSION"),
    "os_name": os.environ.get("OS_NAME"),
    "index": {
        "milvus_uri": os.environ.get("MILVUS_URI"),
        "milvus_port": int(os.environ.get("MILVUS_PORT")),
        "milvus_user": os.environ.get("MILVUS_USER"),
        "milvus_password": os.environ.get("MILVUS_PASSWORD"),
        "dimension": int(os.environ.get("DIMENSION")),
        "model": os.environ.get("MODEL"),
        "name": os.environ.get("INDEX_NAME"),
        "options": {
            'metric_type': os.environ.get("METRIC_TYPE"),
            'index_type': os.environ.get("INDEX_TYPE"),
            'params': {
                "efConstruction": int(os.environ.get("EF_CONSTRUCTION")),
                "M": int(os.environ.get("M"))
            }
        }
    }
}

magnet = Magnet(config)
memory = Memory(magnet)
reso = Resonator(magnet)

async def main():
    await magnet.align()
    await memory.on()
    await reso.on()

    async def message_handler(payload, msg):
        await memory.index(payload, msg, v=1)

    await reso.listen(cb=message_handler)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())