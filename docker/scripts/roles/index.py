import os
from magnet.ize import memory
from magnet.ic import field
from magnet.utils.globals import _f

config = {
    "MILVUS_URI": os.getenv('MILVUS_HOST'),
    "MILVUS_PORT": int(os.getenv('MILVUS_PORT', 19530)),
    "MILVUS_USER": os.getenv('MILVUS_USER'),
    "MILVUS_PASSWORD": os.getenv('MILVUS_PASSWORD'),
    "DIMENSION": int(os.getenv('DIMENSION', 1024)),
    "MODEL": os.getenv('MODEL'),
    "INDEX": os.getenv('INDEX'),
    "INDEX_PARAMS": {
        'metric_type': 'COSINE',
        'index_type':'HNSW',
        'params': {
            "efConstruction": int(os.getenv('efConstruction', 40)),
            "M": int(os.getenv('M', 48))
        },
    }
}

embedder = memory.Embedder(config)

async def handle_payload(payload, msg):
    await embedder.index(payload, msg, verbose=True)

async def main():
    try:
        reso = field.Resonator(os.getenv('RESONATOR_URI'))
    except Exception as e:
        _f('warn', e)
        main()
    await reso.on(category=os.getenv('CATEGORY'), session=os.getenv('SESSION'), stream=os.getenv('STREAM'))
    await reso.listen(cb=handle_payload)