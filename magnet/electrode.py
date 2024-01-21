from magnet.ic import field, filings
from magnet.ize import memory
from magnet.ron import llm

class Electrode:
    def __init__(self, config: dict = None):
        self.config = config if config else {
            "MILVUS_URI": "192.168.2.69"
            , "MILVUS_PORT": 19530
            , "MILVUS_USER": "root"
            , "MILVUS_PASSWORD": "Rrr/Yp6k#<M19rB3j1>Mi4Ta"
            , "NATS_URL": "192.168.2.69"
            , "NATS_USER": "my-user"
            , "NATS_PASSWORD": "T0pS3cr3t"
            , "NATS_CATEGORY": "non_nlp_chunks"
            , "NATS_STREAM": "documents"
            , "NATS_SESSION": "bge_large_en_v15"
            , "DIMENSION": 1024
            , "EMBEDDING_MODEL": "BAAI/bge-large-en-v1.5"
            , "INDEX": "bge_non_nlp"
            , "INDEX_PARAMS": {
                'metric_type': 'COSINE',
                'index_type':'HNSW',
                'params': {
                    "efConstruction": 40
                    , "M": 48
                },
            }
            , "JOB_TYPE": "index"
            , "JOB_N": 10
            , "GENERATION_MODEL": "mistralai/Mistral-7B-Instruct-v0.1"
            , "CREATE": True
        }
    async def auto(self):
        match self.config['JOB_TYPE']:
            case 'index':
                self.reso = field.Resonator(f"{self.config['NATS_USER']}:{self.config['NATS_PASSWORD']}@{self.config['NATS_URL']}")
                self.embedder = memory.Embedder(self.config, create=self.config["CREATE"])
                await self.reso.on(category=self.config['NATS_CATEGORY'], session=self.config['NATS_SESSION'], stream=self.config['NATS_STREAM'])
                await self.reso.listen(cb=self.embedder.index, job_n=self.config['JOB_N'])

        

