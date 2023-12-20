from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
from magnet.utils import _f
from dataclasses import dataclass
import json

@dataclass
class Payload:
    text: str
    document: str

class Embedder:
    def __init__(self, config, create=False):
        self.config = config
        self.model = SentenceTransformer(self.config['MODEL'])
        self.db = MilvusDB(self.config)
        self.db.on()
        if create:
            self.db.create(overwrite=True)
    def embed_and_store(self, payload, verbose=False):
        try:
            # payload = Payload(**json.loads(payload.data))
            _f('info','embedding payload') if verbose else None
            payload['embedding'] = self.model.encode(payload['text'], normalize_embeddings=True)
        except Exception as e:
            _f('fatal',e)
        try:
            _f('info','storing payload') if verbose else None
            self.db.collection.insert([
                    [payload['document']]
                    , [payload['text']]
                    , [payload['embedding']]
                ])
        except Exception as e:
            _f('fatal',e)
    def search(self, payload, limit=100, cb=None):
        payload = {'text':payload}
        payload['embedding'] = self.model.encode(payload['text'], normalize_embeddings=True)
        self.db.collection.load()
        _results = self.db.collection.search(
            data=[payload['embedding']],  # Embeded search value
            anns_field="embedding",  # Search across embeddings
            param={},
            limit = limit,  # Limit to top_k results per search
            output_fields=['text', 'document']
        )
        results = []
        for hits_i, hits in enumerate(_results):
            for hit in hits:
                results.append({
                    'text':hit.entity.get('text')
                    , 'document': hit.entity.get('document')
                    , 'distance': hit.distance
                })
        if cb:
            return cb(payload['text'], results)
        else:
            return results
    def delete(self, name=None):
        if name and name==self.config['INDEX']:
            try:
                self.db.delete_index()
            except Exception as e:
                _f('fatal',e)
        else:
            _f('fatal', "name doesn't match the connection or the connection doesn't exist")
            
class MilvusDB:
    def __init__(self, config):
        self.config = config
        self.fields = [
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name='document', dtype=DataType.VARCHAR, max_length=4096), 
            FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=self.config['DIMENSION'])
        ]
        self.schema = CollectionSchema(fields=self.fields)
        self.index_params = self.config['INDEX_PARAMS']

    def on(self):
        try:
            self.connection = connections.connect(host=self.config['MILVUS_URI'], port=19530)
            self.collection = Collection(name=self.config['INDEX'], schema=self.schema)
            _f('success',f"connected successfully to {self.config['MILVUS_URI']}")
        except Exception as e:
            _f('fatal',e)
    def off(self):
        try:
            self.connection = connections.disconnect(host=self.config['MILVUS_URI'], port=19530)
            _f('warn',f"disconnected from {self.config['MILVUS_URI']}")
        except Exception as e:
            _f('fatal',e)
    def create(self, overwrite=False):
        if utility.has_collection(self.config['INDEX']) and overwrite:
            utility.drop_collection(self.config['INDEX'])
        try:
            self.collection = Collection(name=self.config['INDEX'], schema=self.schema)
            self.collection.create_index(field_name="embedding", index_params=self.index_params)
            _f('success', f"{self.config['INDEX']} created")
        except Exception as e:
            _f('fatal',e)
    def delete_index(self):
        if utility.has_collection(self.config['INDEX']):
            utility.drop_collection(self.config['INDEX'])
            _f('warn', f"{self.config['INDEX']} deleted")