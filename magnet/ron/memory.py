from sentence_transformers import SentenceTransformer
from magnet.utils import _f
from .utils.milvus import *
from dataclasses import dataclass
import json

@dataclass
class EmbeddingPayload:
    document: str
    embedding: list
    text: str
    model: str

class Embedder:
    def __init__(self, config, create=False, field=None):
        self.config = config
        self.model = SentenceTransformer(self.config['MODEL'])
        self.db = MilvusDB(self.config)
        self.db.on()
        if create:
            self.db.create(overwrite=True)
    async def embed_and_store(self, payload, verbose=False):
        try:
            _f('info','embedding payload') if verbose else None
            payload.embedding = self.model.encode(payload.text, normalize_embeddings=True)
        except Exception as e:
            _f('fatal',e)
        else:
            try:
                _f('info','storing payload') if verbose else None
                self.db.collection.insert([
                        [payload.document]
                        , [payload.text]
                        , [payload.embedding]
                    ])
            except Exception as e:
                _f('fatal',e)
    async def embed_and_charge(self, payload, verbose=False, field=None):
        if field:
            self.field = field
        else:
            _f('fatal','field is required')
        try:
            _f('info','embedding payload') if verbose else None
            payload = EmbeddingPayload(
                    model = self.config['MODEL']
                    , embedding = self.model.encode(payload.text, normalize_embeddings=True).tolist()
                    , text = payload.text
                    , document = payload.document
                )
            _f('info',f'sending payload\n{payload}') if verbose else None
            await self.field.pulse(payload)
        except Exception as e:
            _f('fatal',e)

    def search(self, payload, limit=100, cb=None):
        payload = EmbeddingPayload({
                    'text': payload
                    , 'embedding': self.model.encode(payload.data.text, normalize_embeddings=True)
                    , 'model': self.config.model
                })
        self.db.collection.load()
        _results = self.db.collection.search(
            data=[payload.data.embedding],  # Embeded search value
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
            return cb(payload.data.text, results)
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