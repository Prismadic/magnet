from sentence_transformers import SentenceTransformer
from magnet.utils.globals import _f, Utils, break_into_chunks
from magnet.utils.index.milvus import *
from magnet.utils.data_classes import EmbeddingPayload
import re
from magnet.ic.field import Charge, Magnet

from typing import Optional

class Memory:
    """
    The Embedder class is responsible for embedding text using a pre-trained sentence transformer model and storing or sending the embeddings for further processing. It utilizes the Milvus database for storing and searching the embeddings.

    Args:
        config (Config): A Config instance containing the configuration parameters for the Embedder class.
        create (bool, optional): If set to True, a connection to the Milvus database will be created. Defaults to False.

    Attributes:
        config (Config): A Config instance containing the configuration parameters for the Embedder class.
        model (SentenceTransformer): An instance of the SentenceTransformer class from the sentence_transformers library, used for text embedding.
        db (MilvusDB): An instance of the MilvusDB class from the magnet.utils.milvus module, used for connecting to the Milvus database.
    """

    def __init__(self, magnet: Magnet = None):
        self.config = magnet.config
        self._model = None
        
    async def on(self, create: bool = False, initialize: bool = False):
        self._model = SentenceTransformer(self.config.index.model, device=Utils().check_cuda())
        _f('info', f'loading into {self._model.device}')
        self.db = MilvusDB(self.config)
        await self.db.on()
        if create:
            await self.db.create(overwrite=True)
        await self.db.load()
        if initialize:
            self.db.initialize()

    async def index(self, payload, msg, field=None, v=False, instruction="Represent this information for searching relevant passages: "):
        if not msg or not payload:
            return _f('fatal', 'no field message and/or payload to ack!')
        if field:
            self.field = field
        try:
            _f('info', f'encoding payload\n{payload}') if v else None
            _text = re.sub(r'\s+', ' ', payload.text)
            _text = _text.replace('\n', '')
            text_to_encode = f"{instruction} {_text}"
            num_tokens = len(text_to_encode.split())
            if num_tokens > self.config.index.dimension:
                chunks = break_into_chunks(payload.text, self.config.index.dimension)
                for chunk in chunks:
                    embedding = self._model.encode(chunk, normalize_embeddings=True)
                    if not await self.is_dupe(q=embedding):
                        self.db.collection.insert([
                            [payload.document], [chunk], [embedding.tolist()]
                        ])
                        if field:
                            payload = EmbeddingPayload(
                                model=self.config.index.model,
                                embedding=embedding.tolist(),
                                text=chunk,
                                document=payload.document
                            )
                            _f('info', f'sending payload\n{payload}') if v else None
                            await self.field.pulse(payload)
            else:
                embedding = self._model.encode(payload.text, normalize_embeddings=True)
                if not await self.is_dupe(q=embedding):
                    self.db.collection.insert([
                        [payload.document], [payload.text], [embedding.tolist()]
                    ])
                    if field:
                        payload = EmbeddingPayload(
                            model=self.config.index.model,
                            embedding=embedding.tolist(),
                            text=payload.text,
                            document=payload.document
                        )
                        _f('info', f'sending payload\n{payload}') if v else None
                        await self.field.pulse(payload)
                else:
                    _f('warn', f'embedding exists\n{payload}')
            await msg.ack_sync()
            _f('success', f'embedding indexed\n{payload}') if v else None
        except Exception as e:
            await msg.term()
            _f('fatal', e)

    def search(self, payload, limit: int = 100, cb: Optional[callable] = None, instruction: str = "Represent this information for searching relevant passages: "):
        payload = EmbeddingPayload(
            text=payload,
            embedding=self._model.encode(
                f"{instruction} {payload}", normalize_embeddings=True),
            model=self.config.index.model,
            document="none"
        )

        _results = self.db.collection.search(
            data=[payload.embedding],
            anns_field="embedding",
            param=self.config.index.options,
            limit=limit,
            output_fields=['text', 'document']
        )
        results = []
        for hits_i, hits in enumerate(_results):
            for hit in hits:
                results.append({
                    'text': hit.entity.get('text'),
                    'document': hit.entity.get('document'),
                    'distance': hit.distance
                })
        if cb:
            return cb(payload.text, results)
        else:
            return results

    async def info(self):
        return self.db.collection

    async def disconnect(self):
        await self.db.off()

    async def delete(self, name: str = None):
        if name and name == self.config.index.name:
            try:
                return await self.db.delete_index()
            except Exception as e:
                _f('fatal', e)
        else:
            _f('fatal', "name doesn't match the connection or the connection doesn't exist")
    
    async def is_dupe(self, q: str = None):
        match = self.db.collection.search(
            data=[q],
            anns_field="embedding",
            param=self.config.index.options,
            output_fields=['text', 'document'],
            limit=1
        )
        return True if match and match[0] and match[0][0].distance >= 0.98 else False