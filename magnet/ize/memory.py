from sentence_transformers import SentenceTransformer
from magnet.utils.globals import _f
from magnet.utils.milvus import *
from magnet.utils.data_classes import EmbeddingPayload
from magnet.utils.globals import Utils

class Embedder:
    """
    The Embedder class is responsible for embedding text using a pre-trained sentence transformer model and storing or sending the embeddings for further processing. It utilizes the Milvus database for storing and searching the embeddings.

    Args:
        config (dict): A dictionary containing the configuration parameters for the Embedder class, including the model name, index name, and index parameters.
        create (bool, optional): If set to True, a connection to the Milvus database will be created. Defaults to False.

    Attributes:
        config (dict): A dictionary containing the configuration parameters for the Embedder class, including the model name, index name, and index parameters.
        model (SentenceTransformer): An instance of the SentenceTransformer class from the sentence_transformers library, used for text embedding.
        db (MilvusDB): An instance of the MilvusDB class from the magnet.utils.milvus module, used for connecting to the Milvus database.

    Methods:
        embed_and_store(self, payload, verbose=False, field=None): Embeds the given payload using a pre-trained sentence transformer model and stores it in the Milvus database.
        embed_and_charge(self, payload, verbose=False, field=None): Embeds the given payload using a pre-trained sentence transformer model and sends it to a specified field for further processing.

    """

    def __init__(self, config, create=False, initialize=False):
        self.config = config
        self.model = SentenceTransformer(self.config['MODEL'], device=Utils().check_cuda())
        _f('info', f'loading into {self.model.device}')
        self.db = MilvusDB(self.config)
        self.db.on()
        if create:
            self.db.create(overwrite=True)
            self.db.load()
            if initialize:
                self.db.initialize()
            return
        self.db.load()
    async def index(self, payload, msg, verbose=False, field=None, charge=False, instruction: str = "Represent this sentence for searching relevant passages: "):
        """
        Embeds the given payload using a pre-trained sentence transformer model and stores it in a Milvus database.

        Args:
            payload (object): An object containing the text and document attributes to be embedded and stored.
            verbose (bool, optional): A boolean indicating whether additional information should be logged during the embedding process. Defaults to False.
            field (object, optional): An object representing the field to which the encoded payload will be sent for further processing.

        Returns:
            None

        Raises:
            Exception: If an error occurs during the embedding or storing process.

        """
        if not msg or not payload:
            return _f('fatal', 'no field message and/or payload to ack!')
        if field:
            self.field = field
        try:
            _f('info', f'encoding payload\n{payload}') if verbose else None
            payload.embedding = self.model.encode(
                f"{instruction} {payload.text}", normalize_embeddings=True)
        except Exception as e:
            return _f('fatal', e)
        await msg.in_progress()
        try:
            _f('info', f'indexing payload') if verbose else None
            if not self.is_dupe(q=payload.embedding):
                self.db.collection.insert([
                    [payload.document], [payload.text], [payload.embedding]
                ])
                # self.db.collection.flush(collection_name_array=[self.config['INDEX']]) # https://milvus.io/docs/v1.1.1/flush_python.md#:~:text=Milvus%20also%20performs%20an%20automatic,fixed%20interval%20(1%20second).&text=After%20calling%20delete%20%2C%20you%20can,data%20is%20no%20longer%20recoverable.
                if charge:
                    payload = EmbeddingPayload(
                        model=self.config['MODEL'],
                        embedding=self.model.encode(
                            f"{instruction} {payload.text}", normalize_embeddings=True).tolist(),
                        text=payload.text,
                        document=payload.document
                    )
                    if field:
                        _f('info', f'sending payload\n{payload}') if verbose else None
                        await self.field.pulse(payload)
                await msg.ack_sync()
                _f('success', f'embedding indexed\n{payload}') if verbose else None
            else:
                await msg.ack_sync()
                _f('warn', f'embedding exists already\n{payload}') if verbose else None
                
        except Exception as e:
            await msg.term()
            _f('fatal', e)

    def search(self, payload, limit=100, cb=None, instruction="Represent this sentence for searching relevant passages: "):
        """
        Search for relevant passages based on a given input payload.

        Args:
            payload (str): The search query or input payload.
            limit (int, optional): The maximum number of results to return. Defaults to 100.
            cb (function, optional): A callback function to process the results. Defaults to None.

        Returns:
            list: A list of dictionaries, each containing the text, document, and distance of a relevant passage.
        """
        payload = EmbeddingPayload(
            text=payload,
            embedding=self.model.encode(
                f"{instruction} {payload}", normalize_embeddings=True),
            model=self.config['MODEL'],
            document="none"
        )
        
        _results = self.db.collection.search(
            data=[payload.embedding],
            anns_field="embedding",
            param=self.config['INDEX_PARAMS'],
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

    def info(self):
        return self.db.collection

    def disconnect(self):
        return self.db.off()

    def delete(self, name=None):
        """
        Delete an index from the Milvus database.

        Args:
            name (str, optional): The name of the index to be deleted. If not provided, no index will be deleted.

        Returns:
            None

        Raises:
            Exception: If an error occurs during the deletion process.

        Example Usage:
            config = {
                'MODEL': 'bert-base-nli-mean-tokens',
                'INDEX': 'my_index',
                'INDEX_PARAMS': {'nprobe': 16}
            }
            embedder = Embedder(config)
            embedder.delete('my_index')
        """
        if name and name == self.config['INDEX']:
            try:
                self.db.delete_index()
            except Exception as e:
                _f('fatal', e)
        else:
            _f('fatal', "name doesn't match the connection or the connection doesn't exist")
    
    def is_dupe(self, q):
        """
        Check if a given query is a duplicate in the Milvus database.

        Args:
            q (object): The query embedding to check for duplicates.

        Returns:
            bool: True if the query embedding is a duplicate in the Milvus database, False otherwise.
        """
        match = self.db.collection.search(
            data=[q]
            , anns_field = "embedding"
            , param=self.config['INDEX_PARAMS']
            , output_fields=['text', 'document']
            , limit=1
        )
        return True if sum(match[0].distances) >= 0.99 and len(match[0])>0 else False