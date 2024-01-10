from sentence_transformers import SentenceTransformer
from magnet.utils.globals import _f
from magnet.utils.milvus import *
from magnet.utils.data_classes import EmbeddingPayload


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

    def __init__(self, config, create=False):
        self.config = config
        self.model = SentenceTransformer(self.config['MODEL'])
        self.db = MilvusDB(self.config)
        self.db.on()
        if create:
            self.db.create()

    async def embed_and_store(self, payload, verbose=False, field=None):
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
        if field:
            self.field = field
        try:
            _f('info', 'embedding payload') if verbose else None
            payload.embedding = self.model.encode(
                f"Represent this sentence for searching relevant passages: {payload.text}", normalize_embeddings=True)
        except Exception as e:
            _f('fatal', e)
        else:
            try:
                _f('info', 'storing payload') if verbose else None
                if not self.is_dupe(q=payload.embedding):
                    self.db.collection.insert([
                        [payload.document], [payload.text], [payload.embedding]
                    ])
                    self.db.collection.flush(collection_name_array=[
                                            self.config['INDEX']])
            except Exception as e:
                _f('fatal', e)

    async def embed_and_charge(self, payload, verbose=False, field=None):
        """
        Embeds the given payload using a pre-trained sentence transformer model and sends it to a specified field for further processing.

        Args:
            payload (object): The payload to be embedded and charged. It should contain the `text` and `document` attributes.
            verbose (bool, optional): If set to True, additional information will be logged during the embedding process. Defaults to False.
            field (object): The field to which the encoded payload will be sent for further processing.

        Raises:
            ValueError: If the `field` parameter is not provided.

        Returns:
            None

        Example Usage:
            config = {
                'MODEL': 'bert-base-nli-mean-tokens',
                'INDEX': 'my_index',
                'INDEX_PARAMS': {'nprobe': 16}
            }
            embedder = Embedder(config)
            await embedder.embed_and_charge(payload, verbose=True, field=my_field)

        """
        if field:
            self.field = field
        else:
            raise ValueError('field is required')
        try:
            _f('info', 'embedding payload') if verbose else None
            payload = EmbeddingPayload(
                model=self.config['MODEL'],
                embedding=self.model.encode(
                    f"Represent this sentence for searching relevant passages: {payload.text}", normalize_embeddings=True).tolist(),
                text=payload.text,
                document=payload.document
            )
            _f('info', f'sending payload\n{payload}') if verbose else None
            await self.field.pulse(payload)
        except Exception as e:
            _f('fatal', e)

    def search(self, payload, limit=100, cb=None):
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
                f"Represent this sentence for searching relevant passages: {payload}", normalize_embeddings=True),
            model=self.config['MODEL']
        )
        self.db.collection.load()
        _results = self.db.collection.search(
            data=[payload.embedding],  # Embeded search value
            anns_field="embedding",  # Search across embeddings
            param=self.config['INDEX_PARAMS'],
            limit=limit,  # Limit to top_k results per search
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
            , anns_field = "embeddings"
            , param=self.config['INDEX_PARAMS']
            , output_fields=['text', 'id', 'documentId']
            , limit=1
        )
        return True if sum(match[0].distances) == 0.0 else False