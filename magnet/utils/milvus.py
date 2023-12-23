from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from magnet.utils.globals import _f

class MilvusDB:
    """
    A class that provides a high-level interface for interacting with a Milvus database.

    Args:
        config (dict): A configuration dictionary that contains the necessary parameters for connecting to the Milvus server and creating a collection.

    Attributes:
        config (dict): A configuration dictionary that contains the necessary parameters for connecting to the Milvus server and creating a collection.
        fields (list): A list of `FieldSchema` objects that define the fields of the collection.
        schema (CollectionSchema): A `CollectionSchema` object that represents the schema of the collection.
        index_params (dict): A dictionary that contains the parameters for creating an index on the collection.
        connection: The connection object to the Milvus server.
        collection: The collection object in MilvusDB.

    """

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
        """
        Establishes a connection to the Milvus server and creates a collection object.

        Returns:
            None

        Raises:
            Exception: If an error occurs during the connection to the Milvus server.

        """
        try:
            self.connection = connections.connect(host=self.config['MILVUS_URI'], port=19530)
            self.collection = Collection(name=self.config['INDEX'], schema=self.schema)
            _f('success', f"connected successfully to {self.config['MILVUS_URI']}")
        except Exception as e:
            _f('fatal', e)

    def off(self):
        """
        Disconnects from the Milvus server.

        Returns:
            None

        Raises:
            Exception: If an error occurs during the disconnection from the Milvus server.

        """
        try:
            self.connection = connections.disconnect(alias="magnet")
            _f('warn', f"disconnected from {self.config['MILVUS_URI']}")
        except Exception as e:
            _f('fatal', e)

    def create(self, overwrite=False):
        """
        Create a collection in MilvusDB and create an index on a specific field of the collection.

        Args:
            overwrite (bool, optional): A boolean flag indicating whether to overwrite the existing collection with the same name. Default is False.

        Returns:
            None

        Raises:
            Exception: If an error occurs during the creation of the collection or index.

        Example Usage:
            config = {
                'MILVUS_URI': 'localhost',
                'INDEX': 'my_collection',
                'DIMENSION': 128,
                'INDEX_PARAMS': {'index_type': 'IVF_FLAT', 'nlist': 100}
            }
            milvus_db = MilvusDB(config)
            milvus_db.create(overwrite=True)

        """
        if utility.has_collection(self.config['INDEX']) and overwrite:
            utility.drop_collection(self.config['INDEX'])
        try:
            self.collection = Collection(name=self.config['INDEX'], schema=self.schema)
            self.collection.create_index(field_name="embedding", index_params=self.index_params)
            _f('success', f"{self.config['INDEX']} created")
        except Exception as e:
            _f('fatal', e)

    def delete_index(self):
        """
        Deletes the index of a collection in MilvusDB.

        Returns:
            None

        """
        if utility.has_collection(self.config['INDEX']):
            utility.drop_collection(self.config['INDEX'])
            _f('warn', f"{self.config['INDEX']} deleted")