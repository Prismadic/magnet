from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from magnet.utils import _f

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
            self.connection = connections.disconnect(alias="magnet")
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