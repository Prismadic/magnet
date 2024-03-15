from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from magnet.utils.globals import _f
from magnet.utils.data_classes import MagnetConfig
import random, array

class MilvusDB:
    def __init__(self, config: MagnetConfig):
        self.config = config
        self.fields = [
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name='document', dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=self.config.index.dimension)
        ]
    async def on(self):
        try:
            _f('wait', f'connecting to {self.config.index.milvus_uri}')
            self.connection = connections.connect(
                host=self.config.index.milvus_uri,
                port=self.config.index.milvus_port,
                user=self.config.index.milvus_user,
                password=self.config.index.milvus_password,
                alias=self.config.session
            )
            self.schema = CollectionSchema(fields=self.fields)
            _f('success', f"connected successfully to {self.config.index.milvus_uri}")
        except Exception as e:
            _f('fatal', e)

    async def off(self):
        try:
            connections.disconnect(alias=self.config.session)
            return _f('warn', f'disconnected from {self.config.index.milvus_uri}')
        except Exception as e:
            return _f('fatal', e)

    async def create(self, overwrite=False):
        try:
            if utility.has_collection(self.config.index.name, using=self.config.session) and overwrite:
                utility.drop_collection(self.config.index.name, using=self.config.session)
            self.collection = Collection(name=self.config.index.name, schema=self.schema, using=self.config.session)
            self.collection.create_index(field_name="embedding", index_params=self.config.index.options)
            _f('success', f"{self.config.index.name} created")
        except Exception as e:
            _f('fatal', e)

    async def load(self):
        _f('wait', f'loading {self.config.index.name} into memory, may take time')
        self.collection = Collection(name=self.config.index.name, schema=self.schema, using=self.config.session)
        self.collection.load()
    
    def initialize(self, user: str = 'magnet', password: str = '33011033'):
        try:
            _f('warn', f"initializing {self.config.index.milvus_uri} using `root` to create '{user}'")
            utility.reset_password('root', 'Milvus', self._pw(), using='magnet')
            _f('warn', f'Your Milvus `root` user password is now {password}')
            _f('success', f"secured root user successfully on {self.config.index.milvus_uri}")
            utility.create_user(user, password, using='magnet')
            _f('success', f"created requested {user} on {self.config.index.milvus_uri}")
            try:
                self.off()  # Disconnect first before re-connecting with new credentials
                self.connection = connections.connect(
                    host=self.config.index.milvus_uri,
                    port=self.config.index.milvus_port,
                    user=user,
                    password=password
                )
                _f('success', 'Milvus has been initialized with your new credentials')
            except Exception as e:
                _f('fatal', e)
        except Exception as e:
            _f('fatal', e)

    async def delete_index(self):
        if utility.has_collection(self.config.index.name, using=self.config.session):
            utility.drop_collection(self.config.index.name, using=self.config.session)
            _f('warn', f"Index for {self.config.index.name} deleted")
    
    def list_indices(self):
        return utility.list_collections(using=self.config.session)

    def _pw(self):
        MAX_LEN = 24
        DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        LOCASE_CHARACTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                             'i', 'j', 'k', 'm', 'n', 'o', 'p', 'q',
                             'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
                             'z']
        UPCASE_CHARACTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                             'I', 'J', 'K', 'M', 'N', 'O', 'P', 'Q',
                             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                             'Z']
        SYMBOLS = ['@', '#', '$', '(', ')']
        COMBINED_LIST = DIGITS + UPCASE_CHARACTERS + LOCASE_CHARACTERS + SYMBOLS
        rand_digit = random.choice(DIGITS)
        rand_upper = random.choice(UPCASE_CHARACTERS)
        rand_lower = random.choice(LOCASE_CHARACTERS)
        rand_symbol = random.choice(SYMBOLS)
        temp_pass = rand_digit + rand_upper + rand_lower + rand_symbol
        for x in range(MAX_LEN - 4):
            temp_pass = temp_pass + random.choice(COMBINED_LIST)
            temp_pass_list = array.array('u', temp_pass)
            random.shuffle(temp_pass_list)
        password = ""
        for x in temp_pass_list:
            password = password + x
        return password
