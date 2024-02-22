from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from magnet.utils.globals import _f
import random, array

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

    def on(self, alias: str = "magnet"):
        """
        Establishes a connection to the Milvus server and creates a collection object.

        Returns:
            None

        Raises:
            Exception: If an error occurs during the connection to the Milvus server.

        """
        try:
            self.alias = alias
            _f('wait', f'connecting to {self.config["MILVUS_URI"]}')
            self.connection = connections.connect(
                host=self.config['MILVUS_URI']
                , port=self.config['MILVUS_PORT']
                , user=self.config['MILVUS_USER']
                , password=self.config['MILVUS_PASSWORD']
                , alias=self.alias
            ) \
                if 'MILVUS_PASSWORD' in self.config and 'MILVUS_USER' in self.config \
                else connections.connect(
                    host=self.config['MILVUS_URI']
                    , port=self.config['MILVUS_PORT']
                    , alias=self.alias
                )
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
        if utility.has_collection(self.config['INDEX'], using=self.alias) and overwrite:
            utility.drop_collection(self.config['INDEX'], using=self.alias)
        try:
            self.collection = Collection(name=self.config['INDEX'], schema=self.schema, using=self.alias)
            self.collection.create_index(field_name="embedding", index_params=self.index_params)
            _f('success', f"{self.config['INDEX']} created")
        except Exception as e:
            _f('fatal', e)

    def load(self):
        _f('wait', f'loading {self.config["INDEX"]} into memory, may take time')
        self.collection = Collection(name=self.config['INDEX'], schema=self.schema, using='magnet')
        self.collection.load()

    def initialize(self, user: str = 'magnet', password: str = '33011033'):
        try:
            _f('warn', f"initializing {self.config['MILVUS_URI']} using `root` to create '{user}'")
            utility.reset_password('root', 'Milvus', self._pw(), using='magnet')
            _f('success', f"secured root user successfully on {self.config['MILVUS_URI']}")
            utility.create_user(user, password, using='magnet')
            _f('success', f"created requested {user} on {self.config['MILVUS_URI']}")
            try:
                self.off()
                self.connection = connections.connect(
                    host=self.config['MILVUS_URI']
                    , port=self.config['MILVUS_PORT']
                    , user=user
                    , password=password
                )
                _f('success', 'Milvus has been initialized with your new credentials')
            except Exception as e:
                _f('fatal', e)
        except Exception as e:
            _f('fatal', e)
    
    def delete_index(self):
        """
        Deletes the index of a collection in MilvusDB.

        Returns:
            None

        """
        if utility.has_collection(self.config['INDEX'], using=self.alias):
            utility.drop_collection(self.config['INDEX'], using=self.alias)
            _f('warn', f"{self.config['INDEX']} deleted")

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
        _f('warn', f'Your Milvus `root` user password is now {password}')
        return password