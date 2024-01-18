import pandas as pd
import os
from magnet.utils.globals import _f, Utils
from tqdm import tqdm
from magnet.utils.data_classes import *

class Processor:
    """
    The `Processor` class is responsible for loading, processing, and saving data. It provides methods for loading data from different file formats, splitting the data into smaller chunks, and saving the processed data to a file.

    Example Usage:
        # Initialize the Processor class object
        processor = Processor()

        # Load data into the processor
        processor.load(raw_data, text_column="clean", id_column="id")

        # Process and split the loaded data into smaller chunks
        await processor.process(path="processed_data.csv", splitter=None, nlp=True)

        # Save the processed data to a file
        processor.save(filename="processed_data.csv", raw=processor.df)

    Main functionalities:
        - Loading data from different file formats (csv, json, xlsx, parquet) or a pandas DataFrame
        - Splitting the loaded data into smaller chunks either by sentences using natural language processing (NLP) or by a fixed character length
        - Saving the processed data to a file in different formats (csv, json, xlsx, parquet)

    Methods:
        - __init__(): Initializes the Processor class object and sets the df attribute to None and creates an instance of the Utils class.
        - save(filename: str = None, raw: pd.DataFrame = None): Saves the pandas DataFrame to a file with the specified filename and file format.
        - load(raw: str | pd.DataFrame = None, text_column: str = "clean", id_column: str = 'id'): Loads data into the df attribute of the Processor class.
        - process(path: str = None, splitter: any = None, nlp=True): Processes and splits the loaded data into smaller chunks.
        - default_splitter(data, window_size=768, overlap=76, nlp=True): Splits the given input data into smaller chunks either by sentences or by a fixed character length.

    Fields:
        - df: A pandas DataFrame that stores the loaded data.
        - utils: An instance of the Utils class that provides utility functions.
    """
    def __init__(self):
        self.df = None
        self.utils = Utils()
        
    def save(self, filename: str = None, raw: pd.DataFrame = None):
        """
        Save the pandas DataFrame to a file with the specified filename and file format.

        Args:
            filename (str): The name of the file to save the data to.
            raw (pd.DataFrame): The pandas DataFrame containing the data to be saved.

        Raises:
            ValueError: If the data format is unsupported or an error occurs during the saving process.

        Returns:
            None: If an error occurs during the saving process.
            str: Success message if the data is successfully saved to the specified file.
        """
        try:
            file_extension = os.path.splitext(filename)[-1]
            file_handlers = {
                ".csv": raw.to_csv,
                ".json": raw.to_json,
                ".xlsx": raw.to_excel,
                ".parquet": raw.to_parquet,
            }
            if file_extension in file_handlers:
                file_handlers[file_extension](filename)
                _f("success", f"saved - {filename}")
            else:
                _f("fatal", "unsupported data")
        except Exception as e:
            _f("fatal", e)
    def load(self, raw: str | pd.DataFrame = None, text_column: str = "clean", id_column: str = 'id'):
        """
        Load data into the df attribute of the Processor class.

        Args:
            raw (str | pd.DataFrame): The input data to be loaded. It can be either a file path (str) or a pandas DataFrame.
            text_column (str): The name of the column in the input data that contains the text data. Default is "clean".
            id_column (str): The name of the column in the input data that contains the unique identifier for each data entry. Default is "id".

        Raises:
            ValueError: If the file extension is not supported or an exception occurs during the loading process.

        Returns:
            None: If an error occurs during the loading process.
            str: Success message if the data is successfully loaded.
        """
        self.id_column = id_column
        self.text_column = text_column
        try:
            if isinstance(raw, str):
                raw_data_dir = raw
                file_extension = os.path.splitext(raw)[-1]
                file_handlers = {
                    ".csv": pd.read_csv,
                    ".json": pd.read_json,
                    ".xlsx": pd.read_excel,
                    ".parquet": pd.read_parquet,
                }
                if file_extension in file_handlers:
                    _f("wait", f"loading - {raw_data_dir}")
                    self.df = file_handlers[file_extension](raw_data_dir)
                    _f("success", f"loaded - {raw_data_dir}")
                    return _f('success',f"data successfully loaded from {raw_data_dir}")
                else:
                    raise ValueError("Unsupported file type")
            elif isinstance(raw, pd.DataFrame):
                self.df = raw
                _f("success", f"loaded - {raw}")
                return _f('success',f"data successfully loaded from DataFrame")
            else:
                raise ValueError("Data type not in [csv, json, xlsx, parquet, pd.DataFrame]")
        except Exception as e:
            raise ValueError(str(e))
    async def process(self, path: str = None, splitter: any = None, nlp=True):
        """
        Process and split the loaded data into smaller chunks.

        Args:
            path (str): The path to save the processed data.
            splitter (function, optional): A custom function to split the text into chunks. If not provided, the default splitter function in the `Processor` class will be used.
            nlp (bool, optional): A flag indicating whether to split the data into sentences using natural language processing (NLP) or by a fixed character length.

        Returns:
            None: If there is no data loaded or if an error occurs during the processing.
            The processed data saved to the specified file path.
        """
        self.df = self.df.dropna()
        if self.df is not None:
            try:
                _f("wait", f"get coffee or tea - {len(self.df)} processing...")
                sentence_splitter = self.default_splitter if splitter is None else splitter
                chunks = []
                knowledge_base = pd.DataFrame()
                tqdm.pandas()
                self.df["chunks"] = self.df[self.text_column].progress_apply(
                    lambda x: [
                        str(s) for s in sentence_splitter(self.utils.normalize_text(x), nlp=nlp)
                    ]
                )
                for i in tqdm(range(len(self.df))):
                    for c in self.df['chunks'].iloc[i]:
                        d = self.df[self.id_column].iloc[i]
                        chunks.append((d, c))
                knowledge_base['id'] = [c[0] for c in chunks]
                knowledge_base['chunks'] = [c[1] for c in chunks]

                self.df = knowledge_base
                _f('wait', f'saving to {path}')
                self.save(path, self.df)
                return
            except Exception as e:
                _f("fatal", e)
        else:
            return _f("fatal", "no data loaded!")
        
    async def create_charge(self, field=None):
        """
        Process and send data to a field.

        Args:
            field (optional): The field object to which the data will be sent.

        Returns:
            None

        Raises:
            ValueError: If no data is loaded.
            ValueError: If no field is loaded.

        Example Usage:
            # Initialize the Processor class object
            processor = Processor()

            # Load data into the processor
            processor.load(raw_data, text_column="clean", id_column="id")

            # Initialize the field object
            field = Field()

            # Create charges and send data to the field
            await processor.create_charge(field)
        """
        if self.df is not None and field is not None:
            self.field = field
            for i in tqdm(range(len(self.df))):
                d = self.df[self.id_column].iloc[i]
                c = self.df[self.text_column].iloc[i]
                payload = Payload(
                    document=d,
                    text=c
                )
                if self.field:
                    await self.field.pulse(payload)
                else:
                    raise ValueError('No field initialized')
        else:
            if not self.df:
                raise ValueError('No data loaded!')
            else:
                raise ValueError('No field loaded!')
    
    def default_splitter(self, data, window_size=768, overlap=76, nlp=True):
        """
        Splits the given input data into smaller chunks either by sentences or by a fixed character length.

        Args:
            data (str): The input data to be split.
            window_size (int, optional): The size of each chunk when splitting by a fixed character length. Defaults to 768.
            overlap (int, optional): The number of characters to overlap between each chunk when splitting by a fixed character length. Defaults to 76.
            nlp (bool, optional): A flag indicating whether to split the data into sentences using natural language processing (NLP) or by a fixed character length. Defaults to True.

        Returns:
            list: If `nlp` is True, returns a list of sentences extracted from the input data. If `nlp` is False, returns a list of chunks obtained by splitting the input data into fixed-sized chunks.
        """
        if nlp:
            self.utils.nlp.max_length = len(data) + 100
            sentences = [str(x) for x in self.utils.nlp(data).sents]
            return sentences
        else:
            # Perform chunked splitting by a fixed character length
            chunks = []
            start_char_idx = 0
            while start_char_idx < len(data):
                end_char_idx = start_char_idx + window_size
                chunk = data[start_char_idx:end_char_idx]
                chunks.append(chunk)
                start_char_idx += (window_size - overlap)
            return chunks