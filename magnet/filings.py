import pandas as pd
import os
from .utils import _f, Utils
from tqdm import tqdm

class Processor:
    def __init__(self, field=None):
        self.df = None
        self.utils = Utils()
        self.field = field
        
    def save(self, filename: str = None, raw: pd.DataFrame = None):
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
    def load(self, raw: str | pd.DataFrame = None):
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
                    self.df = file_handlers[file_extension](raw_data_dir)
                    _f("success", f"loaded - {raw_data_dir}")
                else:
                    _f("fatal", "unsupported file type")
            elif isinstance(raw, pd.DataFrame):
                self.df = raw
                _f("success", f"loaded - {raw}")
            else:
                _f("fatal", "data type not in [csv, json, xlsx, parquet, pd.DataFrame]")
        except Exception as e:
            _f("fatal", e)
    async def process(self, path: str = None, text_column: str = "clean", id_column: str = 'id', splitter: any = None, nlp=True):
        self.df = self.df.dropna()
        if self.field:
            await self.field.on()
        if self.df is not None:
            try:
                _f("wait", f"get coffee or tea - {len(self.df)} processing...")
                sentence_splitter = self.bge_splitter if splitter is None else splitter
                chunks = []
                knowledge_base = pd.DataFrame()
                tqdm.pandas()
                self.df["chunks"] = self.df[text_column].progress_apply(
                    lambda x: [
                        str(s) for s in sentence_splitter(self.utils.normalize_text(x), nlp=nlp)
                    ]
                )
                for i in tqdm(range(len(self.df))):
                    for s in self.df['chunks'].iloc[i]:
                        a = self.df[id_column].iloc[i]
                        chunks.append((a, s))
                        if self.field:
                            await self.field.pulse(s, a)
                knowledge_base['chunks'] = [x[1] for x in chunks]
                knowledge_base['id'] = [x[0] for x in chunks]
                self.df = knowledge_base
                _f('wait', f'saving to {path}')
                self.save(path, self.df)
                return
            except Exception as e:
                _f("fatal", e)
        else:
            return _f("fatal", "no data loaded!")
    def bge_splitter(self, data, window_size=250, overlap=25, nlp=True):
        if nlp:
            self.utils.nlp.max_length = len(data) + 100
            sentences = [str(x) for x in self.utils.nlp(data).sents]

            new_sentences = []
            for sentence in sentences:
                # Split the sentence into words
                words = sentence.split()

                # Initialize indices
                start_idx = 0
                end_idx = 0

                while end_idx < len(words):
                    # Calculate the end index based on words, not characters
                    while end_idx < len(words) and len(' '.join(words[start_idx:end_idx + 1])) <= window_size:
                        end_idx += 1

                    # Create a chunk with the selected words
                    chunk = ' '.join(words[start_idx:end_idx])

                    # Check if the chunk is not empty and does not contain individual words longer than window_size
                    if chunk and all(len(word) <= window_size for word in chunk.split()):
                        new_sentences.append(chunk)

                    # Update the start index for the next chunk
                    start_idx = end_idx

            return new_sentences
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
    def mistral_splitter(self, data, window_size=768, overlap=76, nlp=True):
        if nlp:
            self.utils.nlp.max_length = len(data) + 100
            sentences = [str(x) for x in self.utils.nlp(data).sents]

            new_sentences = []
            for sentence in sentences:
                tokens = self.utils.nlp.tokenizer(sentence)
                num_tokens = len(tokens)
                
                start_token_idx = 0
                end_token_idx = 0

                while end_token_idx < num_tokens:
                    # Calculate the end index based on tokens
                    while end_token_idx < num_tokens and sum(len(tokens[i].text) for i in range(start_token_idx, end_token_idx + 1)) <= window_size:
                        end_token_idx += 1

                    # Create a chunk with the selected tokens
                    chunk = ' '.join(tokens[start_token_idx:end_token_idx])

                    # Check if the chunk is not empty and does not contain individual words longer than window_size
                    if chunk and all(len(token.text) <= window_size for token in tokens[start_token_idx:end_token_idx]):
                        new_sentences.append(chunk)

                    # Update the start index for the next chunk
                    start_token_idx = end_token_idx

            return new_sentences
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