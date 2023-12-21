import pandas as pd
import os
from .utils import _f, Utils
from tqdm import tqdm
from magnet.ic.utils.data_classes import *

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
                sentence_splitter = self.default_splitter if splitter is None else splitter
                chunks = []
                knowledge_base = pd.DataFrame()
                tqdm.pandas()
                self.df["chunks"] = self.df[text_column].progress_apply(
                    lambda x: [
                        str(s) for s in sentence_splitter(self.utils.normalize_text(x), nlp=nlp)
                    ]
                )
                for i in tqdm(range(len(self.df))):
                    for c in self.df['chunks'].iloc[i]:
                        d = self.df[id_column].iloc[i]
                        chunks.append((d, c))
                        if self.field:
                            payload = Payload(
                                document = d
                                , text = c
                            )
                            await self.field.pulse(payload)
                knowledge_base['chunks'] = [c[1] for c in chunks]
                knowledge_base['id'] = [c[0] for c in chunks]
                self.df = knowledge_base
                _f('wait', f'saving to {path}')
                self.save(path, self.df)
                return
            except Exception as e:
                _f("fatal", e)
        else:
            return _f("fatal", "no data loaded!")
    def default_splitter(self, data, window_size=768, overlap=76, nlp=True):
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