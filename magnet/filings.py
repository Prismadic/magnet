import pandas as pd
import os
from .utils import _f, Utils
from tqdm import tqdm
class Processor:
    def __init__(self):
        self.df = None
        self.utils = Utils()

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

    def export_as_sentences(self, path: str = None, text_column: str = "clean", id_column: str = 'id', splitter: any = None):
        self.df = self.df.dropna()
        if self.df is not None:
            try:
                _f("wait", f"get coffee or tea - {len(self.df)} processing...")
                sentence_splitter = self.bge_sentence_splitter if splitter is None else splitter
                all_sentences = []
                knowledge_base = pd.DataFrame()
                tqdm.pandas()
                self.df["sentences"] = self.df[text_column].progress_apply(
                    lambda x: [
                        str(s) for s in sentence_splitter(self.utils.normalize_text(x))
                    ]
                )
                for i in range(len(self.df)):
                    for s in self.df['sentences'].iloc[i]:
                        a = self.df[id_column].iloc[i]
                        all_sentences.append((a, s))
                knowledge_base['sentences'] = [x[1] for x in all_sentences]
                knowledge_base['id'] = [x[0] for x in all_sentences]
                self.df = knowledge_base
                self.save(path, self.df)
                return
            except Exception as e:
                _f("fatal", e)
        else:
            return _f("fatal", "no data loaded!")
        
    def bge_sentence_splitter(self, data, window_size=768, overlap=76):
        self.utils.nlp.max_length = len(data) + 100
        sentences = [str(x) for x in self.utils.nlp(data).sents]

        new_sentences = []
        for sentence in sentences:
            start_idx = 0
            end_idx = window_size
            
            while start_idx < len(sentence):
                chunk = sentence[start_idx:end_idx]
                new_sentences.append(chunk)
                
                # Slide the window, ensuring we don't exceed the sentence boundaries
                start_idx += (window_size - overlap)
                end_idx = min(start_idx + window_size, len(sentence))

        return new_sentences
    
    def mistral_sentence_splitter(self, data, window_size=768, overlap=76):
        self.utils.nlp.max_length = len(data) + 100
        sentences = [str(x) for x in self.utils.nlp(data).sents]

        new_sentences = []
        for sentence in sentences:
            tokens = self.utils.nlp.tokenizer(sentence)
            num_tokens = len(tokens)
            
            start_token_idx = 0
            end_token_idx = min(window_size, num_tokens)
            
            while start_token_idx < num_tokens:
                start_char_idx = tokens[start_token_idx].idx
                end_char_idx = tokens[end_token_idx - 1].idx + len(tokens[end_token_idx - 1])
                chunk = sentence[start_char_idx:end_char_idx]
                new_sentences.append(chunk)
                
                # Slide the window, ensuring we don't exceed the token or character boundaries
                start_token_idx += (window_size - overlap)
                end_token_idx = min(start_token_idx + window_size, num_tokens)
                
        return new_sentences