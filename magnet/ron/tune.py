import pandas as pd
import random, os, json
from tqdm import tqdm
from magnet.utils import _f, Utils
from magnet.ize import charge

class Prism:
    def __init__(self):
        self.df = None
        self.utils = Utils()

    def load(self, raw: str | pd.DataFrame = None):
        try:
            if isinstance(raw, str):
                raw_data_dir = os.path.join(raw)
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

    def generate_training_data(self
                               , out_dir: str = None
                               , split: int = 16
                               , k: int = 64
                               , index: str = None
                               , num_pos: int = 3
                               , num_neg: int = 7
                               , index_to_gpu: bool = False
                            ):
        data = self.df.sample(int(len(self.df)/split))
        f = open(os.path.join(out_dir,'finetune_kb_dataset.jsonl'), "w")
        pbar = tqdm(data.itertuples(), total=len(data))
        pole = charge.Pole()
        pole.load_embeddings(index, cuda = index_to_gpu)
        for row in pbar:
            kb_index = random.randint(0, len(data))
            q = data["sentences"].iloc[kb_index]
            embeddings = pole.search_document_embeddings(q, k=k, df=self.df)
            pos_results = embeddings[0:num_pos]
            neg_results = embeddings[::-1][0:num_neg]
            json.dump(
                {
                    "query": q,
                    "pos": [x for x in pos_results],
                    "neg": [x for x in neg_results],
                },
                f,
            )
            f.write("\n")
            pbar.set_description(
                _f(
                    "info",
                    f'processed  - "{row.sentences}"',
                    no_print=True,
                    luxe=True,
                ),
                refresh=True,
            )
        _f("success", f"written - {out_dir}")