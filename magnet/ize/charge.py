import faiss
import pandas as pd
from magnet.utils import Utils, _f
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

class Pole:
    def __init__(self, model: str = 'BAAI/bge-large-en-v1.5'):
        self.model = model
        self.sentences_index = None
        self.utils = Utils()
        
    def index_document_embeddings(self, df: pd.DataFrame = None, cuda: bool = False):
        try:
            model = SentenceTransformer(self.model)
            d = model[1].word_embedding_dimension
            M = 32
            dedupe = df.drop_duplicates(subset=['sentences']).dropna()
            sentences = [s for s in dedupe['sentences'].tolist()]
            all_embeddings = []
            cuda = self.utils.check_cuda() if cuda else False
            if cuda:
                sentences_index = faiss.IndexFlatIP(d)
                co, co.shard, co.useFloat16 = faiss.GpuMultipleClonerOptions(), True, True
                sentences_index = faiss.index_cpu_to_all_gpus(sentences_index, co=co)
            else:
                sentences_index = faiss.IndexHNSWFlat(d, M)
            if sentences_index.is_trained:
                sentences_index.hnsw.efConstruction = 40  
                sentences_index.hnsw.efSearch = 40
                pbar = tqdm(range(len(sentences)))
                for i in pbar:
                    embedding = model.encode(sentences[i], normalize_embeddings=True)
                    all_embeddings.append(embedding)
                    pbar.set_description(
                        _f(
                            "success",
                            f"embedded sentence {i}",
                            no_print=True,
                        ),
                        refresh=True,
                    )
                _f('wait', f'indexing {len(all_embeddings)} objects')
                sentences_index.add(np.asarray(all_embeddings, dtype=np.float32))
                self.sentences_index =  faiss.index_gpu_to_cpu(sentences_index) if cuda else sentences_index
                _f('success', 'index created')
            else:
                return _f('fatal', 'index of this type must be trained')
        except Exception as e:
            _f('fatal', e)
    
    def search_document_embeddings(self, q: str = None, k: int = 64, df: pd.DataFrame = None):
        try:
            model = SentenceTransformer(self.model)
            xq = model.encode([q])
            D, I  = self.sentences_index.search(xq, k)
            results = []
            for i, val in enumerate(I[0].tolist()):
                results.append(df['sentences'].iloc[val])
            return results
        except Exception as e:
            _f('fatal', e)
    
    def save_embeddings(self, index_path: str = None):
        if self.sentences_index:
            faiss.write_index(self.sentences_index, index_path)
            _f('success', f'embeddings saved to {index_path}')
        else:
            _f('fatal', 'no index in memory')

    def load_embeddings(self, index_path: str = None, cuda: bool = False):
        try:
            f = open(index_path, 'rb')
            reader = faiss.PyCallbackIOReader(f.read)
            if cuda:
                self.utils.check_cuda()
                co, co.shard, co.useFloat16 = faiss.GpuMultipleClonerOptions(), True, True
                index = faiss.index_cpu_to_all_gpus(index, co=co)
            else:
                index = faiss.read_index(reader)
            self.sentences_index = index
            _f('success', f'index loaded - {index_path}')
        except Exception as e:
            _f('fatal',e)

        