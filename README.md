<p align="center">
   <img height="250" width="250" src="./magnet.png">
   <br>
   <h3 align="center">magnet</h3>
   <p align="center">small, efficient embedding model toolkit</p>
   <p align="center"><i>~ fine-tune SOTA LLMs on knowledge bases rapidly ~</i></p>
</p>

</small>

## 💾 Installation

``` bash
pip install llm-magnet
```
or 
``` bash
python3 setup.py install
```


## 🎉 usage

[check out this notebook, it's really useful](./example.ipynb) `(./example.ipynb)`

<small>a snippet to get you started</small>

``` python
from magnet.filings import Processor
source_data_file = "./raw/kb_export_clean.parquet" # your text documents data
filings = Processor()
filings.load(source_data_file)
await filings.process('./data/filings.parquet','clean','file', nlp=False)
```

## 👏 features

 - so long as your initial data has columns for article text and some unique identifier per source document, `magnet` can do the rest
 - embed & index to vector db with [milvus](https://milvus.io)
 - sequential distributed processing with [NATS](https://nats.io)
 - upload to S3
 - ideal cyberpunk vision of LLM power users in vectorspace

## goals

- [x] add Milvus implementation
- [ ] add [mlx](https://github.com/ml-explore/mlx) support
- [x] finish `README.md`
- [x] add [NATS](https://nats.io) for distributed processing
- [ ] `deepspeed` integration for model parallelism on multiple GPU

## bad code

- [x] `spacy.nlp` is used poorly throughout, need to make it possible to make sentence splitter hooks of our own and `spacy` can be a default fallback