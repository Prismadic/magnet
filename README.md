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

[check out the example notebooks](./examples/)

<small>a snippet to get you started</small>

``` python
from magnet.filings import Processor
# your text documents data
source_data_file = "./raw/export.parquet"
filings = Processor()
filings.load(source_data_file)
# output, text column, id column, and we disable sentence splitting for fastest processing
await filings.process('./data/filings.parquet','clean','file', nlp=False)
```

## 👏 features

 - so long as your initial data has columns for article text and some unique identifier per source document, `magnet` can do the rest
 - embed & index to vector db with [milvus](https://milvus.io)
 - distributed processing with [NATS](https://nats.io)
 - upload to S3
 - ideal cyberpunk vision of LLM power users in vectorspace

## goals

- [x] add [milvus](https://milvus.io) implementation
- [ ] add [mlx](https://github.com/ml-explore/mlx) support
- [x] finish `README.md`
- [x] add [NATS](https://nats.io) for distributed processing
