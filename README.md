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
```

<small>*(yes, this is all it takes to initialize a project!)*</small>

## 😥 compute requirements

_minimum_ requirements for ~6000 documents from a knowledge base:

 1. RAM
    - 32GB RAM
 3. GPU
    - can choose to store your embeddings in VRAM
    - 4x 16GB VRAM (*for finetuning with research efficiency*)
    - otherwise helpful with embedding your data & scoring/ranking (speeds below)

#### ⏱️ "Ready, Set, Go!"

Generally speaking, the size of your documents and the quality of them will impact these times.
The larger datasets listed are curated with a lot more attention to quality for example. So in addition to being larger overall, the documents in the dataset are also larger.

🚧

## 👏 features

 - Apple silicon first class citizen
 - so long as your initial data has columns for article text and ids, `magnet` can do the rest
 - finetune highly performant expert models from 0-1 in very little time
 - upload to S3
 - ideal cyberpunk vision of LLM power users in vectorspace

## goals

- [x] finish `README.md`
- [ ] `deepspeed` integration for model parallelism on multiple GPU

## bad code

- [x] `spacy.nlp` is used poorly throughout, need to make it possible to make sentence splitter hooks of our own and `spacy` can be a default fallback