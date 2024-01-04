<p align="center">
   <img height="300" width="300" src="./magnet.png">
   <br>

   <h1 align="center">magnet</h1>

   <h3 align="center"><a href="https://prismadic.github.io/magnet/">📖 docs</a> | 💻 <a href="https://github.com/Prismadic/magnet/tree/main/examples">examples</a> | 📓 <a href="https://prismadic.substack.com">substack</a></h3> 

   <p align="center">the small distributed language model toolkit</p>
   <p align="center"><i>⚡️ fine-tune state-of-the-art LLMs anywhere, rapidly ⚡️</i></p>
   <div align="center">
</p>

![GitHub release (with filter)](https://img.shields.io/github/v/release/prismadic/magnet)
![PyPI - Version](https://img.shields.io/pypi/v/llm_magnet)
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/prismadic/magnet/python-publish.yml)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/prismadic/magnet)
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/prismadic/magnet/main)
![GitHub issues](https://img.shields.io/github/issues/prismadic/magnet)
![GitHub Repo stars](https://img.shields.io/github/stars/prismadic/magnet)
![GitHub watchers](https://img.shields.io/github/watchers/prismadic/magnet)
![PyPI - Downloads](https://img.shields.io/pypi/dm/llm_magnet)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/llm_magnet)
![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/prismadic?style=social&link=https%3A%2F%2Fx.com%2Fprismadic)

   </div>

</p>

<img src='./divider.png' style="width:100%;height:5px;">

## 🧬 Installation

``` bash
pip install llm-magnet
```

or

``` bash
python3 setup.py install
```

<img src='./divider.png' style="width:100%;height:5px;">

## 🎉 usage

[check out the example notebooks](./examples/)

<small>a snippet to get you started</small>

``` python
from magnet.ize.filings import Processor
source_data_file = "./raw/filings.csv"
export_data_file = "./data/filings_mistral_nlp.parquet"
filings = Processor()
filings.load(
    source_data_file
    , 'clean'
    , 'file'
)
await filings.process(
    export_data_file
    , nlp=True
)
```

<img src='./divider.png' style="width:100%;height:5px;">

## 🔮 features

<center>
<img src="./clustered_bidirectional.png" style="width:50%;transform: rotate(90deg);margin-top:200px;" align="right">
</center>

- ⚡️ **It's Fast**
  - <small>fast on consumer hardware</small>
  - <small>_very_ fast on Apple Silicon</small>
  - <small>**extremely** fast on ROCm/CUDA</small>
- 🫵 **Automatic or your way**
  - <small>rely on established transformer patterns to let `magnet` do the work</small>
  - <small>keep your existing data processing functions, bring them to `magnet`!</small>
- 🛰️ **100% Distributed**
  - <small>processing, embedding, storage, retrieval, querying, or inference from anywhere</small>
  - <small>as much or as little compute as you need</small>
- 🧮 **Choose Inference Method**
  - <small>HuggingFace</small>
  - <small>vLLM node</small>
  - <small>GPU</small>
  - <small>mlx</small>
- 🌎 **Huge Volumes**
  - <small>handle gigantic amounts of data inexpensively</small>
  - <small>fault-tolerant by design</small>
  - <small>decentralized workloads</small>
- 🔐 **Secure**
  - <small>JWT</small>
  - <small>Basic</small>
- 🪵 **World-Class Comprehension**
  - <small>`magnet` optionally logs its own code as it's executed (yes, really)</small>
  - <small>build a self-aware system and allow it to learn from itself</small>
  - <small>emojis are the future</small>

<img src='./divider.png' style="width:100%;height:5px;">

## 🧲 why

- build a distributed LLM research node with any hardware, from Rasbperry Pi to the expensive cloud
- Apple silicon first-class citizen with [mlx](https://github.com/ml-explore/mlx)
- embed & index to vector db with [milvus](https://milvus.io)
- distributed processing with [NATS](https://nats.io)
- upload to S3
- ideal cyberpunk vision of LLM power users in vectorspace

<img src='./divider.png' style="width:100%;height:5px;">
