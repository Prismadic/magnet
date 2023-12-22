<p align="center">
   <img height="300" width="300" src="./magnet.png">
   <br>
   <h3 align="center">magnet</h3>
   <p align="center">small, efficient embedding model toolkit</p>
   <p align="center"><i>~ fine-tune SOTA LLMs on knowledge bases rapidly ~</i></p>
</p>

</small>

## 🧬 Installation

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

## 🔮 features

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
 - 🧾 **Clean Logging**
   - <small>easy to trace</small>
   - <small>emojis are the future</small>


## 🧲 why

- build a distributed LLM research node with any hardware, from Rasbperry Pi to the expensive cloud
- Apple silicon first-class citizen with [mlx](https://github.com/ml-explore/mlx)
- embed & index to vector db with [milvus](https://milvus.io)
- distributed processing with [NATS](https://nats.io)
- upload to S3
- ideal cyberpunk vision of LLM power users in vectorspace

## goals

- [x] add [vllm](https://vllm.ai) implementation
- [x] add [huggingface](https://huggingface.co/docs/api-inference/detailed_parameters?code=python) implementation
- [x] add [milvus](https://milvus.io) implementation
- [x] finish `README.md`
- [x] add [NATS](https://nats.io) for distributed processing
- [x] add [mlx](https://github.com/ml-explore/mlx) support
