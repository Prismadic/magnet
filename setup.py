from setuptools import setup

setup(
    name='llm_magnet',
    version='0.1.2',
    description="the small distributed language model toolkit. fine-tune state-of-the-art LLMs anywhere, rapidly.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        "torch"
        , "spacy"
        , "numpy"
        , "nats-py"
        , "sentence_transformers"
        , "pymilvus"
        , "rich"
        , "xxhash"
    ],
    url = 'https://github.com/Prismadic/magnet'
)