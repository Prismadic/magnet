from setuptools import setup

setup(
    name='llm_magnet',
    version='0.0.6',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        "torch"
        , "spacy"
        , "numpy"
        , "nats-py"
        , "sentence_transformers"
        , "vllm"
    ],
    url = 'https://github.com/Prismadic/magnet'
)