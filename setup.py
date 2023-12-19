from setuptools import setup

setup(
    name='llm_magnet',
    version='0.0.5',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        "torch"
        , "spacy"
        , "numpy"
        , "nats-py"
        , "farm-haystack"
        , "milvus-haystack"
        , "sentence_transformers"
    ],
    url = 'https://github.com/Prismadic/magnet'
)