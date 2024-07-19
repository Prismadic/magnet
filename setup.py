from setuptools import setup

setup(
    name='llm_magnet',
    version='0.3.15',
    description="the small distributed language model toolkit. fine-tune state-of-the-art LLMs anywhere, rapidly.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'torch',
        'sentence-transformers',
        'pandas',
        'tqdm',
        'spacy',
        'pymilvus',
        'boto3',
        'nats-py',
        'rich',
        'xxhash',
        'nkeys',
        'tabulate',
        'milvus',
        'docker',
    ],
    url = 'https://github.com/Prismadic/magnet'
)