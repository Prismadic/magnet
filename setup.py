from setuptools import setup

setup(
    name='llm_magnet',
    version='0.0.4',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        "torch"
        , "spacy"
        , "numpy"
        , "nats-py"
    ],
    url = 'https://github.com/Prismadic/magnet'
)