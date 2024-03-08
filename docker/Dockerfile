FROM python:3.11
RUN python3.11 -m ensurepip 
RUN python3.11 -m pip install torch sentence-transformers pandas tqdm spacy pymilvus boto3 nats-py rich xxhash nkeys -U

WORKDIR /usr/src/app

# Install the required package
RUN pip install --no-cache-dir llm-magnet -U

COPY ./scripts/ ./

ENTRYPOINT ["sh", "-c"]
CMD ["python ./main.py"]