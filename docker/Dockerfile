FROM python:3.8

WORKDIR /magnet_node

RUN git clone https://github.com/prismadic/magnet.git magnet

WORKDIR /magnet_node/magnet

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./docker/scripts/main.py"]