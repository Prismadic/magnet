FROM python:3.11

WORKDIR /magnet_node

RUN git clone https://github.com/prismadic/magnet.git magnet

WORKDIR /magnet_node/magnet

RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

CMD ["python", "./docker/scripts/main.py"]