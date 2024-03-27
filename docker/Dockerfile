FROM python:3.8

WORKDIR /magnet_node
RUN pip install --upgrade pip
RUN git clone https://github.com/prismadic/magnet.git magnet

WORKDIR /magnet_node/magnet

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "/magnet_node/magnet/docker/scripts/main.py"]