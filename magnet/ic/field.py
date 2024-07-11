import json, datetime, xxhash, platform, asyncio, io, base64

from dataclasses import asdict
from tabulate import tabulate

from magnet.base import Magnet
from magnet.utils.globals import _f
from magnet.utils.data_classes import *

from nats.errors import TimeoutError
from nats.js.api import StreamConfig, ConsumerConfig
from nats.js.errors import ServerError
from nats.js.api import ObjectMeta

x = xxhash
dt = datetime.datetime.now(datetime.timezone.utc)
utc_time = dt.replace(tzinfo=datetime.timezone.utc)
utc_timestamp = utc_time.timestamp()

class Charge:
    def __init__(self, magnet: Magnet):
        self.magnet = magnet

    async def list_streams(self):
        try:
            streams = await self.magnet.js.streams_info()
            remote_streams = [x.config.name for x in streams]
            remote_subjects = [x.config.subjects for x in streams]
            data = zip(remote_streams, remote_subjects)
            # Initialize an empty list to store formatted data
            formatted_data = []

            # Loop through each stream and its subjects
            for stream, subjects in data:
                # Check if the current stream name or any of the subjects match the config variables
                match_stream_name = stream == self.magnet.config.stream_name
                match_subject = self.magnet.config.category in subjects

                # If there's a match, format the stream name and subjects with ANSI green color and a magnet emoji
                if match_stream_name or match_subject:
                    formatted_stream = f"\033[92m{stream} \U0001F9F2\033[0m"  # Green and magnet emoji for stream name
                    formatted_subjects = [f"\033[92m{subject} \U0001F9F2\033[0m" if subject == self.magnet.config.category else subject for subject in subjects]
                else:
                    formatted_stream = stream
                    formatted_subjects = subjects

                # Add the formatted stream and subjects to the list
                formatted_data.append([formatted_stream, ', '.join(formatted_subjects)])

            # Creating a table with the formatted data
            table = tabulate(formatted_data, headers=['Stream Name', 'Subjects'], tablefmt="pretty")

            _f("info", f'\n{table}')
        except TimeoutError:
            return _f('fatal', f'could not connect to {self.magnet.config.host}')
        except Exception as e:
            return _f('fatal', e)

    async def on(self):
        try:
            streams = await self.magnet.js.streams_info()
            remote_streams = [x.config.name for x in streams]
            remote_subjects = [x.config.subjects for x in streams]
            if self.magnet.config.stream_name not in remote_streams:
                return _f('fatal', f'{self.magnet.config.stream_name} not found, initialize with `Magnet.align()` first')
            elif self.magnet.config.category not in sum(remote_subjects, []):
                if self.magnet.config.category not in sum([x.config.subjects for x in streams if x.config.name == self.magnet.config.stream_name], []):
                    try:
                        subjects = sum(
                            [x.config.subjects for x in streams if x.config.name == self.magnet.config.stream_name], [])
                        subjects.append(self.magnet.config.category)
                        await self.magnet.js.update_stream(StreamConfig(
                            name=self.magnet.config.stream_name
                            , subjects=subjects
                        ))
                        _f("success", f'created [{self.magnet.config.category}] on\n🛰️ stream: {self.magnet.config.stream_name}')
                    except ServerError as e:
                        _f('fatal', f"couldn't create {self.magnet.config.stream_name} on {self.magnet.config.host}, ensure your `category` is set")
        except TimeoutError:
            return _f('fatal', f'could not connect to {self.magnet.config.host}')
        _f("success", f'ready [{self.magnet.config.category}] on\n🛰️ stream: {self.magnet.config.stream_name}')

    async def off(self):
        """
        Disconnects from the NATS server and prints a warning message.
        """
        await self.magnet.nc.drain()
        await self.magnet.nc.close()
        _f('warn', f'disconnected from {self.magnet.config.host}')

    async def pulse(self, payload: Payload | FilePayload | GeneratedPayload | EmbeddingPayload | JobParams = None, create_job=False, v=False):
        """
        Publishes data to the NATS server using the specified category and payload.

        Args:
            payload (dict): The data to be published.
        """
        if type(payload) == FilePayload:
            _hash = x.xxh64(payload.data).hexdigest()
            payload_data_bytes = base64.b64decode(payload.data.encode('utf-8'))
            bucket_name = self.magnet.config.os_name
            object_name = f"{_hash}"
            meta = ObjectMeta(name=object_name
                              , headers={
                                  "ext": payload.document.split('.')[-1]
                              })
            bucket = await self.magnet.js.object_store(bucket_name)
            await bucket.put(object_name, payload_data_bytes, meta=meta)
            _f('success', f'uploaded to NATS object store in bucket {bucket_name} as {object_name}') if v else None

            if create_job:
                job = Job("process_document", _hash)
                await self.magnet.kv.put(key=job._id, value=json.dumps(asdict(job)).encode('utf-8'))
                _f('info', f'created job {job._id}')
        else:
            try:
                bytes_ = json.dumps(asdict(payload), separators=(
                    ', ', ':')).encode('utf-8')
            except Exception as e:
                return _f('fatal', f'invalid object, more info:\n{e} in [Payload, FilePayload, GeneratedPayload, EmbeddingPayload, JobParams]')
            try:
                _hash = x.xxh64(bytes_).hexdigest()
                msg = await self.magnet.js.publish(
                    self.magnet.config.category, bytes_, headers={
                        "Nats-Msg-Id": _hash
                    }
                )
                _f('success', f'pulsed to {self.magnet.config.category} on {self.magnet.config.stream_name}') if v else None
                _ts = datetime.datetime.now(datetime.timezone.utc)
                msg.ts = _ts
                return msg
            except Exception as e:
                return _f('fatal', f'could not pulse data to {self.magnet.config.host}\n{e}')

    async def excite(self, job: dict = {}):
        """
        Publishes data to the NATS server using the specified category and payload.

        Args:
            job (dict, optional): The data to be published. Defaults to {}.
        """
        try:
            bytes_ = json.dumps(job, separators=(', ', ':')).encode('utf-8')
        except Exception as e:
            _f('fatal', f'invalid JSON\n{e}')
        try:
            _hash = x.xxh64(bytes_).hexdigest()
            await self.magnet.js.publish(
                self.magnet.config.category, bytes_, headers={
                    "Nats-Msg-Id": _hash
                }
            )
        except Exception as e:
            _f('fatal', f'could not send data to {self.magnet.config.host}\n{e}')

    async def emp(self, name=None):
        """
        Deletes the specified stream if the name matches the current stream, or prints an error message if the name doesn't match or the stream doesn't exist.

        Args:
            name (str, optional): The name of the stream to delete. Defaults to None.
        """
        if name and name == self.magnet.config.stream_name:
            await self.magnet.js.delete_stream(name=self.magnet.config.stream_name)
            _f('warn', f'{self.magnet.config.stream_name} stream deleted')
        else:
            _f('fatal', "name doesn't match the stream or stream doesn't exist")

    async def reset(self, name=None):
        """
        Purges the specified category if the name matches the current category, or prints an error message if the name doesn't match or the category doesn't exist.

        Args:
            name (str, optional): The name of the category to purge. Defaults to None.
        """
        if name and name == self.magnet.config.category:
            await self.js.purge_stream(name=self.magnet.config.stream_name, subject=self.magnet.config.category)
            _f('warn', f'{self.magnet.config.category} category deleted')
        else:
            _f('fatal', "name doesn't match the stream category or category doesn't exist")

class Resonator:
    def __init__(self, magnet: Magnet):
        """
        Initializes the `Resonator` class with the NATS server address.

        Args:
            server (str): The address of the NATS server.
        """
        self.magnet = magnet

    async def on(self, job: bool = None, local: bool = False, bandwidth: int = 1000, obj=False):
        """
        Connects to the NATS server, subscribes to a specific category in a stream, and consumes messages from that category.

        Args:
            category (str, optional): The category to subscribe to. Defaults to 'no_category'.
            stream (str, optional): The stream to subscribe to. Defaults to 'documents'.
            session (str, optional): The session name for durable subscriptions. Defaults to 'magnet'.

        Returns:
            None

        Raises:
            TimeoutError: If there is a timeout error while connecting to the NATS server.
            Exception: If there is an error in consuming the message or processing the callback function.
        """
        self.node = f'{platform.node()}_{x.xxh64(platform.node(), seed=int(utc_timestamp)).hexdigest()}' if local else platform.node()
        self.durable = f'{self.node}_job' if job else self.node
        self.consumer_config = ConsumerConfig(
            ack_policy="explicit"
            , max_ack_pending=bandwidth
            , ack_wait=3600
        )
        _f('wait', f'connecting to {self.magnet.config.host.split("@")[1]}')
        try:
            if obj:
                self.sub = await self.magnet.os.watch(include_history=False)
                _f('info',
                    f'subscribed to object store: {self.magnet.config.os_name} as {self.node}')
            else:
                self.sub = await self.magnet.js.pull_subscribe(
                    durable=self.magnet.config.session
                    , subject=self.magnet.config.category
                    , config=self.consumer_config
                )
                _f('info',
                    f'joined worker queue: {self.magnet.config.session} as {self.node}')
        except Exception as e:
            return _f('fatal', e)
        
    async def download(self, obj: object = None):
        if obj and self.magnet.os:
            buffer = io.BytesIO()
            file = await self.magnet.os.get(obj.name, buffer)
            buffer.seek(0)
            chunk_size = 128 * 1024
            with open(f"{file.info.name}.{file.info.headers['ext']}", 'wb') as fh:
                while True:
                    chunk = buffer.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                _f('success', f'downloaded {file.info.name} from {self.magnet.config.os_name}')
        else:
            _f('fatal', 'no object store initialized')

    async def listen(self, cb=print, job_n: int = None, generic: bool = False, v=False):
        try:
            self.sub
        except AttributeError:
            return _f('fatal', 'no subscriber initialized')

        async def deliver_messages(msgs):
            payloads = [msg.data if generic else Payload(**json.loads(msg.data)) for msg in msgs]
            for payload, msg in zip(payloads, msgs):
                try:
                    await cb(payload, msg)
                except ValueError as e:
                    _f('success', f"job of {job_n} fulfilled\n{e}")
                except Exception as e:
                    _f('fatal', e)

        if job_n:
            try:
                if type(self.sub).__name__ == "ObjectWatcher":
                    _f("info", f'consuming objects from [{self.magnet.config.host.split("@")[1]}] from\n🛰️ bucket: {self.magnet.config.os_name}"')
                    msgs = await self.magnet.os.list()
                    for msg in msgs:
                        await self.download(msg)
                        await cb(self.magnet.os, msg)
                else:
                    _f("info", f'consuming {job_n} from [{self.magnet.config.category}] on\n🛰️ stream: {self.magnet.config.stream_name}\n🧲 session: "{self.magnet.config.session}"')
                    msgs = await self.sub.fetch(batch=job_n, timeout=60)
                    await deliver_messages(msgs)
            except ValueError as e:
                _f('warn', f'{self.magnet.config.session} reached the end of {self.magnet.config.category}, {self.magnet.config.name}')
            except Exception as e:
                _f('warn', f"no more data\n{e}")
        else:
            if type(self.sub).__name__ == "ObjectWatcher":
                _f("info", f'consuming objects from [{self.magnet.config.host.split("@")[1]}] from\n🛰️ bucket: {self.magnet.config.stream_name}"')
                e = await self.sub.updates()
                loop = asyncio.get_event_loop()
                loop.create_task(cb(self.magnet.os, e))
                await asyncio.sleep(1)
            else:
                _f("info", f'consuming delta from [{self.magnet.config.category}] on\n🛰️ stream: {self.magnet.config.stream_name}\n🧲 session: "{self.magnet.config.session}"')
                while True:
                    try:
                        msgs = await self.sub.fetch(batch=1, timeout=60)
                        if v:
                            _f('info', f"{msgs}")
                        payload = msgs[0].data if generic else Payload(**json.loads(msgs[0].data))
                        if v:
                            _f('info', f"{payload}")
                        await cb(payload, msgs[0])
                    except Exception as e:
                        if "nats: timeout" in str(e):
                            _f('warn', 'encountered a timeout, retrying in 1s')
                        else:
                            _f('fatal', str(e))
                        _f("warn", f'retrying connection to {self.magnet.config.host.split("@")[1]}\n{e}')
                        _f("info", "this can also be a problem with your callback")
                    await asyncio.sleep(1)


    async def worker(self, cb=print):
        """
        Consume messages from a specific category in a stream and process them as jobs.

        Args:
            cb (function, optional): The callback function to process the received messages. Defaults to `print`.

        Returns:
            None

        Raises:
            Exception: If there is an error in consuming the message or processing the callback function.
        """
        _f("info",
           f'processing jobs from [{self.magnet.config.kv_name}] on\n🛰️ object store: {self.magnet.config.os_name}')
        try:
            keys = await self.magnet.kv.keys()
            for key in keys:
                _job = await self.magnet.kv.get(key)
                job = json.loads(_job.value.decode('utf-8'))
                if not job['_isClaimed']: 
                    await cb(_job, job)
        except Exception as e:
            _f('fatal', f'invalid JSON\n{e}')

    async def conduct(self, cb=print):
        pass

    async def info(self):
        """
        Retrieves information about a consumer in a JetStream stream.

        :param session: A string representing the session name of the consumer. If not provided, information about all consumers in the stream will be retrieved.
        :return: None
        """
        jsm = await self.magnet.js.consumer_info(stream=self.magnet.config.stream_name, consumer=self.magnet.session)
        _f('info', json.dumps(jsm.config.__dict__, indent=2))

    async def off(self):
        """
        Unsubscribes from the category and stream and disconnects from the NATS server.

        :return: None
        """
        await self.sub.unsubscribe()
        _f('warn', f'unsubscribed from {self.magnet.config.stream_name}')
        await self.nc.drain()
        _f('warn', f'safe to disconnect from {self.magnet.config.host.split("@")[1]}')

