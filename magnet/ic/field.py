import nats, json, datetime, xxhash, platform

from dataclasses import asdict
from magnet.utils.globals import _f
from magnet.utils.data_classes import *

from nats.errors import TimeoutError
from nats.js.api import StreamConfig, ConsumerConfig

x = xxhash
dt = datetime.datetime.now(datetime.timezone.utc)
utc_time = dt.replace(tzinfo=datetime.timezone.utc)
utc_timestamp = utc_time.timestamp()

class Charge:
    def __init__(self, prism: PrismConfig):
        self.prism = prism

    async def on(self):
        try:
            streams = await self.prism.js.streams_info()
            remote_streams = [x.config.name for x in streams]
            remote_subjects = [x.config.subjects for x in streams]
            if self.prism.stream_name not in remote_streams:
                return _f('fatal', 'stream not found, initialize with `Prism.align()` first')
            elif self.prism.stream_category not in sum(remote_subjects, []):
                if self.prism.stream_category not in sum([x.config.subjects for x in streams if x.config.name == self.prism.stream_name], []):
                    try:
                        subjects = sum(
                            [x.config.subjects for x in streams if x.config.name == self.prism.stream_name], [])
                        subjects.append(self.prism.stream_category)
                        await self.prism.js.update_stream(StreamConfig(
                            name=self.prism.stream_name
                            , subjects=subjects
                        ))
                        _f("success", f'created [{self.prism.stream_category}] on\n🛰️ stream: {self.prism.stream_name}')
                    except Exception as e:
                        _f('fatal', f"couldn't create {self.prism.stream_name} on {self.prism.host}\n{e}")
        except TimeoutError:
            return _f('fatal', f'could not connect to {self.prism.host}')
        _f("success", f'connected to [{self.prism.stream_category}] on\n🛰️ stream: {self.prism.stream_name}')

    async def off(self):
        """
        Disconnects from the NATS server and prints a warning message.
        """
        await self.prism.nc.drain()
        await self.prism.nc.close()
        _f('warn', f'disconnected from {self.prism.host}')

    async def pulse(self, payload):
        """
        Publishes data to the NATS server using the specified category and payload.

        Args:
            payload (dict): The data to be published.
        """
        try:
            bytes_ = json.dumps(asdict(payload), separators=(
                ', ', ':')).encode('utf-8')
        except Exception as e:
            _f('fatal', f'invalid JSON\n{e}')
        try:
            _hash = x.xxh64(bytes_).hexdigest()
            await self.prism.js.publish(
                self.prism.stream_category, bytes_, headers={
                    "Nats-Msg-Id": _hash
                }
            )
        except Exception as e:
            _f('fatal', f'could not send data to {self.prism.host}\n{e}')

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
            await self.prism.js.publish(
                self.prism.stream_category, bytes_, headers={
                    "Nats-Msg-Id": _hash
                }
            )
        except Exception as e:
            _f('fatal', f'could not send data to {self.prism.host}\n{e}')

    async def emp(self, name=None):
        """
        Deletes the specified stream if the name matches the current stream, or prints an error message if the name doesn't match or the stream doesn't exist.

        Args:
            name (str, optional): The name of the stream to delete. Defaults to None.
        """
        if name and name == self.prism.stream_name:
            await self.prism.js.delete_stream(name=self.prism.stream_name)
            _f('warn', f'{self.prism.stream_name} stream deleted')
        else:
            _f('fatal', "name doesn't match the stream or stream doesn't exist")

    async def reset(self, name=None):
        """
        Purges the specified category if the name matches the current category, or prints an error message if the name doesn't match or the category doesn't exist.

        Args:
            name (str, optional): The name of the category to purge. Defaults to None.
        """
        if name and name == self.prism.stream_category:
            await self.js.purge_stream(name=self.prism.stream_name, subject=self.prism.stream_category)
            _f('warn', f'{self.prism.stream_category} category deleted')
        else:
            _f('fatal', "name doesn't match the stream category or category doesn't exist")

class Resonator:
    def __init__(self, prism: PrismConfig | dict = None):
        """
        Initializes the `Resonator` class with the NATS server address.

        Args:
            server (str): The address of the NATS server.
        """
        self.prism = prism

    async def on(self, job: bool = None, local: bool = False, bandwidth: int = 1000):
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
        _f('wait', f'connecting to {self.prism.host}')
        try:
            self.sub = await self.js.pull_subscribe(
                durable=self.prism.session
                , subject=self.prism.stream_category
                , stream=self.prism.stream_name
                , config=self.consumer_config
            )
            _f('info',
                f'joined worker queue: {self.prism.session} as {self.node}')
        except Exception as e:
            return _f('fatal', e)

    async def listen(self, cb=print, job_n: int = None, generic: bool = False, verbose=False):

        try: self.prism.js.sub
        except: return _f('fatal', 'no subscriber initialized')
        if job_n:
            _f("info",
               f'consuming {job_n} from [{self.prism.stream_category}] on\n🛰️ stream: {self.prism.stream_name}\n🧲 session: "{self.prism.session}"')
            try:
                msgs = await self.prism.js.sub.fetch(batch=job_n, timeout=60)
                payloads = [msg.data if generic else Payload(
                    **json.loads(msg.data)) for msg in msgs]
                try:
                    for payload, msg in zip(payloads, msgs):
                        await cb(payload, msg)
                except ValueError as e:
                    _f('success', f"job of {job_n} fulfilled\n{e}")
                except Exception as e:
                    _f('fatal', e)
            except ValueError as e:
                _f('warn',
                   f'{self.session} reached the end of {self.prism.stream_category}, {self.prism.stream_name}')
            except Exception as e:
                _f('fatal', e)
        else:
            _f("info",
               f'consuming delta from [{self.prism.stream_category}] on\n🛰️ stream: {self.prism.stream_name}\n🧲 session: "{self.prism.session}"')
            while True:
                try:
                    msgs = await self.sub.fetch(batch=1, timeout=60)
                    _f('info', f"{msgs}") if verbose else None
                    payload = msgs[0].data if generic else Payload(
                        **json.loads(msgs[0].data))
                    _f('info', f"{payload}") if verbose else None
                    try:
                        await cb(payload, msgs[0])
                    except Exception as e:
                        _f("warn", f'retrying connection to {self.prism.host}\n{e}')
                        _f("info", "this can also be a problem with your callback")
                except Exception as e:
                    _f('fatal', f'invalid JSON\n{e}')
                    break

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
           f'processing jobs from [{self.prism.stream_category}] on\n🛰️ stream: {self.prism.stream_name}\n🧲 session: "{self.prism.session}"')
        try:
            msg = await self.sub.next_msg(timeout=60)
            payload = JobParams(**json.loads(msg.data))
            try:
                await cb(payload, msg)
            except Exception as e:
                _f("warn", f'something wrong in your callback function!\n{e}')
        except Exception as e:
            _f('fatal', 'invalid JSON')

    async def conduct(self, cb=print):
        pass

    async def info(self):
        """
        Retrieves information about a consumer in a JetStream stream.

        :param session: A string representing the session name of the consumer. If not provided, information about all consumers in the stream will be retrieved.
        :return: None
        """
        jsm = await self.prism.js.consumer_info(stream=self.prism.stream_name, consumer=self.prism.session)
        _f('info', json.dumps(jsm.config.__dict__, indent=2))

    async def off(self):
        """
        Unsubscribes from the category and stream and disconnects from the NATS server.

        :return: None
        """
        await self.prism.js.sub.unsubscribe()
        _f('warn', f'unsubscribed from {self.prism.stream_name}')
        await self.nc.drain()
        _f('warn', f'safe to disconnect from {self.prism.host}')

class Prism:
    def __init__(self, config: PrismConfig | dict = None):
        try:
            if isinstance(config, dict):
                config = PrismConfig(**config)
            elif not isinstance(config, PrismConfig):
                _f("fatal", "config must be a PrismConfig instance or a dictionary")
                raise ValueError
        except Exception as e:
            raise e
        self.config = config
        self.nc = None
        self.js = None
        self.kv = None
        self.os = None

    async def align(self):
        """Connect to NATS and setup configurations."""
        try:
            self.nc = await nats.connect(f'{"nats://" if not self.config.credentials else "tls://"}{self.config.host}:4222',user_credentials=self.config.credentials)
            self.js = self.nc.jetstream(
                domain=self.config.domain 
            ) if self.config.domain is not None else self.nc.jetstream()
            if self.config.stream_name:
                self.config.stream_name = await self._setup_stream()
            if self.config.kv_name:
                self.kv = await self._setup_kv()
            if self.config.os_name:
                self.os = await self._setup_object_store()
            _f("success", f"🧲 connected to \n💎 {self.config} ")
            return [self.js, self.kv, self.os]
        except Exception as e:
            _f("fatal", f"could not align {self.config.host}\n{e}")
            return None

    async def _setup_stream(self):
        """Setup stream."""
        try:
            await self.js.stream_info(self.config.stream_name)
        except Exception as e:
            _f("warn", f"Stream {self.config.stream_name} not found, creating")
            await self.js.add_stream(name=self.config.stream_name, subjects=["magnet.*"])
            _f("success", f"created `{self.config.stream_name}` with default category `magnet.*`")
        return await self.js.stream_info(self.config.stream_name)

    async def _setup_kv(self):
        """Setup key-value store."""
        try:
            await self.js.key_value(self.config.kv_name)
        except Exception as e:
            _f("warn", f"KV bucket {self.config.kv_name} not found, creating")
            await self.js.create_key_value(bucket=self.config.kv_name)
            _f("success", f"created `{self.config.kv_name}`")
        return await self.js.key_value(self.config.kv_name)

    async def _setup_object_store(self):
        """Setup object store."""
        try:
            await self.js.object_store(self.config.os_name)
        except Exception as e:
            _f("warn", f"Object Store {self.config.kv_name} not found, creating")
            await self.js.create_object_store(bucket=self.config.os_name)
            _f("success", f"created `{self.config.os_name}`")
        return await self.js.object_store(self.config.os_name)

    async def off(self):
        """Disconnects from the NATS server and prints a warning message."""
        await self.nc.drain()
        _f('warn', f'disconnected from {self.config.host}')