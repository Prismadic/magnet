import nats, json
from magnet.utils.globals import _f
from dataclasses import asdict
from nats.errors import TimeoutError
from magnet.utils.data_classes import *
from nats.js.api import StreamConfig

class Charge:
    """
    The `Charge` class is responsible for connecting to a NATS server, managing streams and categories, and publishing data to the server.

    Args:
        server (str): The NATS server URL.

    Attributes:
        server (str): The NATS server URL.
        category (str): The current category.
        stream (str): The current stream.
        nc: The NATS connection object.
        js: The JetStream API object.
    """

    def __init__(self, server):
        self.server = server

    async def on(self, category: str = 'no_category', stream: str = 'documents', create: bool = False):
        """
        Connects to the NATS server, creates a stream and category if they don't exist, and prints a success message.

        Args:
            category (str, optional): The category to connect to. Defaults to 'no_category'.
            stream (str, optional): The stream to connect to. Defaults to 'documents'.
            create (bool, optional): Whether to create the stream and category if they don't exist. Defaults to False.
        """
        self.category = category
        self.stream = stream
        try:
            nc = await nats.connect(f'nats://{self.server}:4222')
            self.nc = nc
            self.js = self.nc.jetstream()
            self.js.purge_stream
            streams = await self.js.streams_info()
            if self.stream not in [x.config.name for x in streams] or self.category not in sum([x.config.subjects for x in streams], []):
                try:
                    if self.stream not in [x.config.name for x in streams]:
                        _f("wait", f'creating {self.stream}') \
                        , await self.js.add_stream(name=self.stream, subjects=[self.category]) \
                            if create else _f("warn", f"couldn't create {stream} on {self.server}")
                        streams = await self.js.streams_info()
                    if self.category not in sum([x.config.subjects for x in streams if x.config.name == self.stream], []):
                        subjects = sum([x.config.subjects for x in streams if x.config.name == self.stream], [])
                        print(subjects)
                        subjects.append(self.category)
                        await self.js.update_stream(StreamConfig(
                            name = self.stream
                            , subjects = subjects
                        ))
                        _f("success", f'created [{self.category}] on\n🛰️ stream: {self.stream}')
                except Exception as e:
                    _f('fatal', f"couldn't create {stream} on {self.server}\n{e}")
        except TimeoutError:
            _f('fatal', f'could not connect to {self.server}')
        _f("success", f'connected to [{self.category}] on\n🛰️ stream: {self.stream}')

    async def off(self):
        """
        Disconnects from the NATS server and prints a warning message.
        """
        await self.nc.drain()
        _f('warn', f'disconnected from {self.server}')

    async def pulse(self, payload):
        """
        Publishes data to the NATS server using the specified category and payload.

        Args:
            payload (dict): The data to be published.
        """
        try:
            bytes_ = json.dumps(asdict(payload), separators=(', ', ':')).encode('utf-8')
        except Exception as e:
            _f('fatal', f'invalid JSON\n{e}')
        try:
            await self.js.publish(self.category, bytes_)
        except Exception as e:
            _f('fatal', f'could not send data to {self.server}\n{e}')

    async def emp(self, name=None):
        """
        Deletes the specified stream if the name matches the current stream, or prints an error message if the name doesn't match or the stream doesn't exist.

        Args:
            name (str, optional): The name of the stream to delete. Defaults to None.
        """
        if name and name==self.stream:
            await self.js.delete_stream(name=self.stream)
            _f('warn', f'{self.stream} stream deleted')
        else:
            _f('fatal', "name doesn't match the stream or stream doesn't exist")

    async def reset(self, name=None):
        """
        Purges the specified category if the name matches the current category, or prints an error message if the name doesn't match or the category doesn't exist.

        Args:
            name (str, optional): The name of the category to purge. Defaults to None.
        """
        if name and name==self.category:
            await self.js.purge_stream(name=self.stream, subject=self.category)
            _f('warn', f'{self.category} category deleted')
        else:
            _f('fatal', "name doesn't match the stream category or category doesn't exist")

class Resonator:
    """
    The `Resonator` class is responsible for connecting to a NATS server, subscribing to a specific category in a stream, and consuming messages from that category.

    Args:
        server (str): The address of the NATS server.

    Attributes:
        server (str): The address of the NATS server.
        category (str): The category to subscribe to.
        stream (str): The stream to subscribe to.
        session (str): The session name for durable subscriptions.
        nc (nats.aio.client.Client): The NATS client.
        js (nats.aio.client.JetStream): The JetStream client.
        sub (nats.aio.client.Subscription): The subscription.

    Methods:
        __init__(self, server: str)
        on(self, category: str = 'no_category', stream: str = 'documents', cb=print, session='magnet') -> None
        info(self, session: str = None) -> None
        off(self) -> None
    """

    def __init__(self, server: str):
        """
        Initializes the `Resonator` class with the NATS server address.

        Args:
            server (str): The address of the NATS server.
        """
        self.server = server

    async def on(self, category: str = 'no_category', stream: str = 'documents', cb=print, session='magnet'):
        """
        Connects to a NATS server, subscribes to a specific category in a stream, and consumes messages from that category.
    
        Args:
            category (str, optional): The category to subscribe to. Defaults to 'no_category'.
            stream (str, optional): The stream to subscribe to. Defaults to 'documents'.
            cb (function, optional): The callback function to process the received messages. Defaults to `print`.
            session (str, optional): The session name for durable subscriptions. Defaults to 'magnet'.
    
        Returns:
            None
    
        Raises:
            TimeoutError: If connection to the NATS server times out.
            Exception: If there is an error in consuming the message or processing the callback function.
        """
        self.category = category
        self.stream = stream
        self.session = session
        _f('wait',f'connecting to {self.server}')
        try:
            self.nc = await nats.connect(f'nats://{self.server}:4222')
            self.js = self.nc.jetstream()
            self.sub = await self.js.subscribe(self.category, durable=self.session)
            _f("success", f'connected to {self.server}')
        except TimeoutError:
            _f("fatal", f'could not connect to {self.server}')
        _f("info", f'consuming delta from [{self.category}] on\n🛰️ stream: {self.stream}\n🧲 session: "{self.session}"')
        while True:
            try:
                msg = await self.sub.next_msg(timeout=60)
                payload = Payload(**json.loads(msg.data))
                try:
                    await cb(payload, msg)
                except Exception as e:
                    _f("warn", f'retrying connection to {self.server}')
            except Exception as e:
                _f('fatal','invalid JSON')
    async def info(self, session: str = None):
        """
        Retrieves information about a consumer in a JetStream stream.

        :param session: A string representing the session name of the consumer. If not provided, information about all consumers in the stream will be retrieved.
        :return: None
        """
        jsm = await self.js.consumer_info(stream=self.stream, consumer=session)
        _f('info',json.dumps(jsm.config.__dict__, indent=2))
    async def off(self):
        """
        Unsubscribes from the category and stream and disconnects from the NATS server.

        :return: None
        """
        await self.sub.unsubscribe()
        _f('warn', f'unsubscribed from {self.stream}')
        await self.nc.drain()
        _f('warn', f'disconnected from {self.server}')