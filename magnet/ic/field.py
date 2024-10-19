import json
import uuid
from datetime import datetime, timezone
import xxhash
import platform
import asyncio

from dataclasses import asdict

from magnet.utils.data_classes import Status, Payload, InferenceParams, TrainParams, AcquireParams, FilePayload, GeneratedPayload, EmbeddingPayload, JobParams, Job, ProcessParams
from magnet.ic.helpers import *
from nats.errors import TimeoutError
from nats.js.api import StreamConfig, ConsumerConfig
from nats.js.errors import ServerError
from nats.js.api import ObjectMeta

x = xxhash
utc_now = datetime.now(timezone.utc)
utc_timestamp = utc_now.timestamp()

class Charge:
    def __init__(self, magnet):
        self.magnet = magnet
        self.help = RunHelpers(magnet)  # Use RunHelpers for run-related operations

    async def on(self, subject=None):
        category = self.magnet.config.category if not subject else subject
        try:
            streams = await self.magnet.js.streams_info()
            remote_streams = [x.config.name for x in streams]
            remote_subjects = [x.config.subjects for x in streams]
            if self.magnet.config.stream_name not in remote_streams:
                self.magnet.status_callback(Status(datetime.now(
                    timezone.utc), 'fatal', f'{self.magnet.config.stream_name} not found, initialize with `Magnet.align()` first'))
                return
            elif category not in sum(remote_subjects, []):
                if category not in sum([x.config.subjects for x in streams if x.config.name == self.magnet.config.stream_name], []):
                    try:
                        subjects = sum(
                            [x.config.subjects for x in streams if x.config.name == self.magnet.config.stream_name], [])
                        subjects.append(category)
                        await self.magnet.js.update_stream(StreamConfig(
                            name=self.magnet.config.stream_name,
                            subjects=subjects
                        ))
                        self.magnet.status_callback(Status(datetime.now(
                            timezone.utc), "success", f'created [{category}] on 🛰️ stream: {self.magnet.config.stream_name}'))
                    except ServerError as e:
                        print(e)
                        self.magnet.status_callback(Status(datetime.now(
                            timezone.utc), 'fatal', f"couldn't create {self.magnet.config.stream_name} on {self.magnet.config.host}, ensure your `category` is set"))
        except TimeoutError:
            self.magnet.status_callback(Status(
                datetime.now(timezone.utc), 'fatal', f'could not connect to {self.magnet.config.host}'))

    async def off(self):
        await self.magnet.nc.drain()
        await self.magnet.nc.close()
        self.magnet.status_callback(
            Status(datetime.now(timezone.utc), 'warn', f'disconnected from {self.magnet.config.host}'))

    async def pulse(self, payload: Payload | FilePayload | GeneratedPayload | EmbeddingPayload | JobParams = None, subject: str = None, stream: str = None, v=False):
        try:
            if isinstance(payload, FilePayload):
                payload_data_bytes = payload.data
                bucket_name = await self.help._get_object_store_name(payload._id)
                meta = ObjectMeta(name=payload._id, headers={
                    "ext": payload.original_filename.split('.')[-1]
                })
                bucket = await self.magnet.js.object_store(bucket_name)
                await bucket.put(payload._id, payload_data_bytes, meta=meta)
                if v:
                    self.magnet.status_callback(Status(datetime.now(
                        timezone.utc), 'success', f'uploaded to object store in bucket {bucket_name} as {payload._id}'))
            elif isinstance(payload, Payload):
                bytes_ = json.dumps(asdict(payload), separators=(
                    ', ', ':')).encode('utf-8')
                _hash = x.xxh64(bytes_).hexdigest()
                subject_name = subject if subject else self.magnet.config.category
                msg = await self.magnet.js.publish(
                    subject=subject_name, payload=bytes_, stream=stream, headers={
                        "Nats-Msg-Id": _hash
                    }
                )
                if v:
                    self.magnet.status_callback(Status(datetime.now(
                        timezone.utc), 'success', f'pulsed {payload._id} to {subject_name} on {self.magnet.config.stream_name}'))
                _ts = datetime.now(timezone.utc)
                msg.ts = _ts
                return msg
        except Exception as e:
            self.magnet.status_callback(Status(datetime.now(
                timezone.utc), 'fatal', f'could not pulse data to {self.magnet.config.host}\n{e}'))
            return

    async def excite(self, job_type: str, params: ProcessParams | InferenceParams | TrainParams | AcquireParams):
        try:
            job_params = params
            job_id = f"{job_type}.{self.magnet.config.session}.{uuid.uuid4().hex[:8]}"
            job = Job(params=job_params, _type=job_type, _id=job_id)
            await self.magnet.jobs_kv.put(key=job._id, value=json.dumps(asdict(job)).encode('utf-8'))
            self.magnet.status_callback(
                Status(datetime.now(timezone.utc), 'info', f'Created {job_type} job {job._id}')
            )
            return job
        except Exception as e:
            self.magnet.status_callback(
                Status(datetime.now(timezone.utc), 'fatal', f'Failed to create {job_type} job\n{e}')
            )
            return None

    async def emp(self, name=None):
        if name and name == self.magnet.config.stream_name:
            await self.magnet.js.delete_stream(name=self.magnet.config.stream_name)
            self.magnet.status_callback(Status(
                datetime.now(timezone.utc), 'warn', f'{self.magnet.config.stream_name} stream deleted'))
        else:
            self.magnet.status_callback(Status(datetime.now(
                timezone.utc), 'fatal', "name doesn't match the stream or stream doesn't exist"))

    async def reset(self, name=None):
        if name and name == self.magnet.config.category:
            await self.magnet.js.purge_stream(name=self.magnet.config.stream_name, subject=self.magnet.config.category)
            self.magnet.status_callback(Status(
                datetime.now(timezone.utc), 'warn', f'{self.magnet.config.category} category deleted'))
        else:
            self.magnet.status_callback(Status(datetime.now(
                timezone.utc), 'fatal', "name doesn't match the stream category or category doesn't exist"))

class Resonator:
    def __init__(self, magnet):
        self.magnet = magnet
        self.run_helpers = RunHelpers(magnet)  # Use RunHelpers for run-related operations
        self.node = None
        self.durable = None
        self.consumer_config = None
        self.sub = None

    async def on(self, role: str, local: bool = False, bandwidth: int = 1000, obj=False, subject=None):
        try:
            subject_name = self.magnet.config.category if not subject else subject
            streams = await self.magnet.js.streams_info()
            remote_streams = [x.config.name for x in streams]
            remote_subjects = [x.config.subjects for x in streams]
            if self.magnet.config.stream_name not in remote_streams:
                self.magnet.status_callback(Status(datetime.now(
                    timezone.utc), 'fatal', f'{self.magnet.config.stream_name} not found, initialize with `Magnet.align()` first'))
                return
            elif subject_name not in sum(remote_subjects, []):
                if subject_name not in sum([x.config.subjects for x in streams if x.config.name == self.magnet.config.stream_name], []):
                    try:
                        subjects = sum(
                            [x.config.subjects for x in streams if x.config.name == self.magnet.config.stream_name], [])
                        subjects.append(subject_name)
                        await self.magnet.js.update_stream(StreamConfig(
                            name=self.magnet.config.stream_name,
                            subjects=subjects
                        ))
                        self.magnet.status_callback(Status(datetime.now(
                            timezone.utc), "success", f'created [{subject_name}] on\n🛰️ stream: {self.magnet.config.stream_name}'))
                    except ServerError as e:
                        print(e)
                        self.magnet.status_callback(Status(datetime.now(
                            timezone.utc), 'fatal', f"couldn't create {self.magnet.config.stream_name} on {self.magnet.config.host}, ensure your `category` is set"))
        except TimeoutError:
            self.magnet.status_callback(Status(
                datetime.now(timezone.utc), 'fatal', f'could not connect to {self.magnet.config.host}'))
        self.node = f'{platform.node()}_{xxhash.xxh64(platform.node(), seed=int(datetime.now(timezone.utc).timestamp())).hexdigest()}' if local else platform.node()
        self.durable = f'{self.node}_{role}'  # Include the role in the durable name for clarity
        self.consumer_config = ConsumerConfig(
            ack_policy="explicit",
            max_ack_pending=bandwidth,
            ack_wait=3600
        )
        self.magnet.status_callback(Status(datetime.now(
            timezone.utc), 'wait', f'connecting to {self.magnet.config.host.split("@")[1]} for role {role}'))
        try:
            if obj:
                self.sub = await self.magnet.os.watch(include_history=False)
                self.magnet.status_callback(Status(datetime.now(
                    timezone.utc), 'info', f'subscribed to object store: {self.magnet.config.os_name} as {self.node}')
                )
            else:
                self.magnet.js.__dict__
                self.sub = await self.magnet.js.pull_subscribe(
                    subject=subject_name,
                    config=self.consumer_config
                )
                self.magnet.status_callback(Status(datetime.now(
                    timezone.utc), 'info', f'joined worker queue: {self.magnet.config.session} as {self.node} for role {role}'))
        except Exception as e:
            self.magnet.status_callback(
                Status(datetime.now(timezone.utc), 'fatal', str(e)))
            return

    async def listen(self, batch_size=None, v=False):
        try:
            # Check if subscription is initialized
            if self.sub is None:
                self.magnet.status_callback(
                    Status(datetime.now(timezone.utc), 'fatal', 'No subscriber initialized'))
                return
            
            while True:
                try:
                    # Fetch a batch of messages (replace 10 with the desired batch size)
                    if batch_size:
                        msgs = await self.sub.fetch(batch_size)
                    for msg in msgs:
                        if v:
                            self.magnet.status_callback(Status(datetime.now(
                                timezone.utc), 'info', f'received {msg.subject}'))
                        yield msg

                except TimeoutError as e:
                    self.magnet.status_callback(Status(datetime.now(
                        timezone.utc), 'warn', 'No new messages.'))

        except Exception as e:
            self.magnet.status_callback(
                Status(datetime.now(timezone.utc), 'fatal', f'Error in listen method: {e}'))
            return


    async def worker(self, role=None):
        await self.on(role=role)  # Ensure the resonator is set up for the specific role
        self.magnet.status_callback(Status(datetime.now(
            timezone.utc), "info", f'processing jobs for role [{role}] from [{self.magnet.config.kv_name}] on\n🛰️ object store: {self.magnet.config.os_name}'))
        try:
            kv_store = self.magnet.jobs_kv  # Use the jobs KV store for job retrieval
            keys = await kv_store.keys()
            for key in keys:
                _job = await kv_store.get(key)
                job_data = json.loads(_job.value.decode('utf-8'))
                job = Job(
                    # make params the dataclass through role/type
                    job_data['params'], job_data['_type'], job_data['_id'], job_data['_isClaimed'])

                if not job._isClaimed and job._type == role:
                    # Claim the job first
                    run = Run(
                        _id=job._id,
                        _type=role,
                        _job=job,
                        start_time=datetime.now(timezone.utc).isoformat(),
                    )
                    run = await self.run_helpers._claimant(run, claim=True, status="in_progress")

                    # Process the run
                    await self.handle_run(run)
                else:
                    pass
        except Exception as e:
            self.magnet.status_callback(
                Status(datetime.now(timezone.utc), 'fatal', f'invalid JSON\n{e}'))

    async def handle_run(self, run: Run):
        self.magnet.status_callback(Status(datetime.now(
            timezone.utc), "info", f"Handling run of type: {run._type}"))

        try:
            # Log the start of the run
            self.magnet.status_callback(Status(datetime.now(
                timezone.utc), "info", f"Starting run {run._id}"))

            # Find the appropriate handler for the run type
            run_helpers = {
                'process': self.run_helpers.process,
                'inference': self.run_helpers.inference,
                'train': self.run_helpers.train,
                'acquire': self.run_helpers.acquire
            }

            run_handler = run_helpers.get(run._type)
            if run_handler:
                # Execute the run using the appropriate handler
                await run_handler(run)
            else:
                self.magnet.status_callback(
                    Status(datetime.now(timezone.utc), "warn", f"Unknown run type: {run._type}"))
                run.status = "failed"

        except Exception as e:
            run.status = "failed"
            run.end_time = datetime.now(timezone.utc)
            self.magnet.status_callback(
                Status(datetime.now(timezone.utc), "fatal", f"Run {run._id} failed: {e}"))

        finally:
            # Store or log the run result here if needed
            await self._store_run(run)

    async def _store_run(self, run: Run):
        # Convert datetime objects to ISO format strings for serialization
        run_dict = asdict(run)

        # Ensure datetime objects are converted to ISO format strings
        run_dict['start_time'] = run_dict.get('start_time')
        run_dict['end_time'] = run_dict.get('end_time')

        try:
            await self.magnet.runs_kv.put(key=run._id, value=json.dumps(run_dict).encode('utf-8'))
            self.magnet.status_callback(
                Status(datetime.now(timezone.utc), "info", f"Run {run._id} stored successfully"))
        except Exception as e:
            self.magnet.status_callback(
                Status(datetime.now(timezone.utc), "warn", f"Failed to store run {run._id}: {e}"))

    async def info(self):
        jsm = await self.magnet.js.consumer_info(stream=self.magnet.config.stream_name, consumer=self.magnet.session)
        self.magnet.status_callback(Status(datetime.now(
            timezone.utc), 'info', json.dumps(jsm.config.__dict__, indent=2)))

    async def off(self):
        await self.sub.unsubscribe()
        self.magnet.status_callback(Status(datetime.now(
            timezone.utc), 'warn', f'unsubscribed from {self.magnet.config.stream_name}'))
        await self.magnet.nc.drain()
        self.magnet.status_callback(Status(datetime.now(
            timezone.utc), 'warn', f'safe to disconnect from {self.magnet.config.host.split("@")[1]}'))
