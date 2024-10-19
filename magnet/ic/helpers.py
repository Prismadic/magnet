from datetime import datetime, timezone
import os
import csv
import json
from dataclasses import asdict
from magnet.utils.data_classes import Status, Job, AcquireParams, FilePayload, ProcessParams, TrainParams, Run
from magnet.utils.prism.models.cmamba.run import CMambaRun

class BaseHelpers:
    def __init__(self, magnet):
        self.magnet = magnet

    async def _claimant(self, run: Run, claim: bool, status: str = None, results: dict = None, metrics: dict = None):
        try:
            run._job._isClaimed = claim
            run.status = status
            run.results = results
            run.metrics = metrics
            if status in ["completed", "failed"]:
                run.end_time = datetime.now(timezone.utc).isoformat()
            elif status == "in_progress":
                run.start_time = datetime.now(timezone.utc).isoformat()
            
            await self.magnet.jobs_kv.put(key=run._job._id, value=json.dumps(asdict(run._job)).encode('utf-8'))
            await self.magnet.runs_kv.put(key=run._id, value=json.dumps(asdict(run)).encode('utf-8'))

            return run
            
        except Exception as e:
            self._log_status("fatal", f"Failed to claim job {run._job._id} with run {run._id}: {e}")
    
    async def _log_status(self, level: str, message: str, claim_: object = None):
        self.magnet.status_callback(Status(datetime.now(timezone.utc), level, message))

    async def _download_file(self, object_name: str, local_path: str, object_store):
        try:
            with open(local_path, 'wb') as local_file:
                data = await object_store.get(object_name)
                local_file.write(data.data)
            await self._log_status("info", f"File downloaded to {local_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download file {object_name}: {e}")

    async def _consume_stream_to_csv(self, csv_path: str, type_: str, id_: str, operation: str, resonator):
        # Define a callback to write data to CSV
        async def write_to_csv(payload, msg):
            data = json.loads(msg.data)
            if not os.path.exists(csv_path):
                with open(csv_path, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=data.keys())
                    writer.writeheader()
            with open(csv_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=data.keys())
                writer.writerow(data)

        # Listen to the stream and write to CSV
        subject_name = self._get_stream_subject(type_, id_=id_, operation=operation)
        await resonator.on(subject=subject_name)
        await resonator.listen(cb=write_to_csv)
        await resonator.off()

    async def _load_file_to_memory(self, id_: str) -> bytes:
        location = os.path.join('/tmp/', id_)
        if not os.path.exists(location):
            raise FileNotFoundError(f"File at location {location} not found.")

        await self._log_status("info", f"File found at location: {location}. Reading file...")
        with open(location, 'rb') as file:
            file_data = file.read()
        await self._log_status("success", f"File successfully loaded for id {id_}")
        return file_data


class RunHelpers(BaseHelpers):

    async def process(self, run: Run):
        await self._claimant(run, claim=True, status="in_progress")
        await self._log_status("info", f"Processing run: {run._id}")

        try:
            job_params = ProcessParams(**run._job.params)
            run._job.params = job_params
            if isinstance(job_params, ProcessParams):
                if job_params.model == "cmamba":
                    cmamba = CMambaRun(self.magnet)
                    await cmamba._process(run)
                    # After processing, results should be stored in the runs object store
                    await self._claimant(run, claim=False, status="completed", results={"processed": True})
                else:
                    await self._log_status("info", f"Run {run._id} does not require cmamba processing")
                    await self._claimant(run, claim=False, status="failed", results=None, metrics=None)
            else:
                await self._log_status("warn", f"Run {run._id} does not have valid ProcessParams")
                await self._claimant(run, claim=False, status="failed", results=None, metrics=None)
        except Exception as e:
            await self._log_status("fatal", f"An error occurred while processing run {run._id}: {e}")
            await self._claimant(run, claim=False, status="failed", results=None, metrics=None)

    async def acquire(self, run: Run):
        await self._claimant(run, claim=True, status="in_progress")
        await self._log_status("info", f"Claimed job: {run._job._id}")

        try:
            job_params = AcquireParams(**run._job.params)
            if isinstance(job_params, AcquireParams):
                if job_params.data_source == "local":
                    location = os.path.expanduser(job_params.location)
                    if not os.path.exists(location):
                        await self._claimant(run, claim=False, status="failed", results=None, metrics=None)
                        return
                    file_data = await self._load_file_to_memory(location)

                    file_payload = FilePayload(
                        original_filename=os.path.basename(location),
                        data=file_data,
                        _id=run._job._id
                    )
                    object_name = job_params.resource_id

                    # If this is an acquisition run, data should be uploaded to the jobs object store
                    await self._log_status("info", f"Pulsing {object_name} to jobs object store")
                    await self.magnet.jobs_os.put(object_name, file_payload.data)

                    await self._log_status("success", f"{object_name} successfully pulsed to jobs object store for run {run._id}")
                    await self._claimant(run, claim=True, status="completed", results={"file_pulsed": True}, metrics={"file_size": len(file_data)})
                elif job_params.data_source == "stream_to_csv":
                    stream_name = job_params.acquisition_options.get("stream_name")
                    csv_path = os.path.join('/tmp/', run._id)

                    await self._log_status("info", f"Consuming data from stream {stream_name} and writing to {csv_path}")
                    await self._consume_stream_to_csv(csv_path, run.run_type, run._id, "data", self.magnet.resonator)

                    await self._log_status("success", f"Data successfully written to {csv_path} for run {run._id}")
                    await self._claimant(run, claim=True, status="completed", results={"file_written": csv_path}, metrics={"lines_written": len(open(csv_path).readlines()) - 1})
                elif job_params.data_source == "object_store":
                    object_name = run._job.params["resource_id"]
                    local_path = os.path.join('/tmp/', object_name)

                    await self._log_status("info", f"Downloading object {object_name} to {local_path}")
                    await self._download_file(object_name, local_path, self.magnet.jobs_os)

                    await self._log_status("success", f"Object successfully downloaded to {local_path} for run {run._id}")
                    await self._claimant(run, claim=True, status="completed", results={"file_downloaded": local_path}, metrics={"file_size": os.path.getsize(local_path)})
                else:
                    raise ValueError(f"Acquisition data_source '{job_params.data_source}' is not supported.")
            else:
                await self._log_status("warn", f"Run {run._id} does not have valid AcquireParams")
                await self._claimant(run, claim=False, status="failed", results=None, metrics=None)
        except Exception as e:
            await self._log_status("fatal", f"Run {run._id} does not have valid params: {e}")
            await self._claimant(run, claim=False, status="failed", results=None, metrics=None)

    async def inference(self, run: Run):
        await self._claimant(run, claim=True, status="in_progress")
        await self._log_status("info", f"Inference run: {run._id}")

        # Implement inference logic here using appropriate kv, object store, and stream subjects
        await self._claimant(run, claim=True, status="completed", results=None, metrics=None)

    async def train(self, run: Run):
        await self._claimant(run, claim=True, status="in_progress")
        await self._log_status("info", f"Training run: {run._id}")
        
        try:
            # Initialize TrainParams correctly from job params
            run.params = TrainParams(**run._job.params)
            
            if isinstance(run.params, TrainParams):
                if run.params.model == "cmamba":
                    cmamba = CMambaRun(self.magnet)
                    await cmamba._train(run)

                    # After training, results should be stored in the runs object store
                    await self._claimant(run, claim=True, status="completed", results={"trained": True})
                else:
                    await self._log_status("info", f"Run {run._id} does not require cmamba training")
                    await self._claimant(run, claim=True, status="failed", results=None, metrics=None)
            else:
                await self._log_status("warn", f"Run {run._id} does not have valid TrainParams")
                await self._claimant(run, claim=True, status="failed", results=None, metrics=None)
        
        except TypeError as te:
            await self._log_status("fatal", f"An error occurred while training for run {run._id}: {te}")
            print(f"TypeError: {te}")
            await self._claimant(run, claim=False, status="failed", results=None, metrics=None)

        except Exception as e:
            print(e)
            await self._log_status("fatal", f"An error occurred while training for run {run._id}: {e}")
            await self._claimant(run, claim=False, status="failed", results=None, metrics=None)
