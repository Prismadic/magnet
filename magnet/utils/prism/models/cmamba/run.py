from magnet.utils.data_classes import Payload, FilePayload, Run, Status
from datetime import datetime, timezone
import json
from dataclasses import asdict
from magnet.utils.prism.models.cmamba.data_classes import DataProcessingConfig, CMambaArgs
from magnet.utils.prism.models.cmamba.model.data_processor import DataProcessor
from magnet.utils.prism.models.cmamba.model.train import train_model
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split

class CMambaRun:
    def __init__(self, magnet):
        self.magnet = magnet

    async def _log_status(self, level: str, message: str):
        self.magnet.status_callback(Status(datetime.now(timezone.utc), level, message))

    async def _complete_run(self, run: Run, results: dict, metrics: dict):
        run.end_time = datetime.now(timezone.utc).isoformat()
        run.status = "completed"
        run.results = results
        run.metrics = metrics
        key = run._id
        await self.magnet.runs_kv.put(key=key, value=json.dumps(asdict(run)).encode('utf-8'))
        await self._log_status("info", f"Completed run {run._id} with status {run.status}")

    async def _claim(self, run: Run):
        run.is_claimed = True
        await self.magnet.runs_kv.put(key=run._id, value=json.dumps(asdict(run)).encode('utf-8'))

    async def _unclaim(self, run: Run):
        run.is_claimed = False
        await self.magnet.runs_kv.put(key=run._id, value=json.dumps(asdict(run)).encode('utf-8'))

    async def _handle_failure(self, run: Run, error_message: str):
        run.end_time = datetime.now(timezone.utc).isoformat()
        await self._unclaim(run)
        run.status = "failed"
        run.results = {"error": error_message}
        key = run._id
        await self.magnet.runs_kv.put(key=key, value=json.dumps(asdict(run)).encode('utf-8'))
        await self._log_status("fatal", f"Failed run {run._id}: {error_message}")

    async def _process(self, run: Run):
        try:
            print(run)
            config = DataProcessingConfig(**run._job.params.processing_options)
        except Exception as e:
            await self._log_status("fatal", f"Error initializing DataProcessingConfig: {e}")
            await self._handle_failure(run, str(e))
            return
        data_processor = DataProcessor(run, self.magnet)
        await self._log_status("info", "Fitting scaler with cmamba model...")
        data_processor.fit_scaler()

        await self._log_status("info", "Transforming data with cmamba model...")
        features = data_processor.transform_data()

        await self._log_status("info", "Creating sequences with cmamba model...")
        subject_name = '.'.join([run._id])
        await self._log_status("info", f"Pulsing sequences to stream: {subject_name}")

        sequences = await data_processor.create_sequences(features)

        await self._complete_run(run, {"sequences_created": sequences.shape[0]}, {"feature_shape": features.shape})

    async def _train(self, run: Run):
        try:
            config = CMambaArgs(**run.params.training_options)
            input_length = config.input_length
            forecast_length = config.forecast_len
            batch_size = config.batch_size

            train_datasets, test_datasets = [], []

            sequences, targets = None, None
            if run.params.data_source == 'local':
                try:
                    with open(f'/tmp/{run.params.resource_id}', 'rb') as f:
                        data = torch.load(f)
                        sequences = data['sequences']
                        targets = data['targets']
                except Exception as e:
                    await self._handle_failure(run, f"Error loading local data: {str(e)}")
                    return None, None

            elif run.params.data_source == 'stream':
                parent = run.params.resource_id
                parent = await self.magnet.jobs_kv.get(key=parent)
                parent = json.loads(parent.value).get("params")
                subject_name = '.'.join([run._id, parent])
                await self._log_status("info", f"Subscribing to stream: {subject_name}")

                sequences_list = []
                targets_list = []
                
                # Fetch batches of sequences and targets from the stream
                async for payload in self.magnet.resonator.listen(batch_size=1):
                    if isinstance(payload, Payload):
                        sequence = torch.tensor(payload.content)
                        target = sequence[:, input_length:, :]
                        sequence = sequence[:, :-forecast_length, :]
                        
                        sequences_list.append(sequence)
                        targets_list.append(target)
                        
                        if len(sequences_list) >= batch_size:
                            sequences = torch.cat(sequences_list, dim=0)
                            targets = torch.cat(targets_list, dim=0)
                            break

            if sequences is None or targets is None:
                await self._handle_failure(run, "No data available for training.")
                return None, None

            # Training process begins here
            X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2, random_state=42)

            X_train_tensor = torch.FloatTensor(X_train).permute(0, 2, 1)
            y_train_tensor = torch.FloatTensor(y_train).permute(0, 2, 1)
            X_test_tensor = torch.FloatTensor(X_test).permute(0, 2, 1)
            y_test_tensor = torch.FloatTensor(y_test).permute(0, 2, 1)

            train_datasets.append(TensorDataset(X_train_tensor, y_train_tensor))
            test_datasets.append(TensorDataset(X_test_tensor, y_test_tensor))

            train_dataset = ConcatDataset(train_datasets)
            test_dataset = ConcatDataset(test_datasets)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            del train_datasets, test_datasets  # Free up memory

            await self._log_status("info", f"Total training samples: {len(train_dataset)}")
            await self._log_status("info", f"Total test samples: {len(test_dataset)}")
            await self._log_status("info", f"Beginning training for {run._id}...")
            await train_model(self.magnet, run, train_loader, test_loader, config)

            return train_loader, test_loader

        except Exception as e:
            await self._log_status("fatal", f"Error during training initialization: {e}")
            await self._handle_failure(run, str(e))
            return None, None
        