import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from datetime import datetime, timezone
from magnet.utils.data_classes import Status, Payload, Run
from magnet.utils.prism.models.cmamba.data_classes import DataProcessingConfig

class DataProcessor:
    def __init__(self, run: Run, magnet):
        self.scaler = StandardScaler()
        self.magnet = magnet
        self.run = run
        self.config = DataProcessingConfig(**run._job.params.processing_options)
        self.feature_columns = self.config.features_to_match if len(self.config.features_to_match) > 0 else []

    def process_chunk(self, chunk):
        self.magnet.status_callback(Status(datetime.now(timezone.utc), "info", "Processing a new chunk of data."))
        chunk[self.config.timestamp_column] = pd.to_datetime(chunk[self.config.timestamp_column], errors='coerce')
        chunk = chunk.dropna(subset=[self.config.timestamp_column])
        chunk = chunk.drop_duplicates(subset=[self.config.timestamp_column], keep='first')
        chunk = chunk.sort_values(self.config.timestamp_column)
        chunk = chunk[self.feature_columns + [self.config.timestamp_column]]
        self.magnet.status_callback(Status(datetime.now(timezone.utc), "info",
                                           f"Processed chunk with {chunk.shape[0]} rows and {chunk.shape[1]} columns."))
        return chunk

    def fit_scaler(self):
        self.magnet.status_callback(Status(datetime.now(timezone.utc), "info",
                                           f"Starting to fit scaler on data from {self.run._id}"))
        try:
            for chunk_index, chunk in enumerate(pd.read_csv(f'/tmp/{self.run._job._id}', chunksize=self.config.chunk_size)):
                self.magnet.status_callback(Status(datetime.now(timezone.utc), "info", f"Fitting scaler on chunk {chunk_index + 1}."))
                chunk = self.process_chunk(chunk)
                if not self.feature_columns:
                    self.feature_columns = [col for col in chunk.columns if chunk[col].dtype in ['float64', 'int64']]
                features = chunk[self.feature_columns].values
                self.scaler.partial_fit(features)
            self.magnet.status_callback(Status(datetime.now(timezone.utc), "info", "Scaler fitting completed."))
        except Exception as e:
            self.magnet.status_callback(Status(datetime.now(timezone.utc), "fatal", f"Error during scaler fitting: {str(e)}"))
            raise e

    async def transform_data(self):
        self.magnet.status_callback(Status(datetime.now(timezone.utc), "info", f"Starting to transform data from {self.run._id}"))
        processed_chunks = []
        try:
            for chunk_index, chunk in enumerate(pd.read_csv(f'/tmp/{self.run._job._id}', chunksize=self.config.chunk_size)):
                self.magnet.status_callback(Status(datetime.now(timezone.utc), "info", f"Transforming chunk {chunk_index + 1}"))
                chunk = self.process_chunk(chunk)
                features = chunk[self.feature_columns].values
                scaled_features = self.scaler.transform(features)
                processed_chunks.append(scaled_features)
                self.magnet.status_callback(Status(datetime.now(timezone.utc), "info", f"Chunk {chunk_index + 1} transformed"))

            features = np.concatenate(processed_chunks, axis=0)
            del processed_chunks  # Free up memory
            self.magnet.status_callback(Status(datetime.now(timezone.utc), "info", f"All data transformed. Final shape: {features.shape}"))
            return features
        except Exception as e:
            self.magnet.status_callback(Status(datetime.now(timezone.utc), "fatal", f"Error during data transformation: {str(e)}"))
            raise e

    async def create_sequences(self, features):
        self.magnet.status_callback(Status(datetime.now(timezone.utc), "info", f"Starting to create sequences with input length {self.config.input_length}"))
        sequences = []
        subject_name = '.'.join(['runs', self.run._id, self.run._job.params.resource_id.split('.')[-1]])
        try:
            for start in range(0, len(features) - self.config.input_length + 1, self.config.input_length):
                sequence = features[start:start + self.config.input_length]
                sequences.append(sequence)
                payload = Payload(content=sequence.tolist(), _id=str(start))
                if self.run._job.params.data_source == 'local':
                    await self.magnet.charge.pulse(payload=payload, subject=subject_name, v=1)

            sequences = np.array(sequences)
            self.magnet.status_callback(Status(datetime.now(timezone.utc), "info",
                                               f"Sequence creation completed. Total sequences: {sequences.shape[0]}"))
            return sequences
        except Exception as e:
            self.magnet.status_callback(Status(datetime.now(timezone.utc), "fatal", f"Error during sequence creation: {str(e)}"))
            raise e

