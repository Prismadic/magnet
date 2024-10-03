import torch
from datetime import datetime
from magnet.base import Magnet
from magnet.utils.data_classes import Status
from magnet.utils.prism.models.cmamba.model import CMamba
from magnet.utils.prism.models.cmamba.data_classes import CMambaArgs

class CMambaPrediction:
    def __init__(self, magnet: Magnet, model_args: CMambaArgs, model_path: str):
        self.magnet = magnet
        self.model_args = model_args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize and load the model
        self.model = CMamba(self.model_args, self.magnet)
        self.model.to(self.device)
        self.load_model(model_path)

    def load_model(self, model_path: str):
        try:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()  # Set the model to evaluation mode
            self.magnet.status_callback(Status(datetime.now(), "info", "Model loaded and set to evaluation mode"))
        except Exception as e:
            self.magnet.status_callback(Status(datetime.now(), "fatal", f"Failed to load model: {str(e)}"))
            raise

    def predict(self, input_data: torch.Tensor, return_embeddings: bool = False):
        self.magnet.status_callback(Status(datetime.now(), "info", "Starting prediction"))
        input_data = input_data.to(self.device)

        with torch.no_grad():
            predictions = self.model(input_data, return_embeddings=return_embeddings)

        self.magnet.status_callback(Status(datetime.now(), "success", "Prediction completed"))
        return predictions.cpu().numpy()

    def get_embeddings(self, selected_chunks: torch.Tensor):
        self.magnet.status_callback(Status(datetime.now(), "info", "Extracting embeddings"))
        embeddings = self.predict(selected_chunks, return_embeddings=True)
        reshaped_embeddings = embeddings.reshape(embeddings.shape[0], -1)
        self.magnet.status_callback(Status(datetime.now(), "info", f"Embeddings shape: {reshaped_embeddings.shape}"))
        return reshaped_embeddings
