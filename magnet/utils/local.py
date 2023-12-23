import requests
from magnet.utils.globals import _f
from magnet.utils.mlx import mistral
import json

class LocalInference:
    def __init__(self, model):
        """
        Initializes the LocalInference class with a pre-trained model.

        Args:
            model (str): The path to the pre-trained model used for inference.
        """
        self.model = model
    
    def invoke(self, payload):
        """
        Invokes a local inference using a pre-trained model.

        Args:
            payload (str): A JSON string containing the input parameters for the inference.

        Returns:
            str: The generated response from the local inference.
        """
        payload = json.loads(payload)
        payload = {
            "seed": 2077,
            "model_path": self.model,
            "temp": payload["parameters"]["temperature"],
            "prompt": payload["prompt"],
            "max_tokens": payload["parameters"]["max_new_tokens"],
            "tokens_per_eval": 10
        }
        response = mistral.generate(payload)
        return response