import requests
from magnet.utils.globals import _f
from magnet.utils.mlx import mistral
import json

class LocalInference:
    """
    A class for performing local inference using a pre-trained model.

    Attributes:
        model (str): The path to the pre-trained model.

    Methods:
        __init__(self, model):
            Initializes the LocalInference class with the model path.

        invoke(self, payload):
            Invokes the local inference using the provided payload.

    Example Usage:
        # Create an instance of the LocalInference class
        inference = LocalInference(model_path)

        # Define the input parameters for the inference
        payload = {
            "parameters": {
                "temperature": 0.8,
                "max_new_tokens": 100
            },
            "prompt": "Hello, how are you?"
        }

        # Invoke the local inference
        response = inference.invoke(json.dumps(payload))

        # Print the generated response
        print(response)
    """

    def __init__(self, model):
        """
        Initializes the LocalInference class with the model path.

        Args:
            model (str): The path to the pre-trained model.
        """
        self.model = model
    
    def invoke(self, payload):
        """
        Invokes the local inference using the provided payload.

        Args:
            payload (str): A JSON string containing the input parameters for the inference.

        Returns:
            str: The generated response from the local inference.
        """
        payload = json.loads(payload)
        payload = {
            "parameters": {
                "temperature": 0.8,
                "max_new_tokens": 100
            },
            "prompt": "Hello, how are you?"
        }
        # Call the mistral.generate() function to generate the response
        response = mistral.generate(payload)
        return response
            "seed": 2077,
            "model_path": self.model,
            "temp": payload["parameters"]["temperature"],
            "prompt": payload["prompt"],
            "max_tokens": payload["parameters"]["max_new_tokens"],
            "tokens_per_eval": 10
        }
        response = mistral.generate(payload)
        return response
