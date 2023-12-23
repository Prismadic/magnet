import requests
import json

class InferenceAPI:
    """
    A class that provides a convenient way to make HTTP POST requests to an inference API endpoint.

    Attributes:
        token (str): A string representing the token used for authorization.
    """

    def __init__(self, token):
        """
        Initializes the InferenceAPI class with a token.

        Args:
            token (str): A string representing the token used for authorization.
        """
        self.token = token
    
    def invoke(self, payload):
        """
        Makes an HTTP POST request to an inference API endpoint and returns the response.

        Args:
            payload (str): A JSON string representing the payload to be sent to the inference API. It should contain the model name and input data.

        Returns:
            str: A JSON string representing the response from the inference API.
        """
        payload = json.loads(payload)
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.post(f"https://api-inference.huggingface.co/models/{payload['model']}", headers=headers, json=payload)
        return response.json()