import requests
import json

class InferenceAPI:
    """
    A class that provides a convenient way to make HTTP POST requests to an inference API endpoint and retrieve the response.

    Args:
        token (str): The authorization token used for making requests to the inference API.

    Example Usage:
        ```python
        # Create an instance of the InferenceAPI class with a token
        api = InferenceAPI("your_token_here")

        # Prepare the payload as a JSON string
        payload = '{"model": "your_model_name", "input_data": "your_input_data"}'

        # Invoke the API by passing the payload
        response = api.invoke(payload)

        # Print the response
        print(response)
        ```

    Attributes:
        token (str): The authorization token used for making requests to the inference API.
    """

    def __init__(self, token):
        """
        Initializes an instance of the InferenceAPI class with a token for authorization.

        Args:
            token (str): The authorization token used for making requests to the inference API.
        """
        self.token = token
    
    def invoke(self, payload):
        """
        Makes an HTTP POST request to the inference API endpoint using the provided payload and returns the response as a JSON string.

        Args:
            payload (str): The payload to be sent in the request as a JSON string.

        Returns:
            str: The response from the inference API as a JSON string.
        """
        # Existing code for invoking the API and returning the response
        pass
            payload (str): A JSON string representing the payload to be sent to the inference API. It should contain the model name and input data.

        Returns:
            str: A JSON string representing the response from the inference API.
        """
        payload = json.loads(payload)
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.post(f"https://api-inference.huggingface.co/models/{payload['model']}", headers=headers, json=payload)
        return response.json()
