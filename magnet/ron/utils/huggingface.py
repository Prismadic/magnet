import requests
from magnet.utils import _f
import json
class InferenceAPI:
    def __init__(self, token):
        self.token = token
    
    def invoke(self, payload):
        payload = json.loads(payload)
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.post(f"https://api-inference.huggingface.co/models/{payload['model']}", headers=headers, json=payload)
        return response.json()
