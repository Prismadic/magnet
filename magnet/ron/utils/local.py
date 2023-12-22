import requests
from magnet.utils import _f
from magnet.ron.utils.mlx import mistral
import json

class LocalInference:
    def __init__(self, model):
        self.model = model
    
    def invoke(self, payload):
        payload = json.loads(payload)
        payload = {
            "seed": 2077
            , "model_path": self.model
            , "temp": payload["parameters"]["temperature"]
            , "prompt": payload["prompt"]
            , "max_tokens": payload["parameters"]["max_new_tokens"]
            , "tokens_per_eval": 10
        }
        response = mistral.generate(payload)
        return response
