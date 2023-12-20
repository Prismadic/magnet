from magnet.utils import _f
from .utils.huggingface import InferenceAPI
from .utils.prompts import *
import requests, json

class Generate:
    def __init__(self, server: str = None, field = None, hf_token: str = None):
        self.server = server if not hf_token else None
        self.field = field
        self.token = hf_token

    async def ask(self
                  , m: str = "mistralai/Mistral-7B-Instruct-v0.1"
                  , q: str = "What is your itinerary?"
                  , t: float = 0.0
                  , n: int = 8096
                  , p: str = "qa_ref"
                  , cb: object = None
                  , docs: list = []
                ):
        if self.field:
            await self.field.on(category=self.stream.category, stream=self.stream.stream)
        prompt = getattr(globals()['Prompts'](), p)(docs,q)
        _f('warn', '(p + q) > n') if len(prompt) > n else None
        payload = json.dumps({
            "model": m,
            "prompt": prompt,
            "max_tokens": 8096,
            "max_new_tokens": 8096,
            "temperature": t,
            "inputs": prompt
        })
        headers = {
            'Content-Type': 'application/json'
        }
        if self.token:
            llm = InferenceAPI(self.token)
            response = llm.invoke(payload)
            if not isinstance(response, list) and 'error' in response.keys():
                return _f('fatal', response['error'])
            else:
                response = response[0]['generated_text'].split(json.loads(payload)['prompt'])[1]
        else:
            response = requests.request(
                "POST", f"{self.server}/v1/completions", headers=headers, data=payload).text
        await self.field.pulse(prompt, response) if self.field else None
        if cb:
            cb(response)
        else:
            return response
