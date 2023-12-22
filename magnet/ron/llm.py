from magnet.utils import _f
from .utils.huggingface import InferenceAPI
from .utils.local import LocalInference
from .utils.prompts import *
from .utils.data_classes import *
import requests, json

class Generate:
    def __init__(self, server: str = None, field = None, hf_token: str = None):
        self.server = server if not hf_token else None
        self.field = field
        self.token = hf_token

    async def on(self):
        if self.field:
            pass

    async def ask(self
                  , m: str = "mistralai/Mistral-7B-Instruct-v0.1"
                  , q: str = "What is your itinerary?"
                  , t: float = 1.0
                  , n: int = 8096
                  , p: str = "qa_ref"
                  , cb: object = None
                  , docs: list = []
                  , v: bool = False
                ):
        prompt = getattr(globals()['Prompts'](), p)(docs,q)
        _f('warn', '(p + q + d) > n') if len(prompt) > n else None
        payload = json.dumps({
            "model": m,
            "prompt": prompt,
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": n
                , "temperature": t,
            }
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
        elif v:
            response = requests.request(
                "POST", f"{self.server}/v1/completions", headers=headers, data=payload).text
        else:
            llm = LocalInference(model=m)
            response = llm.invoke(payload)
        if self.field:
            payload = GeneratedPayload(
                        query=q
                        , prompt=prompt
                        , context=docs
                        , result=response
                        , model=m
                    )
            try:
                await self.field.pulse(payload)
            except Exception as e:
                _f('fatal',e)
        if cb:
            cb(response)
        else:
            return response
