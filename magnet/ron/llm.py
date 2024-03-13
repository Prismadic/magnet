from dataclasses import asdict

from magnet.utils.globals import _f
from magnet.utils.huggingface import InferenceAPI
from magnet.utils.local import LocalInference
from magnet.utils.prompts import *
from magnet.utils.data_classes import *

import requests, json

class LLM:
    def __init__(self, server: str = None, field = None, hf_token: str = None):
        self.server = server if not hf_token else None
        self.field = field
        self.token = hf_token

    async def ask(self, params: AskParameters = None):
        prompt = getattr(globals()['Prompts'](), params.p)(params.docs, params.q)
        _f('warn', '(p + q + d) > n') if len(prompt) > params.n else None
        
        headers = {
            'Content-Type': 'application/json'
        }

        generic_payload = json.dumps({
            "model": params.m,
            "prompt": prompt,
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": params.n
                , "temperature": params.t,
            }
        })

        if self.token:
            api = InferenceAPI(self.token)
            response = api.invoke(generic_payload)
            if not isinstance(response, list) and 'error' in response.keys():
                return _f('fatal', response['error'])
            else:
                response = response[0]['generated_text'].split(json.loads(generic_payload)['prompt'])[1]
        elif params.vllm:
            response = requests.request(
                "POST", f"{self.server}/v1/completions", headers=headers, data=generic_payload).text
        else:
            engine = LocalInference(model=params.m)
            response = engine.invoke(payload)
        if self.field:
            payload = GeneratedPayload(
                        query=params.q
                        , prompt=prompt
                        , context=params.docs
                        , result=response
                        , model=params.m
                    )
            try:
                await self.field.pulse(payload)
            except Exception as e:
                _f('fatal',e)
        if params.cb:
            params.cb(response)
        else:
            return response
