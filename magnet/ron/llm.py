from magnet.utils.globals import _f

from magnet.utils.llm.api import InferenceAPI
from magnet.utils.llm.local import LocalInference
from magnet.utils.llm.prompts import *

from magnet.utils.data_classes import *

import requests, json

class LLM:
    def __init__(self, server: str = None, field = None, token: str = None, provider='openai'):
        self.server = server if not token else None
        self.field = field
        self.provider = provider
        self.token = token

    async def ask(self, params: AskParameters = None):
        prompt = getattr(globals()['Prompts'](params), params.p)().replace('\n', ' ')
        _f('warn', '(p + q + d) > n') if len(prompt) > params.n else None
        
        headers = {
            'Content-Type': 'application/json'
        }

        generic_payload = {
            "model": params.m,
            "prompt": prompt,
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": params.n
                , "temperature": params.t,
            }
        }

        if self.token:
            api = InferenceAPI(self.token, self.provider)
            response = api.invoke(generic_payload)
            if not isinstance(response, str):
                return _f('fatal', response)
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
