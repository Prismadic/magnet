from haystack.nodes import PromptTemplate
from magnet.utils import _f
from vllm import LLM
import requests
import json

class Generate:
    def __init__(self, server, field=None):
        self.server = server
        self.field = field
        
    async def ask(self, m: str = "mistralai/Mistral-7B-Instruct-v0.1", q: str = "What is your itinerary?", t: float = 0.0, n: int = 8096, cb=None, p: str = "deepset/question-answering-with-references"):
        if self.field:
            await self.field.on(category=self.stream.category, stream=self.stream.stream)
        prompt = f'{PromptTemplate(p).prompt_text} \n\n {q}'
        _f('warn', '(p + q) > n') if len(prompt)>n else None
        payload = json.dumps({
            "model": m,
            "prompt": prompt,
            "max_tokens": 8096,
            "temperature": 0
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", f"{self.server}/v1/completions", headers=headers, data=payload)
        await self.field.pulse(prompt, response.text) if self.field else None
        if cb:
            cb(response.text)
        else:
            return response.text