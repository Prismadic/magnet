from haystack.nodes import PromptTemplate
from magnet.utils import _f
from vllm import LLM
import requests
import json

class Ask:
    def __init__(self, server):
        self.server = server

    async def gen(self, m: str = "mistralai/Mistral-7B-Instruct-v0.1", q: str = "What is your itinerary?", t: float = 0.0, n: int = 8096, cb=None, p: str = "deepset/question-answering-with-references"):
        prompt = f'{PromptTemplate(p).prompt_text} \n\n {q}'
        _f('warn', 'prompt + query is longer than the maximum context length (n)') if len(prompt)>n else None
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
        if cb:
            cb(response.text)
        else:
            return response.text