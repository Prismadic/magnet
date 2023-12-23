from magnet.utils.globals import _f
from magnet.utils.huggingface import InferenceAPI
from magnet.utils.local import LocalInference
from magnet.utils.prompts import *
from magnet.utils.data_classes import *
import requests, json

class Generate:
    def __init__(self, server: str = None, field = None, hf_token: str = None):
        """
        Initializes the Generate class.

        Args:
            server (str): The URL of the server to be used for generating the response. Default is None.
            field: Placeholder field that can be used for future implementation.
            hf_token (str): The Hugging Face token to be used for authentication when using the local inference API. Default is None.
        """
        self.server = server if not hf_token else None
        self.field = field
        self.token = hf_token

    async def on(self):
        """
        Placeholder method that can be used for future implementation.
        """
        if self.field:
            pass # todo

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
        """
        Generates a response based on a given prompt using a language model.

        Args:
            m (str): The model name or identifier to be used for generating the response. Default is "mistralai/Mistral-7B-Instruct-v0.1".
            q (str): The question or prompt for which a response is to be generated. Default is "What is your itinerary?".
            t (float): The temperature parameter controlling the randomness of the generated response. Default is 1.0.
            n (int): The maximum number of new tokens to be generated in the response. Default is 8096.
            p (str): The type of prompt to be used for generating the response. Default is "qa_ref".
            cb (object): An optional callback function to be executed with the generated response. Default is None.
            docs (list): A list of additional context or documents to be used for generating the response. Default is an empty list.
            v (bool): A flag indicating whether to use the server for generating the response. Default is False.

        Returns:
            str: The generated response.

        Raises:
            Exception: If an error occurs during the execution of the method.
        """
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
