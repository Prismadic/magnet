import requests

class InferenceAPI:
    def __init__(self, token, provider):
        self.token = token
        self.provider = provider

    def invoke(self, payload):
        if self.provider == 'openai':
            return self._invoke_openai(payload)
        elif self.provider == 'huggingface':
            return self._invoke_huggingface(payload)

    def _invoke_openai(self, payload):
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

        # Default to the completions endpoint
        endpoint = "https://api.openai.com/v1/completions"

        # Adjust for chat models
        if payload['model'].startswith('gpt-3.5-turbo'):
            endpoint = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": payload['model'],
                "messages": [{"role": "user", "content": payload['prompt']}]
            }
        else:
            payload = {
                "model": payload['model'],
                "prompt": payload['prompt'],
                "max_tokens": payload.get("max_tokens", 1024)
            }

        response = requests.post(endpoint, headers=headers, json=payload)
        return response.json()['choices'][0]['message']['content']

    def _invoke_huggingface(self, payload):
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.post(f"https://api-inference.huggingface.co/models/{payload['model']}", headers=headers, json={"inputs": payload['prompt']})
        return response.json()[0]['generated_text'].split(payload['prompt'])[1]
