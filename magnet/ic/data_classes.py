from dataclasses import dataclass

@dataclass
class Payload:
    text: str
    document: str

@dataclass
class GeneratedPayload:
    query: str
    prompt: str
    context: list
    model: str