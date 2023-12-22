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
    result: str
    model: str

@dataclass
class EmbeddingPayload:
    document: str
    embedding: list
    text: list
    model: str