from dataclasses import dataclass
from numpy import array

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

@dataclass
class EmbeddingPayload:
    document: str
    embedding: list
    text: list
    model: str