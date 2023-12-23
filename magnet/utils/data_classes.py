from dataclasses import dataclass

@dataclass
class Payload:
    """
    Represents a payload with two fields: `text` and `document`.

    Example Usage:
    payload = Payload(text="Hello", document="example.txt")
    print(payload.text)  # Output: Hello
    print(payload.document)  # Output: example.txt
    """

    text: str
    document: str

@dataclass
class GeneratedPayload:
    """
    Represents a payload generated by a system.

    Attributes:
        query (str): The query associated with the payload.
        prompt (str): The prompt associated with the payload.
        context (list): The context associated with the payload.
        result (str): The result generated by the system.
        model (str): The model used to generate the payload.
    """

    query: str
    prompt: str
    context: list
    result: str
    model: str

@dataclass
class EmbeddingPayload:
    """
    Represents a payload for storing document embeddings.

    Fields:
    - document: A string field that stores the document text.
    - embedding: A list field that stores the embedding vector.
    - text: A list field that stores the original text.
    - model: A string field that stores the model used for generating the embedding.
    """

    document: str
    embedding: list
    text: list
    model: str

@dataclass
class MistralArgs:
    """
    Represents a set of arguments for the Mistral model.

    Args:
        dim (int): The dimensionality of the model.
        n_layers (int): The number of layers in the model.
        head_dim (int): The dimensionality of each attention head.
        hidden_dim (int): The dimensionality of the hidden layer in the feed-forward network.
        n_heads (int): The number of attention heads.
        n_kv_heads (int): The number of attention heads used for key-value attention.
        norm_eps (float): The epsilon value used for numerical stability in layer normalization.
        vocab_size (int): The size of the vocabulary used in the model.
    """

    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int