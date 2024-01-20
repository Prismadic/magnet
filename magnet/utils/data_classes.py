from dataclasses import dataclass

@dataclass
class Payload:
    """
    Represents a payload with two main fields: text and document.

    Args:
        text (str): The text associated with the payload.
        document (str): The document associated with the payload.
    """
    text: str
    document: str

@dataclass
class GeneratedPayload:
    """
    Represents a payload generated by a system.

    Args:
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
    Represents a payload for embedding text data.

    Attributes:
        document (str): The document associated with the text data.
        embedding (list): The embedding of the text data.
        text (list): The text of the data.
        model (str): The model used for embedding the text data.
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

@dataclass
class JobParams:
    milvus_host: str
    milvus_port: int
    milvus_username: str
    milvus_password: str
    nats_host: str
    nats_username: str
    nats_password: str
    job_type: str
    job_n: int
    embedding_model: str
    generation_model: str
