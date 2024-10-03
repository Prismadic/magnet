from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from dataclasses import dataclass, field
from typing import List, Optional, Callable
from datetime import datetime
from numpy import ndarray

@dataclass
class Status:
    timestamp: datetime
    type: str  # e.g., 'success', 'warn', 'fatal', 'info', etc.
    content: str

@dataclass
class ProcessParams:
    resource_id: str
    location: str
    model: str
    data_source: str
    processing_options: Dict[str, Any]  # E.g., {'filter': True, 'normalize': False}

@dataclass
class InferenceParams:
    resource_id: str
    location: str
    data_source: str
    model_id: str
    inference_options: Dict[str, Any]  # E.g., {'batch_size': 32, 'confidence_threshold': 0.5}

@dataclass
class TrainParams:
    resource_id: str
    location: str
    data_source: str
    model: str
    training_options: Dict[str, Any]  # E.g., {'early_stopping': True, 'augmentation': True}

@dataclass
class AcquireParams:
    resource_id: str
    data_source: str
    location: str
    acquisition_options: Dict[str, Any]  # E.g., {'timeout': 120, 'retry_on_failure': True}

# The main Job class remains as you defined it:
@dataclass
class Job:
    params: AcquireParams | TrainParams | InferenceParams | ProcessParams  # This will be one of the above parameter classes
    _type: str
    _id: str
    _isClaimed: bool = False

@dataclass
class Run:
    _id: str
    _job: Job
    _type: str
    start_time: str
    status: Optional[str] = None
    end_time: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None

@dataclass
class AskParameters:
    m: str = "mistralai/Mistral-7B-Instruct-v0.1"
    q: str = "What is your itinerary?"
    t: float = 1.0
    n: int = 8096
    p: str = "qa_ref"
    cb: Optional[Callable] = None
    docs: List[str] = field(default_factory=list)
    context: str = ""
    vllm: bool = False

@dataclass
class IndexConfig:
    milvus_uri: Optional[str] = None
    milvus_port: Optional[int] = None
    milvus_user: Optional[str] = None
    milvus_password: Optional[str] = None
    dimension: Optional[int] = None
    model: Optional[str] = None
    name: Optional[str] = None
    options: Dict[Optional[dict], Any] = field(default_factory=dict)

@dataclass
class MagnetConfig:
    host: str
    domain: str = None
    credentials: str = None
    session: str = None
    stream_name: str = None
    category: str = None
    kv_name: str = None
    os_name: str = None
    index: Optional[IndexConfig] = None

@dataclass
class Payload:
    """
    Represents a payload with two main fields: text and document.

    Args:
        content (object | list | str | dict): The information associated with the payload.
        _id (str): The id associated with the payload.
    """
    content: object | list | str | dict | ndarray
    _id: str

@dataclass
class FilePayload:
    """
    Represents a payload with two main fields: data and _id.

    Args:
        data (list): The bytearray associated with the payload.
        _id (str): The document associated with the payload.
    """
    data: bytes
    original_filename: str
    _id: str

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
    content: list
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
    milvus_collection: str
    nats_host: str
    nats_username: str
    nats_password: str
    nats_stream: str
    nats_category: str
    job_type: str
    job_n: int
    embedding_model: str
    generation_model: str