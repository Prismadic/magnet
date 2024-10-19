from dataclasses import dataclass, field
from typing import Optional, List
import math
from torch import Tensor

@dataclass
class ModelPrediction:
    """
    Data class to store the results of a model prediction and subsequent clustering.

    Attributes:
        input_data (Optional[torch.Tensor]): The input data used for the model prediction.
        embeddings (Optional[torch.Tensor]): The embeddings generated by the model.
        predictions (Optional[torch.Tensor]): The predictions output by the model.
        umap_embeddings (Optional[List[float]]): The UMAP-reduced embeddings.
        cluster_labels (Optional[List[int]]): The cluster labels assigned by the clustering algorithm.
        umap_model_path (Optional[str]): The file path where the UMAP model was saved.
        clusterer_model_path (Optional[str]): The file path where the clusterer model was saved.
    """
    input_data: Optional[Tensor] = None
    embeddings: Optional[Tensor] = None
    predictions: Optional[Tensor] = None
    umap_embeddings: Optional[List[float]] = None
    cluster_labels: Optional[List[int]] = None
    umap_model_path: Optional[str] = None
    clusterer_model_path: Optional[str] = None

@dataclass
class CMambaArgs:
    """
    Data class for CMamba model arguments.

    Args:
        d_model (int): Dimension of the model.
        n_layer (int): Number of C-Mamba blocks.
        seq_len (int): Length of input sequence (look-back window).
        d_state (int): Dimension of SSM state.
        expand (int): Expansion factor for inner dimension.
        dt_rank (Optional[str]): Rank for delta projection, 'auto' sets it to d_model/16.
        d_conv (int): Kernel size for temporal convolution.
        pad_multiple (int): Padding to ensure sequence length is divisible by this.
        conv_bias (bool): Whether to use bias in convolution.
        bias (bool): Whether to use bias in linear layers.
        num_channels (int): Number of numerical channels in your data.
        patch_len (int): Length of each patch.
        stride (int): Stride for patching.
        forecast_len (int): Number of future time steps to predict.
        sigma (float): Standard deviation for channel mixup.
        reduction_ratio (int): Reduction ratio for channel attention.
        verbose (bool): Verbose mode for detailed output.
    """
    num_epochs: int = 5
    learning_rate: float = 0.0005
    d_model: int = 128
    n_layer: int = 4
    seq_len: int = 500
    d_state: int = 16
    expand: int = 2
    dt_rank: Optional[str] = 'auto'
    d_conv: int = 4
    pad_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    batch_size: int = 32
    num_channels: int = 3 # example
    patch_len: int = 16
    stride: int = 8
    forecast_len: int = 500
    input_length: int = 500
    sigma: float = 0.5
    reduction_ratio: int = 8
    verbose: bool = False

    d_inner: int = field(init=False)
    num_patches: int = field(init=False)

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.forecast_len % self.pad_multiple != 0:
            self.forecast_len += (self.pad_multiple - self.forecast_len % self.pad_multiple)
        
        self.num_patches = (self.seq_len - self.patch_len) // self.stride + 1

@dataclass
class DataProcessingConfig:
    chunk_size: int = 10000
    timestamp_column: str = 'timestampRecorded'
    features_to_match: Optional[List[str]] = None
    input_length: int = 500

@dataclass
class DataTrainingConfig:
    pass