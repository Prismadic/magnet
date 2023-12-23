# Copyright © 2023 Apple Inc.
# docstrings - 2023 Prismadic, LLC.

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from sentencepiece import SentencePieceProcessor
from magnet.utils.globals import _f
from magnet.utils.data_classes import MistralArgs


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        """
        Initializes the attributes of the RMSNorm class.

        Args:
            dims (int): The number of dimensions for the weight attribute.
            eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-5.

        Returns:
            None
        """
        super(RMSNorm, self).__init__()
        self.dims = dims
        self.eps = eps

    def forward(self, x):
        """
        Applies RMS normalization to the input array.

        Args:
            x (torch.Tensor): The input array to be normalized.

        Returns:
            torch.Tensor: The normalized array.
        """
        return x / torch.sqrt(torch.mean(x**2, dim=self.dims, keepdim=True) + self.eps)
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        """
        Apply RMS normalization to the input array `x` and return the normalized output.

        Args:
            x (ndarray): The input array to be normalized.

        Returns:
            ndarray: The normalized output array, which is the result of applying RMS normalization to the input array `x`.
        """
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output


class Attention(nn.Module):
    """
    The `Attention` class is responsible for performing the attention computation in a transformer block.

    Args:
        args (MistralArgs): An instance of `MistralArgs` that contains the arguments for the attention computation.

    Attributes:
        args (MistralArgs): An instance of `MistralArgs` that contains the arguments for the attention computation.
        n_heads (int): The number of attention heads.
        n_kv_heads (int): The number of key-value attention heads.
        repeats (int): The number of times to repeat the key-value attention heads.
        scale (float): The scaling factor for the attention scores.
        wq (nn.Linear): Linear layer for the query projection.
        wk (nn.Linear): Linear layer for the key projection.
        wv (nn.Linear): Linear layer for the value projection.
        wo (nn.Linear): Linear layer for the output projection.
        rope (nn.RoPE): Instance of `nn.RoPE` class for relative positional encoding.
    """

    def __init__(self, args: MistralArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads *
                            args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads *
                            args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.rope = nn.RoPE(args.head_dim, traditional=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Perform attention computation on the input array `x`.

        Args:
            x (mx.array): The input array of shape (batch_size, sequence_length, dimension).
            mask (Optional[mx.array]): An optional mask array of shape (batch_size, sequence_length) to mask certain elements in the input array.
            cache (Optional[Tuple[mx.array, mx.array]]): An optional cache tuple containing two arrays of shape (batch_size, sequence_length, dimension) to store intermediate results.

        Returns:
            mx.array: The final output array of shape (batch_size, sequence_length, dimension).
            Optional[Tuple[mx.array, mx.array]]: The updated cache tuple containing two arrays of shape (batch_size, sequence_length, dimension).
        """
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(
            B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32),
                            axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), (keys, values)


class FeedForward(nn.Module):
    """
    Applies a feed-forward neural network to the input data.

    Args:
        args (MistralArgs): The arguments for the model.

    Example Usage:
        args = MistralArgs(dim=512, hidden_dim=2048)
        feed_forward = FeedForward(args)
        output = feed_forward(input)
    """

    def __init__(self, args: MistralArgs):
        """
        Initializes the FeedForward class.

        Args:
            args (MistralArgs): The arguments for the model.
        """
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        """
        Applies the feed-forward neural network to the input data.

        Args:
            x: The input data.

        Returns:
            mx.array: The output of the feed-forward neural network.
        """
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """
    Initializes the attributes of the TransformerBlock class and creates instances of other required classes.

    Args:
        args (MistralArgs): An instance of the MistralArgs class that contains the arguments for the transformer block.

    Example Usage:
        args = MistralArgs(dim=512, n_heads=8, norm_eps=1e-5)
        block = TransformerBlock(args)

    Flow:
        1. Initialize the n_heads attribute of the TransformerBlock instance with the value from args.n_heads.
        2. Initialize the dim attribute of the TransformerBlock instance with the value from args.dim.
        3. Create an instance of the Attention class and assign it to the attention attribute of the TransformerBlock instance, passing args as an argument.
        4. Create an instance of the FeedForward class and assign it to the feed_forward attribute of the TransformerBlock instance, passing args as an argument.
        5. Create an instance of the RMSNorm class and assign it to the attention_norm attribute of the TransformerBlock instance, passing args.dim and args.norm_eps as arguments.
        6. Create an instance of the RMSNorm class and assign it to the ffn_norm attribute of the TransformerBlock instance, passing args.dim and args.norm_eps as arguments.
        7. Assign the args argument to the args attribute of the TransformerBlock instance.

    Returns:
        None
    """

    def __init__(self, args: MistralArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Apply the TransformerBlock to the input array.

        Args:
            x (mx.array): The input array of shape (batch_size, sequence_length).
            mask (Optional[mx.array]): An optional mask array of shape (batch_size, sequence_length) to mask certain elements in the input array.
            cache (Optional[Tuple[mx.array, mx.array]]): An optional cache tuple containing two arrays of shape (batch_size, sequence_length, hidden_dim) to store intermediate results.

        Returns:
            out (mx.array): The final output array of shape (batch_size, sequence_length, hidden_dim).
            cache (Optional[Tuple[mx.array, mx.array]]): The updated cache tuple containing two arrays of shape (batch_size, sequence_length, hidden_dim).
        """
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache


class Mistral(nn.Module):
    """
    A language model that performs a series of operations on an input array using transformer blocks.

    Args:
        args (MistralArgs): The model arguments that define the dimensions and parameters of the language model.

    Attributes:
        args (MistralArgs): The model arguments that define the dimensions and parameters of the language model.
        vocab_size (int): The size of the vocabulary.
        n_layers (int): The number of transformer blocks in the language model.
        tok_embeddings (nn.Embedding): The token embedding layer.
        layers (List[TransformerBlock]): The list of transformer blocks.
        norm (RMSNorm): The RMS normalization layer.
        output (nn.Linear): The output layer.
    """

    def __init__(self, args: MistralArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args)
                       for _ in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        """
        Perform a series of operations on the input array using the layers defined in the class.

        Args:
            inputs (mx.array): An array representing the input data. It should have shape (batch_size, sequence_length).
            cache (list, optional): The cache value for each layer. Default is None.

        Returns:
            mx.array: The output array after passing through the output layer and applying normalization. It has shape (batch_size, sequence_length, vocab_size).
            list: The updated cache after processing the input array through the layers. It is a list of length equal to the number of layers in the model, where each element is the cache value for the corresponding layer.
        """
        h = self.tok_embeddings(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(
                h.shape[1])
            mask = mask.astype(h.dtype)

        # Rest of the code remains the same
        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.output(self.norm(h)), cache


class Tokenizer:
    """
    Initializes the tokenizer object by loading a SentencePiece model from a given file path and setting the separator character.

    Args:
        model_path (str): The file path of the SentencePiece model.

    Raises:
        AssertionError: If the file specified by `model_path` does not exist.
        AssertionError: If the vocabulary size of the model does not match the number of pieces in the model.
    """

    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = "▁"
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def eos_id(self) -> int:
        """
        Returns the ID of the end-of-sentence token in the tokenizer's model.

        Returns:
            int: The ID of the end-of-sentence token in the tokenizer's model.
        """
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        """
        Returns the ID of the padding token in the tokenizer's model.

        Returns:
            int: The ID of the padding token in the tokenizer's model.
        """
        return self._model.pad_id()

    def encode(self, s: str) -> List[int]:
        return [self._model.bos_id(), *self._model.encode(s)]

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): A list of token IDs to be decoded into a string.

        Returns:
            str: The decoded string corresponding to the input list of token IDs.
        """
        out = self._model.decode(t)
        if t and self._model.id_to_piece(t[0])[0] == self._sep:
            return " " + out
        return out


def load_model(folder: str):
    """
    Load a pre-trained language model and tokenizer from a specified folder.

    Args:
        folder (str): The path to the folder containing the pre-trained model.

    Returns:
        model (Mistral): The loaded pre-trained language model.
        tokenizer (Tokenizer): The initialized tokenizer.
    """
    model_path = Path(folder)
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        config.pop("sliding_window", None)
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)
        model_args = MistralArgs(**config)
    weights = mx.load(str(model_path / "weights.npz"))
    weights = tree_unflatten(list(weights.items()))
    model = Mistral(model_args)
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)
    model.update(weights)
    mx.eval(model.parameters())
    return model, tokenizer


def infer(prompt: mx.array, model: Mistral, temp: Optional[float] = 0.0):
    """
    Generates a sequence of tokens using a pre-trained language model.

    Args:
        prompt (mx.array): An mxnet array representing the initial prompt for generating the sequence.
        model (Mistral): An instance of the Mistral class, which is a pre-trained language model.
        temp (float, optional): A float representing the temperature parameter for controlling the randomness of the generated sequence. Defaults to 0.0.

    Yields:
        mx.array: Generated tokens, one by one.

    Example:
        prompt = mx.array(tokenizer.encode("The cat"))
        model = Mistral(args)
        temp = 0.8
        for token in infer(prompt, model, temp):
            print(token)

    """
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))

    logits, cache = model(prompt[None])
    y = sample(logits[:, -1, :])
    yield y

    while True:
        logits, cache = model(y[:, None], cache)
        y = sample(logits.squeeze(1))
        yield y


def generate(payload):
    """
    Generate a sequence of tokens using a pre-trained language model.

    Args:
        payload (dict): A dictionary containing the following keys:
            - 'seed' (int): The random seed for reproducibility.
            - 'model_path' (str): The path to the pre-trained model.
            - 'prompt' (str): The initial prompt for generating the sequence.
            - 'temp' (float): The temperature parameter for controlling the randomness of the generated sequence.
            - 'max_tokens' (int): The maximum number of tokens to generate.

    Returns:
        str: The generated sequence of tokens decoded into a string.
    """
    mx.random.seed(payload['seed'])
    _f('wait', f"loading model from {payload['model_path']}")
    model, tokenizer = load_model(payload['model_path'])
    if model:
        _f('success', f"loaded {payload['model_path']}")
    tic = time.time()
    prompt = mx.array(tokenizer.encode(payload['prompt']))
    tokens = []
    for token, ntoks in zip(infer(prompt, model, payload['temp']), range(payload['max_tokens'])):
        tokens.append(token)
        if ntoks == 0:
            toc = time.time()
            mx.eval(tokens)
            prompt_tps = prompt.size / (toc - tic)
            tic = time.time()

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    return s
