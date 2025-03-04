import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import copy
import numpy as np
import os


BASELINE_MODEL_NUMBER_OF_LAYERS = 6
BASELINE_MODEL_DIMENSION = 512
BASELINE_MODEL_NUMBER_OF_HEADS = 8
BASELINE_MODEL_DROPOUT_PROB = 0.1
BASELINE_MODEL_LABEL_SMOOTHING_VALUE = 0.1


BIG_MODEL_NUMBER_OF_LAYERS = 6
BIG_MODEL_DIMENSION = 1024
BIG_MODEL_NUMBER_OF_HEADS = 16
BIG_MODEL_DROPOUT_PROB = 0.3
BIG_MODEL_LABEL_SMOOTHING_VALUE = 0.1


CHECKPOINTS_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "models", "checkpoints"
)
BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "models", "binaries")
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "data")
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(DATA_DIR_PATH, exist_ok=True)


BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


# ---- Transformer Modules ----
class Transformer(nn.Module):
    """
    A complete Transformer model as described in "Attention is All You Need" (Vaswani et al., 2017).

    This implementation combines the encoder and decoder stacks with embedding layers and
    output generation. The architecture follows the original paper design with optional
    hyperparameter customization.

    Args:
        src_vocab_size (int): Size of the source vocabulary
        tgt_vocab_size (int): Size of the target vocabulary
        model_dimension (int): The embedding and model dimension (d_model in the paper)
        num_heads (int): Number of attention heads
        num_encoder_layers (int): Number of encoder layers in the encoder stack
        num_decoder_layers (int): Number of decoder layers in the decoder stack
        d_ff (int): Dimension of the feed-forward networks
        dropout (float, optional): Dropout probability. Default: 0.1
        max_seq_length (int, optional): Maximum sequence length for positional encoding. Default: 5000
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        model_dimension,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        d_ff,
        dropout=0.1,
        max_seq_length=5000,
    ):
        super().__init__()

        # Input embeddings
        self.src_embedding = Embedding(model_dimension, src_vocab_size)
        self.tgt_embedding = Embedding(model_dimension, tgt_vocab_size)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            model_dimension, max_seq_length, dropout
        )

        # Create encoder
        encoder_layer = EncoderLayer(model_dimension, num_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, num_encoder_layers)

        # Create decoder
        decoder_layer = DecoderLayer(model_dimension, num_heads, d_ff, dropout)
        self.decoder = Decoder(decoder_layer, num_decoder_layers)

        # Output generation
        self.generator = DecoderGenerator(model_dimension, tgt_vocab_size)

        # Initialize parameters using Xavier/Glorot initialization
        self._init_parameters()

        # Model configuration (useful for debugging and visualization)
        self.model_dimension = model_dimension
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

    def _init_parameters(self):
        """
        Initialize parameters using Xavier/Glorot uniform initialization for better convergence.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        """
        Encode the source sequence.

        Args:
            src (Tensor): Source token indices [batch_size, src_seq_len]
            src_mask (Tensor, optional): Mask to avoid attending to padding tokens
                                        [batch_size, 1, src_seq_len]

        Returns:
            Tensor: Encoded representation of source [batch_size, src_seq_len, model_dimension]
        """
        # Convert token indices to embeddings and add positional encoding
        src_embeddings = self.src_embedding(src)
        src_embeddings = self.positional_encoding(src_embeddings)

        # Pass through encoder stack
        return self.encoder(src_embeddings, src_mask)

    def decode(self, tgt, memory, tgt_mask=None, src_mask=None):
        """
        Decode the target sequence given the encoder memory.

        Args:
            tgt (Tensor): Target token indices [batch_size, tgt_seq_len]
            memory (Tensor): Output from the encoder [batch_size, src_seq_len, model_dimension]
            tgt_mask (Tensor, optional): Mask for autoregressive property and padding
                                        [batch_size, tgt_seq_len, tgt_seq_len]
            src_mask (Tensor, optional): Mask for source padding [batch_size, 1, src_seq_len]

        Returns:
            Tensor: Decoded representation [batch_size, tgt_seq_len, model_dimension]
        """
        # Convert token indices to embeddings and add positional encoding
        tgt_embeddings = self.tgt_embedding(tgt)
        tgt_embeddings = self.positional_encoding(tgt_embeddings)

        # Pass through decoder stack
        return self.decoder(tgt_embeddings, memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass through the full Transformer.

        Args:
            src (Tensor): Source token indices [batch_size, src_seq_len]
            tgt (Tensor): Target token indices [batch_size, tgt_seq_len]
            src_mask (Tensor, optional): Mask for source padding [batch_size, 1, src_seq_len]
            tgt_mask (Tensor, optional): Mask for target padding and autoregressive property
                                        [batch_size, tgt_seq_len, tgt_seq_len]

        Returns:
            Tensor: Output probabilities over target vocabulary
                   [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # Encode source sequence
        encoder_output = self.encode(src, src_mask)

        # Decode target sequence
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)

        # Generate output probabilities
        return self.generator(decoder_output)

    def greedy_decode(self, src, src_mask=None, max_len=100, start_symbol=None):
        """
        Greedy decoding for inference.

        Args:
            src (Tensor): Source token indices [batch_size, src_seq_len]
            src_mask (Tensor, optional): Mask for source padding [batch_size, 1, src_seq_len]
            max_len (int, optional): Maximum decoding length. Default: 100
            start_symbol (int, optional): Start symbol index. Default: None

        Returns:
            Tensor: Generated output sequence [batch_size, out_seq_len]
        """
        if start_symbol is None:
            # Default to using the BOS token index as the start symbol
            # This assumes the BOS token is the first special token in the vocabulary
            start_symbol = 0  # This should be replaced with the actual BOS token index

        batch_size = src.size(0)
        device = src.device

        # Encode the source sequence
        memory = self.encode(src, src_mask)

        # Initialize the decoder input with the start symbol
        ys = torch.ones(batch_size, 1).fill_(start_symbol).long().to(device)

        # Iteratively generate each token
        for i in range(max_len - 1):
            # Create a mask for previously generated tokens
            tgt_mask = self._generate_square_subsequent_mask(ys.size(1)).to(device)

            # Decode next token
            out = self.decode(ys, memory, tgt_mask, src_mask)

            # Get probabilities and select the most likely token
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            # Append to the sequence
            next_word = next_word.unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)

        return ys

    def _generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence. The mask ensures that the
        predictions for position i can depend only on the known outputs at
        positions less than i.

        Args:
            sz (int): Size of the square mask

        Returns:
            Tensor: Mask with shape [sz, sz]
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    @staticmethod
    def create_padding_mask(seq, pad_idx):
        """
        Create a mask to hide padding tokens.

        Args:
            seq (Tensor): Sequence of token indices [batch_size, seq_len]
            pad_idx (int): Padding token index

        Returns:
            Tensor: Mask with shape [batch_size, 1, seq_len]
        """
        mask = (seq == pad_idx).unsqueeze(1)
        return mask

    @classmethod
    def build_base_transformer(cls, src_vocab_size, tgt_vocab_size):
        """
        Build a transformer with the base configuration from the paper.

        Args:
            src_vocab_size (int): Size of the source vocabulary
            tgt_vocab_size (int): Size of the target vocabulary

        Returns:
            Transformer: A transformer with the base configuration
        """
        return cls(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            model_dimension=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=2048,
            dropout=0.1,
        )

    @classmethod
    def build_big_transformer(cls, src_vocab_size, tgt_vocab_size):
        """
        Build a transformer with the big configuration from the paper.

        Args:
            src_vocab_size (int): Size of the source vocabulary
            tgt_vocab_size (int): Size of the target vocabulary

        Returns:
            Transformer: A transformer with the big configuration
        """
        return cls(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            model_dimension=BIG_MODEL_DIMENSION,
            num_heads=BIG_MODEL_NUMBER_OF_HEADS,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=4096,
            dropout=BIG_MODEL_DROPOUT_PROB,
        )


# ---- Encoder Modules ----


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()

        assert isinstance(
            encoder_layer, EncoderLayer
        ), f"encoder_layer must be an instance of EncoderLayer, got {type(encoder_layer)}"
        assert (
            isinstance(num_layers, int) and num_layers > 0
        ), f"num_layers must be a positive integer, got {num_layers}"

        self.encoder_layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.model_dimension)

    def forward(self, src_representations_batch, src_mask=None):
        for layer in self.encoder_layers:
            # masks are used to prevent the model from attending to future tokens by
            # masking/ignoring padded tokens in the multi-head attention module
            src_representations_batch = layer(src_representations_batch, src_mask)

        # apply layer normalization to the final output, different from the paper
        # consequence of using LayerNorm before instead of after the sublayers
        return self.norm(src_representations_batch)


# ---- Encoder Layer ----
class EncoderLayer(nn.Module):
    """
    This module is a single encoder layer of the transformer.
    It consists of a self-attention mechanism, a feed-forward network, and (two) residual connections.
    Two residual connections are used in the paper "Attention is All You Need" by Vaswani et al.
    The flow of the forward pass is:
    1. Apply self-attention to the input representations.
    2. Apply the sublayer norm dropout to the output of the self-attention layer (with a residual connection)
    3. Apply a feed-forward network to the output of the sublayer norm dropout.
    4. Apply the sublayer norm dropout to the output of the feed-forward network (with a residual connection)
    5. Return the result.

    Flow Diagram:
        Input batch representations
                    │
                    ▼
          [Self-Attention]
                    │
                    ▼
          [Sublayer Norm Dropout]
                    │
                    ▼
          [Residual Addition: Input + 1st Sublayer output]
                    │
                    ▼
          [Feed-Forward]
                    │
                    ▼
          [Sublayer Norm Dropout]
                    │
                    ▼
          [Residual Addition: 1st Sublayer output + 2nd Sublayer output]
                    │
                    ▼
        Output batch representations
    """

    def __init__(self, model_dimension, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(model_dimension, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(model_dimension, d_ff, dropout)
        self.sublayers = nn.ModuleList(
            [SublayerNormDropout(model_dimension, dropout) for _ in range(2)]
        )
        self.model_dimension = model_dimension

    def forward(self, src_representations_batch, src_mask=None):
        # Apply self-attention with residual connection and layer norm
        # lambda function is used to pass the input to the self-attention layer
        # this is done to avoid having to pass the input to the self-attention layer
        # as an argument to the sublayer norm dropout
        src_representations_batch = self.sublayers[0](
            src_representations_batch,
            lambda x: self.self_attn(
                query=x,
                key=x,
                value=x,
                mask=src_mask,
            ),
        )

        # Apply feed-forward with residual connection and layer norm
        src_representations_batch = self.sublayers[1](
            src_representations_batch, self.feed_forward
        )

        return src_representations_batch


# ---- Decoder Modules ----


class DecoderLayer(nn.Module):
    """
    This module is a single decoder layer of the transformer.
    It consists of three sublayers as described in the paper "Attention is All You Need":
      1. Masked self-attention on the target representations.
      2. Encoder-decoder (cross) attention where the target attends to the encoder output.
      3. A position-wise feed-forward network.

    Each sublayer is augmented with layer normalization, dropout, and a residual connection.

    The forward pass flows as follows:
      1. Apply masked self-attention to the target representations.
      2. Apply sublayer normalization and dropout, then add the original target representations (residual connection).
      3. Apply encoder-decoder (cross) attention using the encoder's output.
      4. Apply sublayer normalization and dropout, then add the previous output (residual connection).
      5. Apply the feed-forward network.
      6. Apply sublayer normalization and dropout, then add the previous output (residual connection).

    Flow Diagram:
         Target representations
                    │
                    ▼
         [Masked Self-Attention]
                    │
                    ▼
       [Sublayer Norm, Dropout, Residual]
                    │
                    ▼
      [Encoder-Decoder (Cross) Attention]
                    │
                    ▼
       [Sublayer Norm, Dropout, Residual]
                    │
                    ▼
         [Feed-Forward Network]
                    │
                    ▼
       [Sublayer Norm, Dropout, Residual]
                    │
                    ▼
         Output representations
    """

    def __init__(self, model_dimension, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 3 sublayers used in the paper "Attention is All You Need" by Vaswani et al.
        self.sublayers = nn.ModuleList(
            [
                SublayerNormDropout(model_dimension, dropout),
                SublayerNormDropout(model_dimension, dropout),
                SublayerNormDropout(model_dimension, dropout),
            ]
        )
        self.self_attn = MultiHeadAttention(model_dimension, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(model_dimension, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(model_dimension, d_ff, dropout)
        self.model_dimension = model_dimension

    def forward(
        self,
        target_representations_batch,
        src_representations_batch,
        src_mask=None,
        target_mask=None,
    ):

        # ---- Define the lambda functions for the attention layers ----

        # Alias src_representations_batch as srb for a concise reference.
        # Here, srb and other variables like target_mask and src_mask are "cached" —
        # meaning they are captured by the lambda functions' closures when these
        # functions are defined. This allows the lambdas to access these variables later
        # without needing them as explicit parameters.
        srb = src_representations_batch

        # Define a lambda for masked self-attention.
        # This lambda takes only the target representations batch (trb) as input.
        # Other necessary inputs (self.self_attn and target_mask) are captured from the outer scope.
        decoder_target_self_attn = lambda trb: self.self_attn(
            query=trb, key=trb, value=trb, mask=target_mask
        )

        # Define a lambda for cross-attention using the encoder output.
        # This lambda also only accepts trb as input, while srb (the cached encoder output)
        # and src_mask are captured from the outer context.
        # Caching here ensures that these variables are maintained with their defined values
        # when the lambda is executed.
        decoder_target_cross_attn = lambda trb: self.cross_attn(
            query=trb, key=srb, value=srb, mask=src_mask
        )

        # ---- Start the forward pass, starting from the first attention layer ----
        print("Starting forward pass for decoder layer")

        print("Shape of target mask:", target_mask.shape)

        # Apply the first self-attention layer, followed by the first sublayer norm dropout + residual connection
        target_representations_batch = self.sublayers[0](
            target_representations_batch, decoder_target_self_attn
        )

        print("Completed first sublayer")

        # Apply the second self-attention layer, followed by the sublayer similarly.
        # Here the lambda function corresponds to the encoder-decoder cross-attention.
        target_representations_batch = self.sublayers[1](
            target_representations_batch, decoder_target_cross_attn
        )

        print("Completed second sublayer")

        # Apply the feed-forward network, followed by the third sublayer norm dropout + residual connection
        target_representations_batch = self.sublayers[2](
            target_representations_batch, self.feed_forward
        )

        print("Completed third sublayer")

        # Return the final output
        return target_representations_batch


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.decoder_layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(decoder_layer.model_dimension)

    def forward(
        self,
        target_representations_batch,
        src_representations_batch,
        target_mask=None,
        src_mask=None,
    ):

        # forward pass through the decoder layers
        for layer in self.decoder_layers:
            # target_mask is used to prevent the model from attending to future tokens by
            # masking/ignoring padded tokens in the multi-head attention module
            target_representations_batch = layer(
                target_representations_batch,
                src_representations_batch,
                target_mask,
                src_mask,
            )

        # apply layer normalization to the final output, different from the paper
        # consequence of using LayerNorm before instead of after the sublayers
        return self.norm(target_representations_batch)


# ---- Helper Modules ----


class SublayerNormDropout(nn.Module):
    """
    This module performs a residual connection between the input and output of a sublayer, and a dropout layer.
    The following steps are performed:
      1. Normalize the input representations.
      2. Process the normalized inputs through a sublayer (nn.Module).
      3. Apply dropout to the sublayer's output.
      4. Add the original input back (residual connection) to produce the final output.

    Flow Diagram:
         Input batch representations
                    │
                    ▼
          [Layer Normalization]
                    │
                    ▼
           [Sublayer (nn.Module)]
                    │
                    ▼
               [Dropout Layer]
                    │
                    ▼
         [Residual Addition: Input + Processed Output]
                    │
                    ▼
         Output batch representations

    Args:
        model_dimension (int): The dimension of the model.
        dropout_prob (float): The probability of dropout.
    """

    def __init__(self, model_dimension, dropout_prob):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, batch_representations, sublayer):
        # Residual connection between input and output of the sublayer
        # This achieves a regularization effect
        return (
            self.dropout(sublayer(self.norm(batch_representations)))
            + batch_representations
        )


class DecoderGenerator(nn.Module):
    """
    Projects decoder hidden states to log probability distributions over the vocabulary.

    This module performs a linear projection from the model's hidden dimension to the vocabulary size,
    followed by applying a log softmax over the vocabulary dimension. Such a transformation converts
    the raw output logits into log probabilities that are suitable for criteria like nn.KLDivLoss.

    Attributes:
        linear (nn.Linear): A linear layer mapping from model_dimension to vocab_size.
        log_softmax (nn.LogSoftmax): Log softmax activation applied over the last dimension.
    """

    def __init__(self, model_dimension, vocab_size):
        super().__init__()
        self.linear = nn.Linear(model_dimension, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, decoder_hidden_states):
        """
        Transforms decoder hidden states into log probabilities over the vocabulary.

        Args:
            decoder_hidden_states (torch.Tensor): A tensor of shape
                (batch_size, sequence_length, model_dimension) representing the decoder outputs.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, vocab_size) where each
            element corresponds to a log probability over the vocabulary.
        """
        return self.log_softmax(self.linear(decoder_hidden_states))


class PositionwiseFeedForward(nn.Module):
    """
    Implements a position-wise feed-forward network (FFN) with ReLU activation.

    This module performs a two-layer linear transformation with ReLU activation in between.
    It maps an input tensor of shape (batch_size, sequence_length, model_dimension) to a tensor of shape
    (batch_size, sequence_length, model_dimension).
    """

    def __init__(self, model_dimension, ff_hidden_dimension, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(model_dimension, ff_hidden_dimension)
        self.linear2 = nn.Linear(ff_hidden_dimension, model_dimension)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, batch_representations):
        # Implement FFN with ReLU activation
        return self.linear2(
            self.dropout(self.relu(self.linear1(batch_representations)))
        )


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # Get dimensions
        d_k = query.size(-1)

        # Compute scaled dot product attention
        # matmul: (batch_size, num_heads, seq_len, d_k) x (batch_size, num_heads, d_k, seq_len)
        # result: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        print("scores shape:", scores.shape)

        # Apply mask if provided - set masked positions to -inf before softmax
        # mask shape = (B, 1, 1, S) or (B, 1, T, T) will get broad-casted (copied) as needed to match scores shape
        print("mask shape:", mask.shape)
        if mask is not None:
            scores = scores.masked_fill_(mask == torch.tensor(False), -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply dropout to attention weights (regularization)
        attention_weights = self.dropout(attention_weights)

        # Compute weighted sum of values using attention weights
        # matmul: (batch_size, num_heads, seq_len, seq_len) x (batch_size, num_heads, seq_len, d_k)
        # result: (batch_size, num_heads, seq_len, d_k)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Save parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension of each head

        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # Output projection
        self.output_linear = nn.Linear(d_model, d_model)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Save model dimension for EncoderLayer's access
        self.model_dimension = d_model

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Perform linear projections and split into multiple heads
        # Original: (batch_size, seq_len, d_model)
        # After projection: (batch_size, seq_len, d_model)
        # After reshape: (batch_size, seq_len, num_heads, d_k)
        # After transpose: (batch_size, num_heads, seq_len, d_k)
        q = (
            self.q_linear(query)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(key)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(value)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # # Adjust mask for multiple heads if provided
        # if mask is not None:
        #     # Expand mask to have the same batch size and number of heads
        #     # (batch_size, 1, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        #     mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # Apply scaled dot-product attention
        attn_output, attention_weights = self.attention(q, k, v, mask)

        # Reshape back to original dimensions
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k)
        attn_output = attn_output.transpose(1, 2)

        # (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        attn_output = attn_output.contiguous().view(batch_size, -1, self.d_model)

        # Apply final projection
        output = self.output_linear(attn_output)

        return output


# ---- Input Embeddings ----


class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Embeds the input tokens into a dense vector space.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing the input tokens.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, d_model) containing the embedded tokens.
        """
        assert (
            x.dim() == 2
        ), f"x must be a 2D tensor, got {x.dim()}D"  # Embed the input tokens

        embedded = self.embedding(x)

        # Scale the embedded tokens by the square root of the model dimension
        return embedded * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for the input embeddings.

    This module generates a positional encoding matrix that is added to the input embeddings
    to provide information about the position of the tokens in the sequence.

    The positional encoding is a sine and cosine function that is applied to the input embeddings.
    The sine and cosine functions are applied to the even and odd indices of the input embeddings, respectively.

    Args:
        model_dimension (int): The dimension of the model.
        max_seq_length (int): The maximum sequence length.
        dropout (float): The dropout rate.
    """

    def __init__(self, model_dimension, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, model_dimension)
        position = torch.arange(0, max_seq_length).unsqueeze(1)

        # heuristic division term for sinusoidal encoding
        frequencies = torch.pow(
            10000,
            -torch.arange(0, model_dimension, 2, dtype=torch.float32) / model_dimension,
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * frequencies)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * frequencies)

        # Add batch dimension and register as buffer
        self.register_buffer("pe", pe)

    def forward(self, batch_embeddings):

        # check for valid input dimensions
        assert (
            batch_embeddings.ndim == 3
            and batch_embeddings.shape[-1] == self.pe.shape[1]
        ), f"Expected (batch size, max token sequence length, model dimension) got {batch_embeddings.shape}"

        # Batch embeddings has shape (batch_size, max_seq_length, model_dimension)
        # where max_seq_length is the maximum sequence length of the batch (max out of source or target sequences)
        # the positional encoding will then have shape (max_seq_length, model_dimension),
        # which is broadcastable to the batch embeddings (batch_size, max_seq_length, model_dimension)
        positional_encoding = self.pe[: batch_embeddings.size(1)]

        print("positional_encoding shape:", positional_encoding.shape)
        print("batch_embeddings shape:", batch_embeddings.shape)

        # Add positional encoding to input embeddings with broadcasting from (max_seq_length, model_dimension) to (batch_size, max_seq_length, model_dimension)
        batch_embeddings = batch_embeddings + positional_encoding

        # Apply dropout to the input embeddings
        return self.dropout(batch_embeddings)
