#!/usr/bin/env python3
"""
Transformer Usage Example

This script demonstrates how to use the Transformer model for a simple translation task.
It shows how to:
1. Create a Transformer model
2. Prepare data for the model
3. Train the model (simplified example)
4. Perform inference with the trained model

For a real-world application, you would use more sophisticated data processing
and training procedures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from transformer import (
    Transformer,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN
)


def create_sample_data(batch_size=64, src_seq_len=10, tgt_seq_len=12, src_vocab_size=1000, tgt_vocab_size=1000):
    """
    Create random sample data for demonstration.
    In a real scenario, you would load and preprocess actual text data.
    
    Returns:
        tuple: (src_data, tgt_data) batches of token indices
    """
    # Create random source and target sequences
    src_data = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # Add EOS tokens (assume token index 2)
    eos_idx = 2
    for i in range(batch_size):
        # Randomly set the EOS position
        src_eos_pos = np.random.randint(1, src_seq_len)
        tgt_eos_pos = np.random.randint(1, tgt_seq_len)
        
        # Set EOS tokens
        src_data[i, src_eos_pos:] = eos_idx
        tgt_data[i, tgt_eos_pos:] = eos_idx
    
    return src_data, tgt_data


def create_masks(src, tgt, pad_idx=0):
    """
    Create masks for transformer training.
    
    Args:
        src (Tensor): Source token indices [batch_size, src_seq_len]
        tgt (Tensor): Target token indices [batch_size, tgt_seq_len]
        pad_idx (int): Padding token index
        
    Returns:
        tuple: (src_mask, tgt_mask) for transformer model
    """
    # Source padding mask
    src_mask = Transformer.create_padding_mask(src, pad_idx)
    
    # Target padding mask
    tgt_pad_mask = Transformer.create_padding_mask(tgt, pad_idx)
    
    # Target subsequent mask (to prevent attending to future tokens)
    sz = tgt.size(1)
    tgt_subseq_mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    tgt_subseq_mask = tgt_subseq_mask.unsqueeze(0).expand(tgt.size(0), -1, -1)
    
    # Combine padding and subsequent mask
    tgt_mask = tgt_pad_mask | tgt_subseq_mask
    
    return src_mask, tgt_mask


class LabelSmoothing(nn.Module):
    """
    Implement label smoothing for better model training.
    """
    def __init__(self, size, padding_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))  # -2 for PAD and correct
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


def train_example():
    """Example of training a Transformer model."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define model parameters
    src_vocab_size = 1000
    tgt_vocab_size = 1200
    model_dimension = 128
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    d_ff = 512
    dropout = 0.1
    
    # Create model
    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        model_dimension,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        d_ff,
        dropout
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # Create a learning rate scheduler with warmup
    def lr_lambda(step):
        warmup_steps = 4000
        d_model = model_dimension
        
        # Linear warmup followed by inverse square root decay
        step = max(1, step)  # Avoid division by zero
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        else:
            return math.sqrt(warmup_steps / step)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create label smoothing loss
    pad_idx = 0
    criterion = LabelSmoothing(size=tgt_vocab_size, padding_idx=pad_idx, smoothing=0.1)
    
    # Training loop parameters
    batch_size = 64
    epochs = 3
    
    # Example training loop (simplified)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        # In a real scenario, you would iterate over batches from a DataLoader
        # Here we use random data for simplicity
        for batch_idx in range(10):  # Simulating 10 batches per epoch
            # Get a batch of data
            src_batch, tgt_batch = create_sample_data(
                batch_size=batch_size,
                src_seq_len=15,
                tgt_seq_len=15,
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size
            )
            
            # Create input and output sequences for teacher forcing
            # Shift target sequence to create input and output (remove EOS from input, BOS from output)
            tgt_in = tgt_batch[:, :-1]  # Input to the decoder
            tgt_out = tgt_batch[:, 1:]  # Expected output from the decoder
            
            # Create masks
            src_mask, tgt_mask = create_masks(src_batch, tgt_in, pad_idx)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(src_batch, tgt_in, src_mask, tgt_mask)
            
            # Compute loss
            loss = criterion(output.view(-1, tgt_vocab_size), tgt_out.contiguous().view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Print batch statistics
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/10, Loss: {loss.item() / (tgt_out.size(0) * tgt_out.size(1)):.4f}")
        
        # Print epoch statistics
        avg_loss = total_loss / (10 * batch_size * (tgt_batch.size(1) - 1))
        print(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "transformer_example_model.pt")
    
    return model


def inference_example(model):
    """
    Example of performing inference with a trained model.
    
    In a real scenario, you would:
    1. Load trained model weights
    2. Process actual text input (tokenization, etc.)
    3. Generate translated text
    4. Post-process the output
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create a sample source sequence
    src_vocab_size = 1000
    batch_size = 2
    src_seq_len = 10
    
    # Create random source data
    src_data = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    
    # Create source mask (for padding)
    src_mask = Transformer.create_padding_mask(src_data, pad_idx=0)
    
    # Assume BOS token index is 1
    bos_idx = 1
    
    # Perform greedy decoding
    with torch.no_grad():
        output_seq = model.greedy_decode(
            src_data,
            src_mask,
            max_len=15,
            start_symbol=bos_idx
        )
    
    # Print results (in a real scenario, you would convert indices to tokens)
    print("Source sequences:", src_data)
    print("Generated sequences:", output_seq)
    
    return output_seq


def main():
    """Main entry point for the example."""
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train a simple model
    print("\n===== Training Example =====")
    model = train_example()
    
    # Perform inference
    print("\n===== Inference Example =====")
    inference_example(model)
    
    print("\nTransformer example completed successfully!")


if __name__ == "__main__":
    main() 