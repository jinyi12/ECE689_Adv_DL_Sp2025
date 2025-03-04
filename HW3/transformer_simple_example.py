#!/usr/bin/env python3
"""
Transformer Usage Example for English-French Translation

This script demonstrates how to use the Transformer model for the English to French translation task
as specified in the HW3 notebook. It shows how to:
1. Load and prepare the data from en-fr.txt
2. Tokenize and preprocess sequences
3. Train the Transformer model
4. Perform inference with the trained model for translation

This example follows the dataset preparation steps from the HW3.ipynb notebook.
"""

import math
import os
import re
from unicodedata import normalize

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from transformer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, Transformer
from transformers import PreTrainedTokenizerFast

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Token indices - must match what's used in transformer.py'
PAD_IDX = 0
START_IDX = 1
END_IDX = 2
UNK_IDX = 3


# Modern vocabulary approach using HuggingFace
class HFVocabulary:
    """Modern vocabulary class using HuggingFace tokenizers."""

    def __init__(self):
        """Initialize tokenizer with basic configuration."""
        # Create a new tokenizer from scratch - alternatively could use a pretrained one
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=None,  # We'll train it from scratch
            unk_token=UNK_TOKEN,
            pad_token=PAD_TOKEN,
            bos_token=BOS_TOKEN,
            eos_token=EOS_TOKEN,
        )
        # For compatbility with the rest of the code
        self.PAD_IDX = self.tokenizer.pad_token_id
        self.UNK_IDX = self.tokenizer.unk_token_id
        self.BOS_IDX = self.tokenizer.bos_token_id
        self.EOS_IDX = self.tokenizer.eos_token_id

    def fit_on_texts(self, texts):
        """Build vocabulary from list of texts.

        In HuggingFace, we'd typically use a tokenizer.train_new_from_iterator method,
        but for simplicity and compatibility, we'll create a basic word-level vocabulary.
        """
        # Create a corpus for training the tokenizer
        words = set()
        for text in texts:
            words.update(text.split())

        # Add words to the tokenizer's vocabulary
        self.tokenizer.add_tokens(list(words))

        # For compatibility
        self.num_words = len(self.tokenizer)

    def texts_to_sequences(self, texts):
        """Convert list of texts to list of sequences of indices."""
        # Use the tokenizer to encode the texts
        sequences = []
        for text in texts:
            # Encode without special tokens for compatibility
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            sequences.append(encoded)
        return sequences

    def sequences_to_texts(self, sequences):
        """Convert list of sequences of indices to list of texts."""
        texts = []
        for sequence in sequences:
            # Filter out padding tokens
            filtered_sequence = [idx for idx in sequence if idx != self.PAD_IDX]
            # Decode the sequence
            text = self.tokenizer.decode(filtered_sequence, skip_special_tokens=True)
            texts.append(text)
        return texts


# Use the new vocabulary class
Vocabulary = HFVocabulary


def clean_text(text):
    """Normalize text, remove punctuation and non-alphabetic characters, and convert to lowercase."""
    text = normalize("NFD", text.lower())
    # Remove all non-alphabetic characters
    text = re.sub("[^A-Za-z ]+", "", text)
    return text


def clean_and_prepare_text(text):
    """Add [start] and [end] tokens to French text."""
    text = f"{BOS_TOKEN} {clean_text(text)} {EOS_TOKEN}"
    return text


def load_and_prepare_data(file_path="Data/en-fr.txt", max_samples=None):
    """
    Load and prepare the data from en-fr.txt file.

    Args:
        file_path: Path to the en-fr.txt file
        max_samples: Maximum number of samples to use (useful for debugging)

    Returns:
        DataFrame with 'en' and 'fr' columns containing cleaned text
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}. Please download the en-fr.txt file."
        )

    # Load data
    df = pd.read_csv(
        file_path, names=["en", "fr", "attr"], usecols=["en", "fr"], sep="\t"
    )

    # Shuffle and limit samples if needed
    df = df.sample(frac=1, random_state=42)
    if max_samples:
        df = df.head(max_samples)

    df = df.reset_index(drop=True)

    # Clean text
    df["en"] = df["en"].apply(lambda row: clean_text(row))
    df["fr"] = df["fr"].apply(lambda row: clean_and_prepare_text(row))

    return df


def prepare_dataset(df, en_tokenizer, fr_tokenizer, batch_size=64):
    """
    Prepare the dataset for training and validation.

    Args:
        df: DataFrame with 'en' and 'fr' columns
        en_tokenizer: Tokenizer for English text
        fr_tokenizer: Tokenizer for French text
        batch_size: Batch size for DataLoader

    Returns:
        train_dataloader, val_dataloader, sequence_length
    """
    # Find max sequence length for padding
    en_max_len = max(len(line.split()) for line in df["en"])
    fr_max_len = max(len(line.split()) for line in df["fr"])
    sequence_len = max(en_max_len, fr_max_len)

    print(f"Max phrase length (English): {en_max_len}")
    print(f"Max phrase length (French): {fr_max_len}")
    print(f"Sequence length: {sequence_len}")

    # Tokenize texts
    en_sequences = en_tokenizer.texts_to_sequences(df["en"])
    fr_sequences = fr_tokenizer.texts_to_sequences(df["fr"])

    # Convert to tensors
    en_tensors = [torch.tensor(seq, dtype=torch.long) for seq in en_sequences]
    fr_tensors = [torch.tensor(seq, dtype=torch.long) for seq in fr_sequences]

    # Pad sequences
    en_padded = pad_sequence(en_tensors, batch_first=True, padding_value=PAD_IDX)
    fr_padded = pad_sequence(fr_tensors, batch_first=True, padding_value=PAD_IDX)

    # Create decoder input and output
    decoder_input = fr_padded[:, :-1]  # remove last token
    decoder_output = fr_padded[:, 1:]  # remove first token ([start])

    # Create dataset
    dataset = TensorDataset(en_padded, decoder_input, decoder_output)

    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, sequence_len


def create_masks(src, tgt, pad_idx=PAD_IDX):
    """
    Create masks for transformer training.

    The function generates two masks to be used in the transformer model:
    1. Source Mask (src_mask): Masks padding tokens in the source sequence to prevent the model from
       attending to those positions.
    2. Target Mask (tgt_mask): Combines two types of masks for the target sequence:
        - Padding Mask: Masks padding tokens.
        - No-Look-Ahead Mask: Prevents positions in the target sequence from attending to future tokens
         (enforcing an autoregressive behavior during training).

    Steps:
    1. Determine the batch size from the shape of the source tensor.
    2. Source mask creation:
        - Compare the source tensor with the padding token index to get a binary mask where tokens not equal to
         the padding token are True.
        - Reshape the mask to have shape (batch_size, 1, 1, src_seq_len) so it can be broadcasted across
         the multi-head attention computations.
        - This shape is used because the attention weight matrix has the shape (batch_size, num_heads, query_length, key_length).
         Therefore the (1, 1) dimensions are used to broadcast the mask across the query and key lengths.

    3. Target mask creation:
        a. Determine the target sequence length.
        b. Generate the target padding mask similar to the source mask:
            - This masks out the padding tokens of the target sequence.
            - The target padding mask shape is adjusted to (batch_size, 1, 1, tgt_seq_len).
        c. Create the no-look-ahead mask (autoregressive constraint):
            - Build a mask of shape (1, 1, tgt_seq_len, tgt_seq_len) using torch.triu, which creates an upper-triangular matrix.
            - The mask is used to ensure a position in the target sequence can only attend to positions in the past and
            the present (i.e., lower triangular including the diagonal). Transposing is applied though in this case it
            maintains the same shape.
        d. Combine the target padding mask and the no-look-ahead mask using a logical AND:
            - Both conditions must be satisfied. That is, a token is attendable if it is not a pad token and if it is
            not in the future relative to the current token.
        e. Note that combining the target padding mask and the no-look-ahead mask results in a tensor of shape (batch_size, 1, tgt_seq_len, tgt_seq_len).
    4. Return the source mask and the combined target mask.

    Args:
        src (Tensor): Source token indices with shape [batch_size, src_seq_len].
        tgt (Tensor): Target token indices with shape [batch_size, tgt_seq_len].
        pad_idx (int): The index representing the padding token in the vocabulary.

    Returns:
        tuple: (src_mask, tgt_mask) where:
            - src_mask: Tensor of shape [batch_size, 1, 1, src_seq_len] used to mask source padding tokens.
            - tgt_mask: Tensor of shape [batch_size, 1, tgt_seq_len, tgt_seq_len] used to mask both target padding tokens
              and future tokens (no look-ahead).
    """
    # 1. Determine the batch size from the source tensor shape.
    batch_size = src.shape[0]

    # 2. Create the source mask:
    # Compare each token with pad_idx -> True where token != pad_idx.
    # Reshape to (batch_size, 1, 1, src_seq_len) for compatibility in multi-head attention.
    src_mask = (src != pad_idx).view(batch_size, 1, 1, -1)

    # 3. Create the target mask:
    # a. Determine the target sequence length.
    tgt_seq_len = tgt.shape[1]

    # b. Create the target padding mask:
    # This is similar to the src_mask but for the target. It marks all positions that are not pad tokens.
    # Reshape to (batch_size, 1, 1, tgt_seq_len).
    tgt_padding_mask = (tgt != pad_idx).view(batch_size, 1, 1, -1)

    # c. Create the no-look-ahead mask (autoregressive constraint):
    # This mask is designed to prevent each position from attending to future positions.
    # The mask is created as an upper triangular matrix of ones.
    # Here, we first create a matrix of ones with shape (1, 1, tgt_seq_len, tgt_seq_len) on the same device as tgt,
    # then use torch.triu to keep only the upper triangular region.
    # Transposing in this manner (though optional in this specific case) ensures the correct shape and orientation.
    device = tgt.device
    tgt_no_look_forward_mask = torch.triu(
        torch.ones((1, 1, tgt_seq_len, tgt_seq_len), device=device) == 1
    ).transpose(2, 3)

    # d. Combine both target masks:
    # The final target mask is the logical AND operation between the padding mask and the no-look-ahead mask.
    # This ensures that only valid positions (non-padding and not attending to future tokens) are allowed.
    tgt_mask = tgt_padding_mask & tgt_no_look_forward_mask

    return src_mask, tgt_mask


class LabelSmoothing(nn.Module):
    """
    A module that implements label smoothing for classification tasks using Kullback-Leibler Divergence loss.

    Label smoothing is a regularization technique that replaces the hard target (one-hot) labels
    with a softer distribution. Instead of assigning a probability of 1.0 to the ground-truth class and 0.0 to others,
    label smoothing assigns a probability of (1 - smoothing) to the true class and distributes the smoothing value
    equally among the remaining classes. This approach can help the model to generalize better by preventing it
    from becoming overconfident in its predictions.

    Steps:
    1. Input Verification:
       - The `forward` method asserts that the last dimension of the input tensor `x` corresponds to the number
         of classes (`size`). This ensures consistency in the prediction tensor's shape.

    2. Initialization of the Target Distribution:
       - A copy of the input tensor `x` is created to serve as a template for the target (smoothed) distribution.
       - The entire target distribution tensor is then filled with a uniform smoothing value calculated as:
         smoothing value = smoothing / (size - 2)
         This formula assumes that one slot is reserved for the correct class and another for the padding token.

    3. Assigning the Correct Class Probability:
       - For each position in the target tensor, the correct class (as specified by `target`) is assigned a high
         confidence value of (1.0 - smoothing) by scattering the value into the correct position.

    4. Handling of the Padding Token:
       - The probability assigned to the padding index is explicitly set to 0 so that it does not influence the loss.
       - Additionally, any position in the target containing a padding token is masked out by zeroing out its entire distribution.

    5. Loss Computation:
       - The smoothed target distribution is compared against the model's predicted distribution using KL-divergence.
       - Since the criterion expects log-probabilities from the model predictions, log(x) is provided as input
         to the KLDivLoss function.

    Args:
        size (int): Number of classes (i.e., the vocabulary size).
        padding_idx (int): Index used for the padding token. No probability mass is assigned to this index.
        smoothing (float, optional): Smoothing factor to use. Default is 0.1. The true label is assigned
                                     a probability of \((1.0 - \text{smoothing})\) and the remaining probability mass is
                                     uniformly distributed over the other classes.

    Attributes:
        criterion (nn.KLDivLoss): The loss function instance used to compute the divergence between the predicted
                                  distribution (after log-softmax) and the smoothed target distribution.
        confidence (float): The probability reserved for the correct class, computed as \((1.0 - \text{smoothing})\).
        smoothing (float): The smoothing parameter provided.
        size (int): The number of classes.
        true_dist (Tensor): A tensor storing the computed smoothed target distribution (updated each forward pass).
    """

    def __init__(self, size, padding_idx, smoothing=0.1):
        super().__init__()
        # Initialize the KLDivLoss with reduction sum to accumulate the loss over all items.
        self.criterion = nn.KLDivLoss(reduction="sum")

        # The index for the padding token.
        self.padding_idx = padding_idx

        # Confidence is the probability mass to assign to the true label.
        self.confidence = 1.0 - smoothing

        # The smoothing factor used.
        self.smoothing = smoothing

        # The total number of classes.
        self.size = size

        # A placeholder for the target distribution (useful for debugging or analysis).
        self.true_dist = None

    def forward(self, x, target):
        """
        Compute the label smoothing loss given the model predictions and the target labels.

        Steps:
        1. Verify that the last dimension of x matches 'size'.
        2. Create a copy of x to initialize the true distribution.
        3. Fill the distribution with a uniform value \(\frac{\text{smoothing}}{\text{size} - 2}\).
        4. Scatter the high confidence value into the position of the actual target token.
        5. Set the probability for the padding index to 0.
        6. Mask out any positions where the target is the padding token.
        7. Compute and return the KL-divergence loss between the log of predictions and the smoothed targets.

        Args:
            x (Tensor): Model output predictions with shape (batch_size, sequence_length, size).
                        These are raw scores (not log-probabilities).
            target (Tensor): Ground truth token indices with shape (batch_size, sequence_length).

        Returns:
            Tensor: The computed KL-divergence loss.
        """
        # Ensure the prediction tensor is of the expected shape.
        assert x.size(-1) == self.size, (
            "The last dimension of x must be equal to the number of classes (size)."
        )

        # Create a copy of x to use as a template for the true (smoothed) distribution.
        true_dist = x.clone()  # shape: (batch_size, sequence_length, size)

        # Fill the entire true_dist with the uniform smoothing value.
        # We subtract 2 from size to account for the correct class and the padding token.
        true_dist.fill_(
            self.smoothing / (self.size - 2)
        )  # shape: (batch_size, sequence_length, size)

        # Scatter the confidence value into the positions corresponding to the ground truth tokens.
        # For each position in the batch, the target token index gets the high probability (confidence).
        true_dist.scatter_(
            -1, target.unsqueeze(-1), self.confidence
        )  # shape: (batch_size, sequence_length, size)

        # Ensure that the padding index is assigned a probability of 0.
        true_dist[:, self.padding_idx] = 0

        # Create a mask for positions where the target is the padding token.
        mask = target == self.padding_idx

        # For these positions, zero out the probability distribution.
        if mask.sum() > 0:
            true_dist.masked_fill_(mask.unsqueeze(-1), 0.0)

        # Optionally store the computed smoothed true distribution for debugging or analysis.
        self.true_dist = true_dist

        # Compute and return the KL divergence loss.
        # Note that x is converted to log-probabilities before loss calculation.
        return self.criterion(x.log(), true_dist)


def train_model(
    train_dataloader,
    val_dataloader,
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs=10,
    device="cpu",
):
    """
    Train the transformer model.

    Args:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        model: Transformer model
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on

    Returns:
        history: Dictionary with training history
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (src, tgt_in, tgt_out) in enumerate(train_dataloader):
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

            # Create masks
            src_mask, tgt_mask = create_masks(src, tgt_in)

            # Forward pass
            optimizer.zero_grad()
            # print("src_mask shape:", src_mask.shape)
            # print("tgt_mask shape:", tgt_mask.shape)
            # print("src shape:", src.shape)
            # print("tgt_in shape:", tgt_in.shape)
            output = model(src, tgt_in, src_mask, tgt_mask)

            # Reshape for loss calculation
            output_flat = output.view(-1, output.size(-1))
            tgt_out_flat = tgt_out.reshape(-1)

            # Compute loss
            loss = criterion(output_flat, tgt_out_flat)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update parameters
            optimizer.step()
            scheduler.step()

            # Compute accuracy
            _, predicted = torch.max(output.detach(), dim=-1)
            correct = (
                (predicted == tgt_out)
                .masked_fill(tgt_out == PAD_IDX, False)
                .sum()
                .item()
            )
            total = (tgt_out != PAD_IDX).sum().item()

            # Accumulate metrics
            train_loss += loss.item()
            train_correct += correct
            train_total += total

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_dataloader)}, "
                    f"Loss: {loss.item() / total:.4f}, Acc: {100 * correct / total:.2f}%"
                )

        # Compute epoch metrics
        epoch_train_loss = train_loss / len(train_dataloader)
        epoch_train_acc = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for src, tgt_in, tgt_out in val_dataloader:
                src, tgt_in, tgt_out = (
                    src.to(device),
                    tgt_in.to(device),
                    tgt_out.to(device),
                )

                # Create masks
                src_mask, tgt_mask = create_masks(src, tgt_in)

                # Forward pass
                output = model(src, tgt_in, src_mask, tgt_mask)

                # Reshape for loss calculation
                output_flat = output.view(-1, output.size(-1))
                tgt_out_flat = tgt_out.reshape(-1)

                # Compute loss
                loss = criterion(output_flat, tgt_out_flat)

                # Compute accuracy
                _, predicted = torch.max(output, dim=-1)
                correct = (
                    (predicted == tgt_out)
                    .masked_fill(tgt_out == PAD_IDX, False)
                    .sum()
                    .item()
                )
                total = (tgt_out != PAD_IDX).sum().item()

                # Accumulate metrics
                val_loss += loss.item()
                val_correct += correct
                val_total += total

        # Compute epoch metrics
        epoch_val_loss = val_loss / len(val_dataloader)
        epoch_val_acc = 100 * val_correct / val_total

        # Update history
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        # Print epoch metrics
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%"
        )

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "best_transformer_model.pt")
            print(f"Model saved with validation accuracy: {best_val_acc:.2f}%")

    return history


def plot_history(history):
    """Plot training and validation accuracy."""
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], "-", label="Training accuracy")
    plt.plot(history["val_acc"], ":", label="Validation accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="lower right")
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], "-", label="Training loss")
    plt.plot(history["val_loss"], ":", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("transformer_training_history.png")
    plt.show()


def translate_sentence(
    model, sentence, en_tokenizer, fr_tokenizer, device="cpu", max_len=20
):
    """
    Translate an English sentence to French.

    Args:
        model: Trained Transformer model
        sentence: English sentence to translate
        en_tokenizer: English tokenizer
        fr_tokenizer: French tokenizer
        device: Device to run inference on
        max_len: Maximum length of generated translation

    Returns:
        str: Translated French sentence
    """
    model.eval()

    # Clean and tokenize the sentence
    cleaned_sentence = clean_text(sentence)
    tokens = cleaned_sentence.split()

    # Convert to tensor
    indexed = [en_tokenizer.token_to_idx.get(token, UNK_IDX) for token in tokens]
    source = torch.tensor([indexed]).to(device)

    # Create source mask
    src_mask = (source == PAD_IDX).unsqueeze(1).to(device)

    # Generate translation
    with torch.no_grad():
        output = model.greedy_decode(
            source, src_mask=src_mask, max_len=max_len, start_symbol=START_IDX
        )

    # Convert to text
    output_text = fr_tokenizer.sequences_to_texts(output.cpu().numpy())

    return output_text[0]


def main():
    """Main function to run the example."""
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    print("\nLoading and preparing data...")
    data_file = "Data/en-fr.txt"
    max_samples = 10000  # Limit samples for demonstration

    if not os.path.exists(data_file):
        print(
            f"Warning: {data_file} not found. Using synthetic data for demonstration."
        )
        # Create synthetic data for demonstration
        en_samples = [
            "hello world",
            "how are you",
            "good morning",
            "thank you",
            "goodbye",
        ]
        fr_samples = [
            "bonjour le monde",
            "comment allez vous",
            "bonjour",
            "merci",
            "au revoir",
        ]

        df = pd.DataFrame(
            {
                "en": en_samples * 10,
                "fr": [f"{BOS_TOKEN} {fr} {EOS_TOKEN}" for fr in fr_samples] * 10,
            }
        )
    else:
        df = load_and_prepare_data(data_file, max_samples)

    # Print sample data
    print("\nSample data:")
    print(df.head())

    # Create tokenizers
    print("\nCreating tokenizers...")
    en_tokenizer = Vocabulary()
    fr_tokenizer = Vocabulary()

    # Fit tokenizers on texts
    en_tokenizer.fit_on_texts(df["en"])
    fr_tokenizer.fit_on_texts(df["fr"])

    # Print vocabulary sizes
    print(f"English vocabulary size: {en_tokenizer.num_words}")
    print(f"French vocabulary size: {fr_tokenizer.num_words}")

    # Prepare dataset
    print("\nPreparing dataset...")
    batch_size = 64
    train_dataloader, val_dataloader, sequence_len = prepare_dataset(
        df, en_tokenizer, fr_tokenizer, batch_size
    )

    # Create model
    print("\nCreating Transformer model...")
    model = Transformer(
        src_vocab_size=en_tokenizer.num_words,
        tgt_vocab_size=fr_tokenizer.num_words,
        model_dimension=256,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=512,
        dropout=0.1,
        max_seq_length=sequence_len + 100,  # Add some margin
        # max_seq_length=sequence_len,
    ).to(device)

    # Print model summary
    print(
        f"\nModel has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
    )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # Create learning rate scheduler with warmup
    warmup_steps = 4000

    def lr_lambda(step):
        step = max(1, step)  # Avoid division by zero
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        return math.sqrt(warmup_steps / step)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create loss function with label smoothing
    criterion = LabelSmoothing(fr_tokenizer.num_words, PAD_IDX, smoothing=0.1)

    # Train model
    print("\nTraining model...")
    num_epochs = 5  # Reduced for demonstration
    history = train_model(
        train_dataloader,
        val_dataloader,
        model,
        criterion,
        optimizer,
        scheduler,
        num_epochs=num_epochs,
        device=device,
    )

    # Plot training history
    plot_history(history)

    # Load best model for inference
    if os.path.exists("best_transformer_model.pt"):
        model.load_state_dict(
            torch.load("best_transformer_model.pt", map_location=device)
        )

    # Translate example sentences
    print("\nTranslating example sentences:")
    test_sentences = [
        "hello world",
        "how are you",
        "thank you very much",
        "good morning everyone",
        "have a nice day",
    ]

    for sentence in test_sentences:
        translation = translate_sentence(
            model, sentence, en_tokenizer, fr_tokenizer, device
        )
        print(f"English: {sentence}")
        print(f"French:  {translation}")
        print()

    print("\nTransformer example completed successfully!")


if __name__ == "__main__":
    main()
