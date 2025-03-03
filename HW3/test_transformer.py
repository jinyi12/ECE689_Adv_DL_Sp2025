import unittest
import torch
import torch.nn as nn
from transformer import (
    Transformer,
    PositionalEncoding,
    Encoder,
    EncoderLayer,
    Decoder,
    DecoderLayer,
    Embedding,
    DecoderGenerator,
    MultiHeadAttention,
    PositionwiseFeedForward,
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
)

class TestTransformer(unittest.TestCase):
    """
    Tests for the Transformer model and its components.
    """
    
    def setUp(self):
        """Set up shared test variables."""
        # Test dimensions
        self.batch_size = 4
        self.src_seq_len = 10
        self.tgt_seq_len = 8
        self.src_vocab_size = 1000
        self.tgt_vocab_size = 1000
        self.model_dimension = 32
        self.num_heads = 4
        self.d_ff = 128
        self.pad_idx = 0  # Assume 0 is padding token index
        
        # Create sample data
        self.src = torch.randint(1, self.src_vocab_size, (self.batch_size, self.src_seq_len))
        self.tgt = torch.randint(1, self.tgt_vocab_size, (self.batch_size, self.tgt_seq_len))
        
        # Create padding mask
        self.src_mask = Transformer.create_padding_mask(self.src, self.pad_idx)
        
        # Create a square subsequent mask for the target
        self.tgt_mask = torch.triu(torch.ones(self.tgt_seq_len, self.tgt_seq_len), diagonal=1).bool() 
        self.tgt_mask = self.tgt_mask.unsqueeze(0).expand(self.batch_size, -1, -1)

    def test_embedding(self):
        """Test the Embedding module."""
        embedding = Embedding(self.model_dimension, self.src_vocab_size)
        output = embedding(self.src)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.src_seq_len, self.model_dimension))
        
        # Check if output values are reasonable (not all zeros or NaNs)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse((output == 0).all())

    def test_positional_encoding(self):
        """Test the PositionalEncoding module."""
        pos_encoding = PositionalEncoding(self.model_dimension, max_seq_length=1000)
        
        # Create sample input
        embeddings = torch.rand(self.batch_size, self.src_seq_len, self.model_dimension)
        output = pos_encoding(embeddings)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.src_seq_len, self.model_dimension))
        
        # Check if output values are different from input (positional encoding added)
        self.assertFalse(torch.allclose(output, embeddings))

    def test_encoder_layer(self):
        """Test a single EncoderLayer."""
        encoder_layer = EncoderLayer(self.model_dimension, self.num_heads, self.d_ff)
        
        # Input representation with shape [batch_size, seq_len, d_model]
        x = torch.rand(self.batch_size, self.src_seq_len, self.model_dimension)
        output = encoder_layer(x, self.src_mask)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.src_seq_len, self.model_dimension))

    def test_encoder(self):
        """Test the full Encoder."""
        encoder_layer = EncoderLayer(self.model_dimension, self.num_heads, self.d_ff)
        encoder = Encoder(encoder_layer, num_layers=2)
        
        # Input representation with shape [batch_size, seq_len, d_model]
        x = torch.rand(self.batch_size, self.src_seq_len, self.model_dimension)
        output = encoder(x, self.src_mask)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.src_seq_len, self.model_dimension))

    def test_decoder_layer(self):
        """Test a single DecoderLayer."""
        decoder_layer = DecoderLayer(self.model_dimension, self.num_heads, self.d_ff)
        
        # Input representations
        tgt_x = torch.rand(self.batch_size, self.tgt_seq_len, self.model_dimension)
        memory = torch.rand(self.batch_size, self.src_seq_len, self.model_dimension)
        
        # Forward pass
        output = decoder_layer(tgt_x, memory, self.src_mask, self.tgt_mask)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.tgt_seq_len, self.model_dimension))

    def test_decoder(self):
        """Test the full Decoder."""
        decoder_layer = DecoderLayer(self.model_dimension, self.num_heads, self.d_ff)
        decoder = Decoder(decoder_layer, num_layers=2)
        
        # Input representations
        tgt_x = torch.rand(self.batch_size, self.tgt_seq_len, self.model_dimension)
        memory = torch.rand(self.batch_size, self.src_seq_len, self.model_dimension)
        
        # Forward pass
        output = decoder(tgt_x, memory, self.tgt_mask, self.src_mask)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.tgt_seq_len, self.model_dimension))

    def test_transformer_init(self):
        """Test Transformer initialization."""
        transformer = Transformer(
            self.src_vocab_size,
            self.tgt_vocab_size,
            self.model_dimension,
            self.num_heads,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=self.d_ff
        )
        
        # Check that all components are properly initialized
        self.assertIsInstance(transformer.src_embedding, Embedding)
        self.assertIsInstance(transformer.tgt_embedding, Embedding)
        self.assertIsInstance(transformer.positional_encoding, PositionalEncoding)
        self.assertIsInstance(transformer.encoder, Encoder)
        self.assertIsInstance(transformer.decoder, Decoder)
        self.assertIsInstance(transformer.generator, DecoderGenerator)

    def test_transformer_forward(self):
        """Test the forward pass of the full Transformer."""
        transformer = Transformer(
            self.src_vocab_size,
            self.tgt_vocab_size,
            self.model_dimension,
            self.num_heads,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=self.d_ff
        )
        
        # Forward pass
        output = transformer(self.src, self.tgt, self.src_mask, self.tgt_mask)
        
        # Check output shape [batch_size, tgt_seq_len, tgt_vocab_size]
        self.assertEqual(output.shape, (self.batch_size, self.tgt_seq_len, self.tgt_vocab_size))
        
        # Check if output contains valid probabilities
        self.assertTrue(torch.all(output >= 0))  # All values should be non-negative
        
        # Sum of probabilities per position should be close to 1
        prob_sums = output.sum(dim=2)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5))

    def test_transformer_greedy_decode(self):
        """Test the greedy decoding functionality."""
        transformer = Transformer(
            self.src_vocab_size,
            self.tgt_vocab_size,
            self.model_dimension,
            self.num_heads,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=self.d_ff
        )
        
        # Set a start symbol
        start_symbol = 1  # Assume 1 is BOS token index
        
        # Run greedy decoding
        output = transformer.greedy_decode(
            self.src,
            self.src_mask,
            max_len=15,
            start_symbol=start_symbol
        )
        
        # Check shape [batch_size, max_len]
        self.assertEqual(output.shape, (self.batch_size, 15))
        
        # First token should be the start symbol
        self.assertTrue(torch.all(output[:, 0] == start_symbol))
        
        # Values should be valid indices
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output < self.tgt_vocab_size))

    def test_factory_methods(self):
        """Test the factory methods for creating pre-configured transformers."""
        # Test base transformer
        base_transformer = Transformer.build_base_transformer(
            self.src_vocab_size,
            self.tgt_vocab_size
        )
        
        self.assertEqual(base_transformer.model_dimension, 512)
        self.assertEqual(base_transformer.num_heads, 8)
        self.assertEqual(base_transformer.num_encoder_layers, 6)
        self.assertEqual(base_transformer.num_decoder_layers, 6)
        self.assertEqual(base_transformer.d_ff, 2048)
        
        # Test big transformer
        big_transformer = Transformer.build_big_transformer(
            self.src_vocab_size,
            self.tgt_vocab_size
        )
        
        # Assert big model has larger dimensions
        self.assertTrue(big_transformer.model_dimension > base_transformer.model_dimension)
        self.assertTrue(big_transformer.num_heads > base_transformer.num_heads)
        self.assertTrue(big_transformer.d_ff > base_transformer.d_ff)

    def test_mask_generation(self):
        """Test the mask generation utilities."""
        # Test padding mask
        seq = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])
        pad_idx = 0
        
        mask = Transformer.create_padding_mask(seq, pad_idx)
        
        # Check shape [batch_size, 1, seq_len]
        self.assertEqual(mask.shape, (2, 1, 4))
        
        # Verify mask values
        expected_mask = torch.tensor([[[False, False, True, True]], 
                                     [[False, True, True, True]]])
        self.assertTrue(torch.all(mask == expected_mask))
        
        # Test subsequent mask
        sz = 4
        # Create a simple transformer instance for this test
        transformer = Transformer(
            src_vocab_size=10,
            tgt_vocab_size=10,
            model_dimension=8,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_ff=16
        )
        subseq_mask = transformer._generate_square_subsequent_mask(sz)
        
        # Check shape [sz, sz]
        self.assertEqual(subseq_mask.shape, (4, 4))
        
        # Lower triangular should be zeros (or finite values)
        lower_tri = torch.tril(torch.ones(sz, sz), diagonal=0)
        self.assertTrue(torch.all(torch.isfinite(subseq_mask) == lower_tri))

if __name__ == '__main__':
    unittest.main() 