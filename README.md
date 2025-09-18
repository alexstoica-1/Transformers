
# Implementing Transformer from Scratch

This repository provides a detailed guide and implementation of the Transformer architecture from the ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper. The implementation focuses on understanding each component through clear code, comprehensive testing, and visual aids.

## Summary and Key Insights

### Paper Reference
- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- Key sections: 
  - 3.1: Encoder and Decoder Stacks
  - 3.2: Attention Mechanism
  - 3.3: Position-wise Feed-Forward Networks
  - 3.4: Embeddings and Softmax
  - 3.5: Positional Encoding
  - 5.4: Regularization (dropout strategy)

### Implementation Strategy
Breaking down the architecture into manageable pieces and gradually adding complexity:

1. Start with foundational components:
   - Embedding + Positional Encoding
   - Single-head self-attention
   
2. Build up attention mechanism:
   - Extend to multi-head attention
   - Add cross-attention capability
   - Implement attention masking
   
3. Construct larger components:
   - Encoder (self-attention + FFN)
   - Decoder (masked self-attention + cross-attention + FFN)
   
4. Combine into final architecture:
   - Encoder-Decoder stack
   - Full Transformer with input/output layers


## Implementation Details

### Embedding and Positional Encoding
This implements the input embedding and positional encoding from Section 3.5 of the paper. Key points:
- Embedding dimension can differ from model dimension (using projection)
- Positional encoding uses sine and cosine functions
- Scale embeddings by √d_model
- Apply dropout to the sum of embeddings and positional encodings

### Transformer Attention
Implements the core attention mechanism from Section 3.2.1. Formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V

Key points:
- Supports both self-attention and cross-attention
- Handles different sequence lengths for encoder/decoder
- Scales dot products by 1/√d_k
- Applies attention masking before softmax

### Feed-Forward Network (FFN)
Implements the position-wise feed-forward network from Section 3.3: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

Key points:
- Two linear transformations with ReLU in between
- Inner layer dimension (d_ff) is typically 2048
- Applied identically to each position

### Transformer Decoder
Implements decoder layer from Section 3.1, with three sub-layers:
- Masked multi-head self-attention
- Multi-head cross-attention with encoder output
- Position-wise feed-forward network

Key points:
- Self-attention uses causal masking
- Cross-attention allows attending to all encoder outputs
- Each sub-layer followed by residual connection and layer normalization
- Create causal mask using upper triangular matrix:
 ```python
 mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
 mask = mask.masked_fill(mask == 1, float('-inf'))
 ```

This creates a pattern where position i can only attend to positions ≤ i
Using -inf ensures zero attention to future positions after softmax
Visualization of mask for seq_len=5:\
 [[0, -inf, -inf, -inf, -inf],\
 [0,    0, -inf, -inf, -inf],\
 [0,    0,    0, -inf, -inf],\
 [0,    0,    0,    0, -inf],\
 [0,    0,    0,    0,    0]]

### Encoder-Decoder Stack
Implements the full stack of encoder and decoder layers from Section 3.1.
Key points:
- Multiple encoder and decoder layers (typically 6)
- Each encoder output feeds into all decoder layers
- Maintains residual connections throughout the stack

### Full Transformer
Combines all components into complete architecture:
- Input embeddings for source and target
- Positional encoding
- Encoder-decoder stack
- Final linear and softmax layer

Key points:
- Handles different vocabulary sizes for source/target
- Shifts decoder inputs for teacher forcing
- Projects outputs to target vocabulary size
- Applies log softmax for training stability

### Testing
The implementation includes comprehensive tests for each component:

- Shape preservation through layers
- Masking effectiveness
- Attention pattern verification
- Forward/backward pass validation
- Parameter and gradient checks

See the notebook for detailed test implementations and results.

### Visualizations
The implementation includes visualizations of:

- Attention patterns
- Positional encodings
- Masking effects
- Layer connectivity
