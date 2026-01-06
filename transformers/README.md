# Transformer Implementation (60 points)

The Transformer architecture is important for ML because it revolutionized NLP by introducing the self-attention mechanism, enabling effective modeling of long-range dependencies in sequential data. This led to state-of-the-art performance in various NLP tasks such as machine translation, text generation, and sentiment analysis. In this problem, you will implement a tiny decoder-only Transformer architecture and gain insight into its basic structure.

The main purpose of this assignment is to familiarize you with the PyTorch framework and the Transformer architecture. To do this, you will train a small decoder-only Transformer model on the tiny Shakespeare dataset.

Before diving into the coding, you may want to familiarize yourself with the Transformer architecture by reading the [original paper](https://arxiv.org/abs/1706.03762), "Attention Is All You Need". 

## General Instructions

You will implement core components of a decoder-only transformer model by completing the provided functions. The code structure follows the original transformer architecture with some simplifications for educational purposes.

- Do not import libraries other than those already imported in the original code.
- Follow the type annotations and ensure function return values match the required types.
- Only modifications in `transformer_model.py` will be accepted and graded.
- You can work directly in Vocareum or download the files from "work", code in your own workspace, and upload the changes (recommended). If you work on Vocareum directly, you need to refresh the page for the generated plots to show up.
- Click Submit when ready to submit your code for auto-grading. Your final grade is determined by your last submission.

## Implementation Tasks

In `transformer_model.py` there are seven "TODO" items for you to complete.
General instructions for each are below (mirroring the comments in `transformer_model.py`—which additionally contains specific hints for how to approach each operation syntactically).
For any PyTorch (`torch`) functionality you are not familiar with, refer to the [documentation](https://pytorch.org/docs/stable/index.html). There is also an official [tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html) that may serve as a useful guide for familiarizing yourself with core concepts and syntax.

### TODO 1: Attention Head Implementation
In function `Head.forward()`, implement the self-attention mechanism that computes attention scores and weighted values.

Required steps:
1. Compute key, query, and value projections using provided linear layers
2. Calculate scaled attention scores: (Q @ K.transpose(-2,-1)) / sqrt(head_size)
3. Apply causal mask using the provided tril buffer
4. Apply softmax to get attention weights
5. Compute weighted sum of values

### TODO 2: Multi-Head Attention
In function `MultiHeadAttention.forward()`, implement parallel attention heads and concatenate their outputs.

Required steps:
1. Apply each attention head to the input
2. Concatenate the outputs along the feature dimension

### TODO 3: Transformer Block
In function `Block.forward()`, implement a complete transformer block combining attention and feedforward layers.

Required steps:
1. Apply layer normalization and self-attention with residual connection
2. Apply layer normalization and feedforward layer with residual connection

### TODO 4: Token and Position Embeddings
In function `BigramLanguageModel.forward()`, implement the embedding layer that combines token and positional information.

Required steps:
1. Get token embeddings from embedding table
2. Generate position embeddings using position embedding table
3. Combine token and position embeddings

### TODO 5: Transformer Application
In function `BigramLanguageModel.forward()`, implement the main transformer computation.

Required steps:
1. Pass combined embeddings through transformer blocks
2. Apply final layer normalization
3. Project to vocabulary size

### TODO 6: Loss Computation
In function `BigramLanguageModel.forward()`, implement the cross-entropy loss computation.

Required steps:
1. Reshape logits and targets appropriately
2. Compute cross entropy loss using F.cross_entropy

### TODO 7: Generation Loop
In function `BigramLanguageModel.generate()`, implement autoregressive token generation.

Required steps:
1. Implement token generation loop
2. Handle sequence cropping
3. Implement sampling logic

## Grading Criteria

Our grading scripts run 10 tests to verify your implementation works correctly. Each test is worth 6 points, for a total of 60 points:

1. **Attention Head Implementation** (6 points): Tests if the attention head correctly computes attention scores and applies weights (TODO 1)

2. **Multi-Head Attention Implementation** (6 points): Verifies that multiple attention heads process and combine information properly (TODO 2)

3. **Transformer Block Implementation** (6 points): Tests the complete transformer block with layer normalization and residual connections (TODO 3)

4. **Embeddings Implementation** (6 points): Tests token and position embeddings functionality (TODO 4)

5. **Transformer Forward Implementation** (6 points): Tests the main transformer computation and output projection (TODO 5)

6. **Loss Computation** (6 points): Verifies that cross-entropy loss is computed correctly (TODO 6)

7. **Generation Implementation** (6 points): Tests the autoregressive token generation functionality (TODO 7)

8. **Attention Mask** (6 points): Tests if causal masking in attention works correctly (related to TODO 1)

9. **Layer Normalization** (6 points): Tests if layer normalization is applied correctly (related to TODO 3)

10. **End-to-End Training** (6 points): Tests that the complete model can train and improve on the Shakespeare dataset (requires all TODOs to be implemented correctly)

Total: 60 points

## Testing

Run the training script to verify your implementation:

```bash
python3 train.py
```

You should observe the losses decreasing over the course of training, and the generated text improving by the end of training. Your model is likely functioning properly if:

1. The training loss starts around 4.2-4.7 and decreases steadily
2. The validation loss follows a similar pattern to the training loss
3. After 5000 training steps, the test loss should be approximately 1.9.
4. The model can generate coherent text samples that resemble Shakespeare's style (perhaps a little rough around the edges, given our computational constraints:)

The training process takes a few minutes on a CPU and less on a GPU. The model will generate a text sample before and after training so you can observe the improvement. 

## How Training and Evaluation Work

This assignment implements a character-level language model trained on Shakespeare's text. Here's how the training and evaluation process works:

### Data Processing
- The code downloads a "tiny Shakespeare" dataset and processes it at the character level
- Each character is mapped to a unique integer ID using a vocabulary
- The data is split into train/validation/test sets (70%/15%/15%)

### Training Process
- The model is trained to predict the next character in a sequence
- During each training step:
  - Random chunks of text of length `block_size` (32 by default) are sampled
  - The model tries to predict each next character in the sequence
  - Cross-entropy loss measures how well the predictions match the actual next characters
  - Weights are updated using backpropagation and Adam optimizer

### Evaluation
The model is evaluated in two complementary ways:

1. **Loss Calculation**:
   - The `estimate_loss()` function calculates loss on train/validation/test sets
   - For each evaluation, it:
     - Samples random chunks from the respective dataset
     - Computes how well the model predicts the next character
     - Averages the loss across multiple batches
   - This quantitative measure tells us how well the model generalizes to unseen text

2. **Text Generation**:
   - The model generates new text starting from an empty context
   - It works autoregressively:
     - Predicts probabilities for the next character
     - Samples one character from this probability distribution
     - Adds this character to the context
     - Repeats the process for the desired length
   - This qualitative evaluation let us see if the generated text resembles Shakespeare's style

The test loss is not evaluating the quality of generated text - it's measuring how well the model predicts the next character in Shakespeare text it hasn't seen during training. A lower test loss generally correlates with better text generation, but the ultimate test is reading the generated output and seeing if it captures Shakespeare's patterns. 