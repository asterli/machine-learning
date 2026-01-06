import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        # Initialize linear mappings for k, q, v — we do not need bias for these layers
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Create self.tril (using register_buffer because it's an untrainable constant, but still part of our model)
        # NOTE: torch.tril creates lower triangular matrix—used for causal mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """ Input of shape (batch_size=B, sequence_length=T, embedding_dim=C)
            Output of shape (B, T, head_size)
        """
        B, T, C = x.shape
        
        ############################################################################
        # TODO 1: Implement the forward pass of the attention head:
        # 1. Project input into key, query, value using the respective linear layers
        # 2. Calculate attention scores between query and key (don't forget scaling!)
        # 3. Mask future positions using self.tril and mask_fill
        # 4. Apply softmax to get attention weights
        # 5. Use attention weights to combine values
        #
        # Hints:
        # - Use self.key(x), self.query(x), and self.value(x) to get outputs from the linear layers
        # - For matrix multiplication between two tensors, use the @ operator (e.g., <mat1> @ <mat2>)
        # - For transposing dimensions, use tensor.transpose
        # - Scale attention scores by dividing by sqrt(head_size) (use <tensor>.size)
        # - You will use self.tril to mask future positions; use this line: 
        #   weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # - Apply softmax with F.softmax
        # - The final output is the weighted sum of values (use matrix multiplication)
        ############################################################################

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        weights = q @ k.transpose(-2,-1) / (k.size(-1) ** 0.5)
        
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        weights = F.softmax(weights,dim=-1)
        
        out = weights @ v

        ############################################################################
        # END TODO 1
        ############################################################################

        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        """ Input of shape (B, T, C), output of shape (B, T, C) """
        
        ############################################################################
        # TODO 2: Implement multi-head attention:
        # 1. Apply each attention head to input x
        # 2. Concatenate all head outputs along the last dimension
        
        # Hints:
        # - Apply each head using [h(x) for h in self.heads]
        # - Use torch.cat to concatenate outputs along the feature dimension
        ############################################################################

        out = torch.cat([h(x) for h in self.heads],dim=-1)

        ############################################################################
        # END TODO 2
        ############################################################################

        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        ############################################################################
        # TODO 3: Implement transformer block:
        # 1. Apply layer normalization (self.ln1) and self-attention (self.sa) with residual connection
        # 2. Apply layer normalization (self.ln2) and feedforward (self.ffwd) with residual connection
        
        # Hints:
        # - Remember to apply layer normalization BEFORE attention
        # - Use residual connections with addition, i.e. x + f(x)
        ############################################################################

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        ############################################################################
        # END TODO 3
        ############################################################################
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        ############################################################################
        # TODO 4: Get token and position embeddings
        # 1. Get token embeddings using token_embedding_table
        # 2. Get position embeddings using position_embedding_table and torch.arange
        # 3. Combine token and position embeddings
        
        # Hints:
        # - Pass indices to self.position_embedding_table like:
        #   self.position_embedding_table(torch.arange(T, device=device))
        # - Combine token and position embeddings with simple addition: tok_emb + pos_emb
        ############################################################################

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        ############################################################################
        # END TODO 4
        ############################################################################

        ############################################################################
        # TODO 5: Apply transformer blocks and final layer normalization
        # 1. Pass embeddings through transformer blocks using self.blocks
        # 2. Apply final layer normalization using self.ln_f
        # 3. Project to vocabulary size using self.lm_head
        ############################################################################

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        ############################################################################
        # END TODO 5
        ############################################################################

        if targets is None:
            loss = None
        else:
            ############################################################################
            # TODO 6: Compute cross entropy loss
            # 1. Reshape logits and targets for cross_entropy function
            # 2. Calculate loss using F.cross_entropy
            
            # Hints:
            # - For loss computation, reshape logits from (B, T, C) to (B*T, C)
            # - Similarly reshape targets from (B, T) to (B*T)
            # - Return logits in shape (B, T -1)
            ############################################################################

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits,targets)

            ############################################################################
            # END TODO 6
            ############################################################################
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            ############################################################################
            # TODO 7: Implement the generation loop:
            # 1. Crop idx to last block_size tokens
            # 2. Get predictions from model
            # 3. Focus only on last time step
            # 4. Apply softmax to get probabilities
            # 5. Sample from distribution
            # 6. Append sampled token to sequence
            
            # Hints:
            # - Crop the context to the last block_size tokens: idx[:, -self.block_size:]
            # - Get predictions with self(idx_cond) which returns logits and loss
            # - Select only the last timestep predictions: logits[:, -1, :]
            # - Convert to probabilities with F.softmax
            # - Sample from distribution with torch.multinomial
            # - Concatenate to existing sequence with torch.cat
            ############################################################################

            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)

            ############################################################################
            # END TODO 7
            ############################################################################
            
        return idx