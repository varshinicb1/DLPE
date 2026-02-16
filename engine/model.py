import torch
import torch.nn as nn
from torch.nn import functional as F

# KissanGPT-Nano Config
# Small enough to train on CPU/Edge for demo, scalable for GPU
class KissanConfig:
    vocab_size = 5000  # Matches the saved checkpoint
    n_embd = 256       # Embedding dimension
    n_head = 4         # Attention heads
    n_layer = 4        # Transformer layers
    block_size = 64    # Context window (sequence length)
    dropout = 0.1
    device = 'cpu'     # Force CPU for local brain compatibility

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(KissanConfig.n_embd, head_size, bias=False)
        self.query = nn.Linear(KissanConfig.n_embd, head_size, bias=False)
        self.value = nn.Linear(KissanConfig.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(KissanConfig.block_size, KissanConfig.block_size)))
        self.dropout = nn.Dropout(KissanConfig.dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,16)
        q = self.query(x) # (B,T,16)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, 16) @ (B, 16, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,16)
        out = wei @ v # (B, T, T) @ (B, T, 16) -> (B, T, 16)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(KissanConfig.n_embd, KissanConfig.n_embd)
        self.dropout = nn.Dropout(KissanConfig.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(KissanConfig.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class AgiCoreModel(nn.Module):
    """ The Kissan Neural Network"""

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, KissanConfig.n_embd)
        self.position_embedding_table = nn.Embedding(KissanConfig.block_size, KissanConfig.n_embd)
        self.blocks = nn.Sequential(*[Block(KissanConfig.n_embd, n_head=KissanConfig.n_head) for _ in range(KissanConfig.n_layer)])
        self.ln_f = nn.LayerNorm(KissanConfig.n_embd) # final layer norm
        self.lm_head = nn.Linear(KissanConfig.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -KissanConfig.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
