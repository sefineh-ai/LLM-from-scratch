import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import string

class CausalAttention(nn.Module):
    def __init__(self, d_model, T, n_heads, dropout=0.1):
        super(CausalAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(1, 1, T, T)))
    def forward(self, x): 
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        d_head = self.d_model // self.n_heads
        B, T, C = x.size() 
        q, k, v = self.qkv(x).split(self.d_model, dim = 2)
        q = q.view(B, T, self.n_heads, d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, d_head).transpose(1, 2)
        attn = q@k.transpose(-2, -1)* (1.0/math.sqrt(d_head))
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim = -1)
        attn = self.dropout(attn)
        attn = attn@v
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.proj(attn)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))
    def forward(self, x):
        return F.layer_norm(x, (x.size(-1),), self.g, self.b, self.eps)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, T, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = CausalAttention(d_model, T, n_heads, dropout)
        self.ln1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln2 = LayerNorm(d_model)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, T, n_heads, n_layers, d_ff, dropout=0.1):
        super(GPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)  # (vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, T, d_model))  # (1, T, d_model)
        self.T = T  # Store max sequence length
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, T, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    def forward(self, idx):
        B, T = idx.size()
        if T > self.T:
            raise ValueError(f"Input sequence length {T} exceeds model's max positional embedding length {self.T}.")
        tok_emb = self.token_embedding(idx)  # (B, T, d_model)
        pos_emb = self.pos_embedding[:, :T, :]  # (1, T, d_model)
        # Debug prints
        print(f"token embedding output shape: {tok_emb.shape}")
        print(f"positional embedding shape: {pos_emb.shape}")
        x = tok_emb + pos_emb  # (B, T, d_model)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# Training and Sampling

def train(model, data, epochs, batch_size, learning_rate, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    T = 32
    for epoch in range(epochs):
        for batch in data:
            idx = batch[:, :T]
            idx_cond = idx[:, :-1]
            idx_pred = idx[:, 1:]
            logits = model(idx_cond)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), idx_pred.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} loss: {loss.item()}")

def sample(model, context, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(context)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)
    return context

# Data Loading

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def get_batch_data(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))   
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Hyperparameters

batch_size = 16
block_size = 256
d_model = 64
n_heads = 2
d_ff = 128
n_layers = 2
dropout = 0.1
vocab_size = 100  # For toy example

# --- Character-level tokenizer utilities ---
def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    return [stoi[c] for c in text]

def decode(indices, itos):
    return ''.join([itos[i] for i in indices])

# --- Sample conversational data ---
# Load data from file
with open('sample_chat_corpus.txt', 'r', encoding='utf-8') as f:
    sample_text = f.read()

# --- Chatbot sampling function ---
def generate_response(model, context, stoi, itos, max_new_tokens=100, temperature=1.0, device='cpu'):
    model.eval()
    idx = torch.tensor([encode(context, stoi)], dtype=torch.long).to(device)
    for _ in range(max_new_tokens):
        logits = model(idx)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1).unsqueeze(0)  # Greedy
        idx = torch.cat((idx, next_token), dim=1)
        if itos[next_token.item()] == '\n':
            break
    return decode(idx[0].tolist(), itos)

def main():
    # Build vocab and encode data
    stoi, itos = build_vocab(sample_text)
    vocab_size = len(stoi)
    data = torch.tensor(encode(sample_text, stoi), dtype=torch.long)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Model hyperparameters
    batch_size = 16
    block_size = 256
    d_model = 64
    n_heads = 2
    d_ff = 128
    n_layers = 2
    dropout = 0.1
    model = GPT(vocab_size, d_model, block_size, n_heads, n_layers, d_ff, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Training loop
    for epoch in range(20):
        x, y = get_batch(data, batch_size, block_size)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} loss: {loss.item():.4f}")
    # --- Chat simulation ---
    print("\n--- Chat Simulation ---")
    while True:
        prompt = input("You: ")
        if prompt.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        # Add a newline to simulate end of user input
        context = prompt + '\n'
        response = generate_response(model, context, stoi, itos, max_new_tokens=100, device=device)
        # Remove the prompt from the response
        bot_reply = response[len(context):].strip()
        print(f"AI: {bot_reply}")

if __name__ == "__main__":
    main()
