import torch
import torch.nn.functional as F
from model import GPT
from tokenizer import CharTokenizer

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def train_model(model, data, optimizer, batch_size, block_size, device, epochs=20):
    print("Training...")
    model.train()
    for epoch in range(epochs):
        x, y = get_batch(data, batch_size, block_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} loss: {loss.item():.4f}")
    print("Training complete.") 