import os
import torch
from model import GPT
from tokenizer import CharTokenizer
from train import train_model
from chat import chat_with_model

def main():
    data_path = 'sample_chat_corpus.txt'
    if not os.path.exists(data_path):
        print(f"Data file '{data_path}' not found. Please provide a sample_chat_corpus.txt file.")
        return
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyperparameters
    batch_size = 16
    block_size = 256
    d_model = 64
    n_heads = 2
    d_ff = 128
    n_layers = 2
    dropout = 0.1
    model = GPT(tokenizer.vocab_size, d_model, block_size, n_heads, n_layers, d_ff, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_model(model, data, optimizer, batch_size, block_size, device, epochs=20)
    chat_with_model(model, tokenizer, device)

if __name__ == "__main__":
    main() 