import torch
from tokenizer import CharTokenizer
from model import GPT

def generate_response(model, tokenizer, prompt, device, max_new_tokens=100, temperature=0.8, top_k=10):
    model.eval()
    context = prompt + '\n'
    idx = torch.tensor([tokenizer.encode(context)], dtype=torch.long).to(device)
    for _ in range(max_new_tokens):
        logits = model(idx)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
        if tokenizer.itos[next_token.item()] == '\n':
            break
    response = tokenizer.decode(idx[0].tolist())
    return response[len(context):].strip()

def chat_with_model(model, tokenizer, device):
    print("\n--- Chat Simulation ---")
    print("Type your message and press Enter. Type 'exit' or 'quit' to stop.")
    while True:
        prompt = input("You: ")
        if prompt.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        bot_reply = generate_response(model, tokenizer, prompt, device)
        print(f"AI: {bot_reply}") 