class CharTokenizer:
    """Character-level tokenizer."""
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
    def encode(self, text):
        return [self.stoi[c] for c in text]
    def decode(self, indices):
        return ''.join([self.itos[i] for i in indices])
    @property
    def vocab_size(self):
        return len(self.chars) 