# LLM-from-scratch

A minimal, educational implementation of a GPT-style next-token generator using PyTorch. This project demonstrates how to build, train, and interact with a character-level language model from scratch, suitable for learning and experimentation.

## Features
- Transformer-based architecture (multi-head causal self-attention)
- Character-level tokenization
- Next-token generation (auto-regressive text generation)
- Simple training and chat interface
- Easily extensible for research and learning

## Dataset
The default dataset is [Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11) from Project Gutenberg, but you can use any plain text file by replacing `sample_chat_corpus.txt`.

## Requirements
- Python 3.8+
- PyTorch 2.x

Install requirements (if using a virtual environment):
```bash
pip install torch
```

## Setup & Usage
1. **Clone the repository and enter the project directory:**
   ```bash
   git clone https://github.com/sefineh-ai/LLM-from-scratch.git
   cd LLM-from-scratch
   ```
2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install torch
   ```
4. **(Optional) Replace `sample_chat_corpus.txt` with your own plain text data.**

## Training & Chat
Run the main script to train the model and enter chat mode:
```bash
python3 model.py
```
- The model will train on the text in `sample_chat_corpus.txt`.
- After training, you can type a prompt and the model will generate a continuation.
- Type `exit` or `quit` to end the chat.

## Customization
- Adjust model size, batch size, and training epochs in `model.py` for better results.
- Replace the dataset with any plain text for different domains or languages.

## Credits
- Model and code: [Your Name or GitHub handle]
- Dataset: [Project Gutenberg, Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11)

## License
This project is for educational and research purposes. The included dataset is in the public domain. 