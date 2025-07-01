# LLM-from-scratch

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-blue?logo=pytorch)](https://pytorch.org/) [![License: Public Domain](https://img.shields.io/badge/license-Public%20Domain-brightgreen.svg)](https://www.gutenberg.org/policy/license.html)

> **A minimal, educational GPT-style next-token generator built from scratch in PyTorch.**

---

## Table of Contents
- [LLM-from-scratch](#llm-from-scratch)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Dataset](#dataset)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training](#training)
    - [Chat/Generation](#chatgeneration)
  - [Customization](#customization)
  - [Example Output](#example-output)
  - [Contributing](#contributing)
  - [License \& Credits](#license--credits)

---

## Overview

**LLM-from-scratch** is a simple, readable implementation of a GPT-style character-level language model. It is designed for learning, experimentation, and as a starting point for research. The model can be trained on any plain text and used for next-token (auto-regressive) text generation.

---

## Features
- Transformer-based architecture (multi-head causal self-attention)
- Character-level tokenization (no external dependencies)
- Next-token generation (auto-regressive)
- Simple training and interactive chat/generation interface
- Easily extensible for research and learning

---

## Dataset

The default dataset is [Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11) from Project Gutenberg, but you can use any plain text file by replacing `sample_chat_corpus.txt`.

---

## Requirements
- Python 3.8+
- PyTorch 2.x

---

## Installation

1. **Clone the repository:**
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

---

## Usage

### Training

The model will automatically train on the text in `sample_chat_corpus.txt`:
```bash
python3 model.py
```

### Chat/Generation

After training, the script enters interactive mode. Type a prompt and the model will generate a continuation:
```
You: Once upon a time
AI:  there was a little girl named Alice who...
```
Type `exit` or `quit` to end the session.

---

## Customization
- **Change the dataset:** Replace `sample_chat_corpus.txt` with any plain text file.
- **Adjust model/training parameters:** Edit `model.py` to change model size, batch size, epochs, etc.
- **Experiment:** Try different sampling parameters (`temperature`, `top_k`) for more creative or conservative outputs.

---

## Example Output

```
You: The rabbit
AI:  was running very fast and Alice followed him down the hole.
```

---

## Contributing

Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request.

---

## License & Credits

- **Code:** MIT License or Public Domain (choose your preferred open license)
- **Dataset:** [Project Gutenberg, Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11) (public domain)

---

**Created by [Your Name or GitHub handle]** 