# Build a LLM From Scratch

A PyTorch implementation of GPT-2 trained from scratch. This is a project where I built and trained the model from scratch based on PyTorch

## Components

- **`scr/attentionLayers.py`** — Multi-head masked self-attention
- **`scr/gptmodel.py`** — The full GPT model: token/position embeddings, transformer blocks, layer norm, GELU activation, and a linear output head
- **`scr/dataLoader.py`** — A simple dataset class that tokenizes text using the GPT-2 tokenizer and uses a sliding window to create input/target pairs
- **`scr/training.py`** — Training loop, loss calculation, and greedy text generation
- **`notebooks/main.ipynb`** — The notebook where I put everything together and ran the training
- **`training_data/the-verdict.txt`** — The text used for training

## High level overview

The model follows the standard GPT-2 architecture: token and positional embeddings are added together, passed through a stack of transformer blocks (each with pre-norm, multi-head attention, and a feedforward layer with residual connections), and projected to vocabulary logits. Training uses cross-entropy loss with the AdamW optimizer.
