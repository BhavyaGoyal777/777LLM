#!/usr/bin/env python

import torch
import tiktoken
import PyPDF2
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# Assuming GPTModel and GPT_CONFIG_124M are defined in chap4.py
# from chap4 import GPTModel  # Uncomment if chap4.py is available and contains GPTModel
# from chap4 import GPT_CONFIG_124M # Uncomment if chap4.py is available and contains GPT_CONFIG_124M

# If chap4.py is not available, define GPT_CONFIG_124M here:
GPT_CONFIG_124M={
'vocab_size': 50257,
 'context_length': 256,
 'emb_dim': 768,
 'n_heads': 12,
 'n_layers': 12,
 'drop_rate': 0.1,
 'bias_': False}


def text_to_token_ids(text, tokenizer):
    """Encodes text into token IDs using the given tokenizer."""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(input_ids, tokenizer):
    """Decodes token IDs back into text using the given tokenizer."""
    flat = input_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def extract_text_from_pdf(pdf_path, footer_lines=2):
    """Extracts text from a PDF file, removing footer content."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    lines = page_text.split("\n")
                    cleaned_text = "\n".join(lines[:-footer_lines])  # Remove last few lines
                    text += cleaned_text + "\n\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

class GPTDatasetV1(Dataset):
    """Dataset for GPT model, creating input and target token sequences."""
    def __init__(self, text, tokenizer, max_length, stride):
        super().__init__()
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=267,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    """Creates a DataLoader for GPT dataset."""
    bpe_tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, bpe_tokenizer, max_length, stride)
    dataloader = DataLoader(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)
    return dataloader

def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculates cross-entropy loss for a batch."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader_data(data_loader, model, device, num_batches=None):
    """Calculates average loss over a DataLoader."""
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_dataloader, val_dataloader, eval_iter, device):
    """Evaluates model loss on train and validation dataloaders."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader_data(
            train_dataloader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader_data(
            val_dataloader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    """Generates text sample and prints it."""
    model.eval()
    context_size = model.pos_emb.weight.shape[0] # Assuming pos_emb is a member of model
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple( # Assuming generate_text_simple is defined elsewhere or chap4
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def train_model_simple(model, train_dataloader, val_dataloader,
                       optimizer, device, num_epochs, eval_freq,
                       eval_iter, start_context, tokenizer):
    """Simple training loop for GPT model."""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model=model, device=device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_dataloader, val_dataloader, eval_iter, device
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f'Ep {epoch+1} (step {global_step:06d}): '
                      f'Train loss {train_loss:.3f} | Val loss {val_loss:.3f}')

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, num_epochs):
    """Plots training and validation losses."""
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()  # 1
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 2
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

def softmax_with_temp(logits, temp):
    """Applies softmax with temperature scaling."""
    scaled_logits = logits / temp
    return torch.softmax(scaled_logits, dim=-1)

def print_sampled_tokens(probas, inverse_vocab):
    """Samples tokens based on probabilities and prints frequency."""
    torch.manual_seed(123)
    sample = [
        torch.multinomial(probas, num_samples=1).item()
        for _ in range(1000)
    ]
    sample_ids = torch.bincount(torch.tensor(sample))
    print(sample_ids.shape)
    for i, freq in enumerate(sample_ids):
        if i in inverse_vocab:
            print(f"{freq} x {inverse_vocab[i]}")
        else:
            print(f"{freq} x Token ID {i}") # Handle cases where token id is not in vocab

def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    """Generates text tokens from the model."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf'), device=logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and idx_next == eos_id: # Check for eos_id
            break
        idx = torch.cat((idx, idx_next), dim=-1)

    return idx

def assign(left, right):
    """Assigns values from right tensor to left parameter, ensuring shape match."""
    if left.shape != right.shape:
        raise ValueError(
            f'Shape mismatch. Left: {left.shape}, Right: {right.shape}'
        )
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    """Loads weights from a parameter dictionary into a GPT model."""
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):
        attention = gpt.trf_blocks[b].attention
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        attention.W_query.weight = assign(
            attention.W_query.weight, q_w.T)
        attention.W_key.weight = assign(
            attention.W_key.weight, k_w.T)
        attention.W_value.weight = assign(
            attention.W_value.weight, v_w.T)
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        attention.W_query.bias = assign(
            attention.W_query.bias, q_b)
        attention.W_key.bias = assign(
            attention.W_key.bias, k_b)
        attention.W_value.bias = assign(
            attention.W_value.bias, v_b)
        attention.out_proj.weight = assign(
            attention.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        attention.out_proj.bias = assign(
            attention.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

# Placeholder for generate_text_simple if it's not defined in chap4.py or elsewhere in your script
def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Placeholder for simple text generation function.
       Replace with your actual implementation if available in chap4.py"""
    generated_tokens = []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=-1)
        generated_tokens.append(idx_next)
    return idx


