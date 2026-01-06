import torch
from pathlib import Path
from urllib.request import urlopen

def download_shakespeare():
    """Download the tiny Shakespeare dataset"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    file_path = data_dir / "input.txt"
    if not file_path.exists():
        with urlopen(url) as response:
            content = response.read().decode('utf-8')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    return file_path

def load_shakespeare(file_path, include_test=False):
    """Load and preprocess the Shakespeare dataset
    
    Args:
        file_path: Path to the Shakespeare text file
        include_test: If True, returns train/val/test split; otherwise train/val
    
    Returns:
        If include_test=False: (train_data, val_data, encode, decode, vocab_size)
        If include_test=True: (train_data, val_data, test_data, encode, decode, vocab_size)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Get unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create encoding/decoding mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Create encoder and decoder functions
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Create train/val/test split (70%/15%/15%)
    data = torch.tensor(encode(text), dtype=torch.long)
    
    if include_test:
        train_size = int(0.7 * len(data))
        val_size = int(0.15 * len(data))
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size+val_size]
        test_data = data[train_size+val_size:]
        
        return train_data, val_data, test_data, encode, decode, vocab_size
    else:
        # Original 90/10 split for backward compatibility
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]
        
        return train_data, val_data, encode, decode, vocab_size

def get_batch(data, batch_size, block_size, device):
    """Generate a small batch of data of inputs x and targets y"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters=200, test_data=None):
    """Estimate loss on train, validation, and optionally test sets"""
    out = {}
    model.eval()
    
    data_splits = [('train', train_data), ('val', val_data)]
    if test_data is not None:
        data_splits.append(('test', test_data))
        
    for split, data in data_splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, next(model.parameters()).device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
