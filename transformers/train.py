import torch
import matplotlib.pyplot as plt
from pathlib import Path
from data_utils import download_shakespeare, load_shakespeare, get_batch, estimate_loss
from transformer_model import BigramLanguageModel

def train(hyperparams=None):
    # Default hyperparameters
    hp = {
        'batch_size': 16,
        'block_size': 32,
        'max_iters': 5000,
        'eval_interval': 100,
        'learning_rate': 1e-3,
        'n_embd': 64,
        'n_head': 4,
        'n_layer': 4
    }
    
    # Update with provided hyperparameters
    if hyperparams is not None:
        hp.update(hyperparams)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(1337)
    
    # Load data
    file_path = download_shakespeare()
    train_data, val_data, encode, decode, vocab_size = load_shakespeare(file_path)
    
    # Initialize model
    model = BigramLanguageModel(
        vocab_size=vocab_size,
        n_embd=hp['n_embd'],
        block_size=hp['block_size'],
        n_head=hp['n_head'],
        n_layer=hp['n_layer']
    ).to(device)
    
    # Print model size
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Generate text before training
    print("\nGeneration before training:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp['learning_rate'])
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for iter in range(hp['max_iters']):
        if iter % hp['eval_interval'] == 0:
            losses = estimate_loss(model, train_data, val_data, 
                                hp['batch_size'], hp['block_size'])
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Sample batch and get loss
        xb, yb = get_batch(train_data, hp['batch_size'], hp['block_size'], device)
        logits, loss = model(xb, yb)
        
        # Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.title('Training and Validation Loss')
    plt.xlabel('Steps (x100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    
    # Generate text after training
    print("\nGeneration after training:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    
    return model, (train_losses, val_losses)

if __name__ == '__main__':
    model, losses = train()