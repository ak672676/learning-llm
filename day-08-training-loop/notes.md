# Day 08: The Training Loop — Loss Functions, Optimizers, Learning Rate

## Why This Matters

You've trained simple things before: linear regression, single neuron, XOR network. The training loop has been roughly the same:

```python
y_pred = model(X)
loss = loss_fn(y_pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

But there's a LOT happening in those 5 lines. Today we'll dig into:
1. **Loss functions** — different ones for different tasks
2. **Optimizers** — SGD, Adam, AdamW, why each exists
3. **Learning rate scheduling** — adjusting LR during training
4. **Mini-batching** — splitting data into chunks
5. **Building a robust trainer** — code you can reuse for ANY model

This is the lap before we tackle real text data (Day 10+).

---

## 1. Loss Functions — The Right Tool for Each Job

Every model needs to know "how wrong am I?" That's the loss. Different tasks need different loss functions.

### Mean Squared Error (MSE) — for regression

```
MSE = mean((y_pred - y_true)²)
```

When predicting CONTINUOUS values (price, temperature, score), use MSE.

```python
loss = torch.nn.functional.mse_loss(y_pred, y_true)
# Or:
loss = ((y_pred - y_true) ** 2).mean()
```

### Binary Cross-Entropy (BCE) — for yes/no classification

```
BCE = -[y * log(ŷ) + (1-y) * log(1-ŷ)]
```

When predicting a SINGLE 0/1 label (spam/not, cat/dog), use BCE. Punishes confident-but-wrong predictions heavily.

```python
loss = torch.nn.functional.binary_cross_entropy(y_pred, y_true)
# Or with logits (numerically stable, preferred):
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y_true)
```

### Cross-Entropy (CE) — for multi-class classification

```
CE = -sum(y_true * log(softmax(z)))
```

When predicting ONE class out of MANY (digits 0-9, ImageNet, next word), use CE. **This is what LLMs use** — predicting the next token from a vocabulary of 50,000.

```python
loss = torch.nn.functional.cross_entropy(z, y_true)
# z = raw logits (no softmax!)
# y_true = class indices (long integers, not one-hot)
```

### Quick reference

| Task | Output | Loss | PyTorch |
|------|--------|------|---------|
| Regression (predict number) | Single number | MSE | `mse_loss` |
| Binary classification | 0 or 1 | BCE | `binary_cross_entropy_with_logits` |
| Multi-class classification | One of K classes | Cross-Entropy | `cross_entropy` |
| LLM next-token prediction | Probabilities over vocab | Cross-Entropy | `cross_entropy` |

---

## 2. Optimizers — How Weights Get Updated

The optimizer takes gradients and updates weights. Different optimizers do this in different ways.

### SGD (Stochastic Gradient Descent) — the simplest

```python
weight = weight - lr * gradient
```

Pure subtraction. Simple and works, but can be slow.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### SGD with Momentum — accumulates "velocity"

```
velocity = 0.9 * velocity + gradient
weight = weight - lr * velocity
```

Like a ball rolling downhill — picks up speed in consistent directions, dampens oscillations.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### Adam — the all-around favorite

Adam adapts the learning rate **per parameter** based on past gradients. Goes faster for stable directions, slower for noisy ones.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Use Adam for almost everything.** It "just works."

### AdamW — Adam with proper weight decay (used for LLMs)

Adam, but with weight decay (L2 regularization) applied correctly. **This is what GPT-2, GPT-3, Llama all use.**

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### Quick reference

| Optimizer | When to use | Default LR |
|-----------|-------------|-----------|
| **SGD** | Simple problems, when you want explicit control | 0.01-0.1 |
| **SGD + momentum** | Computer vision (ResNet etc.) | 0.01-0.1, momentum=0.9 |
| **Adam** | Default for almost anything | 0.001 |
| **AdamW** | Transformers, LLMs | 0.0001-0.001 |

---

## 3. Learning Rate Scheduling

The learning rate matters more than any other hyperparameter. But the BEST learning rate isn't constant — it should change during training.

### Why?

- **Early training:** weights are random, big steps help (high LR)
- **Late training:** approaching the minimum, big steps overshoot (need low LR)

### Common schedules

#### Step decay
Drop LR by some factor every N epochs:
```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# Drop LR by 10x every 30 epochs
```

#### Cosine annealing
LR follows a cosine curve from initial to ~0:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# Smooth decrease over 100 epochs
```

This is what most modern LLMs use.

#### Warmup + cosine
LR increases linearly for a few steps, then decreases. Used by GPT, Llama, etc.

```
LR
 ↑
 |   ___
 |  /   \___
 | /        \___
 |/             \___
 +--------------------→ steps
   warmup    cosine decay
```

### Using a scheduler

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    train_one_epoch(...)
    scheduler.step()      # update LR
    print(f"LR is now: {scheduler.get_last_lr()}")
```

---

## 4. Mini-Batching

So far we've trained on the WHOLE dataset at once. Real training uses **mini-batches** — small chunks of data processed one at a time.

### Why mini-batches?

1. **Memory:** can't fit millions of samples in GPU memory at once
2. **Faster training:** more updates per epoch
3. **Noise helps:** stochastic updates escape local minima better

### Common batch sizes

| Task | Batch size |
|------|-----------|
| Tiny experiments | 4-32 |
| Vision models | 32-256 |
| LLM training | 256-2048 (per GPU; total can be huge) |

### Standard pattern

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        # ONE update per batch (not per epoch!)
        y_pred = model(batch_x)
        loss = loss_fn(y_pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

For 10000 samples with batch_size=32, you do 312 updates per epoch. Each update is faster, and you converge in fewer epochs.

---

## 5. Anatomy of a Production Training Loop

Real training loops do more than the basic 5 lines:

```python
def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()                  # set to training mode (enables dropout etc.)
    total_loss = 0
    
    for batch_x, batch_y in train_loader:
        # Move data to device (GPU/CPU)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        y_pred = model(batch_x)
        loss = loss_fn(y_pred, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, loss_fn, device):
    model.eval()                    # set to eval mode (disables dropout)
    total_loss = 0
    
    with torch.no_grad():           # don't track gradients during eval
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)
```

### Key new things:
- **`model.train()` / `model.eval()`** — toggles dropout/batchnorm behavior
- **`.to(device)`** — moves tensors to GPU
- **`with torch.no_grad():`** — disables gradient tracking (saves memory)
- **Batch-level loop** — one update per batch, not per epoch

---

## 6. Saving and Loading Models (Checkpointing)

After training for hours, you want to save your work!

```python
# Save the model + optimizer state
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Load it back
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

### Best practice: save the BEST model
```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), 'best_model.pth')
```

Don't keep training a worsening model — save the version that did best on validation, even if you train longer.

---

## 7. Early Stopping

Stop training when validation loss stops improving:

```python
patience = 10
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_loss = evaluate(...)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

Saves time and prevents overfitting.

---

## Mental Model

```
The training loop is just a pipeline:

  Dataset → DataLoader → Batches
                          ↓
                       [model] → predictions
                          ↓
                       [loss] → number
                          ↓
                       [autograd] → gradients
                          ↓
                       [optimizer] → updated weights
                          ↓
              (repeat thousands of times)

Tools you toggle:
  - Loss function: matches your task
  - Optimizer: how aggressively to update
  - Learning rate: step size (most important!)
  - Batch size: how much data per update
  - Scheduler: change LR over time
  - Early stopping: when to call it done
```

---

## Summary

| Concept | Key Idea |
|---------|----------|
| **MSE** | Regression (predict numbers) |
| **BCE** | Binary classification (yes/no) |
| **Cross-Entropy** | Multi-class classification (LLMs use this!) |
| **SGD** | Simplest optimizer: weight -= lr * gradient |
| **Adam** | Adapts LR per parameter — default for most tasks |
| **AdamW** | Adam + weight decay — what LLMs use |
| **Scheduler** | Changes LR during training (warmup + cosine for LLMs) |
| **Mini-batch** | Small chunks of data per update (memory + speed) |
| **train()/eval()** | Toggle modes — dropout/batchnorm differ |
| **Checkpointing** | Save model state to disk |
| **Early stopping** | Stop when val loss stops improving |

**Tomorrow:** We'll rebuild Day 7's XOR network using PyTorch's `nn.Module` properly, with all the production features we learned today.
