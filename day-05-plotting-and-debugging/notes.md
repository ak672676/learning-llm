# Day 05: Plotting & Debugging — Reading Loss Curves Like a Pro

## Why This Matters

You're about to start training real models. When training goes wrong (and it will), the **loss curve** is your detective. A loss curve tells you:

- Is the model learning at all?
- Is the learning rate too high/low?
- Is the model memorizing instead of learning?
- Should I stop training or keep going?

Reading loss curves well is a skill that separates beginners from people who actually ship working ML. Today we'll build that skill.

---

## 1. The Training Loop with Monitoring

A real training loop tracks two losses:

```python
for epoch in range(num_epochs):
    # 1. Train step
    train_loss = train_one_epoch(model, train_loader)
    
    # 2. Validation step (no gradient updates!)
    with torch.no_grad():
        val_loss = evaluate(model, val_loader)
    
    # 3. Track both
    train_losses.append(train_loss)
    val_losses.append(val_loss)
```

**Why two losses?**

- **Training loss** — how wrong the model is on data it's SEEN
- **Validation loss** — how wrong it is on data it HASN'T seen

Both should go down. If training goes down but validation goes up → the model is memorizing (overfitting).

---

## 2. What Loss Curves Look Like — A Visual Guide

### Healthy training

```
Loss ↑
     |  \
     |   \
     |    \_____ 
     |         \____
     |              \___train  (both dropping together)
     |              -----val    (val slightly higher)
     +----------------------→ Epoch
```

Both losses drop together, level off. **This is what you want.**

### Underfitting (model too simple)

```
Loss ↑
     |  \
     |   \___
     |       \________________  (levels off HIGH — model can't learn more)
     |         ---------------- (train == val, both high)
     +----------------------→ Epoch
```

Loss stops dropping at a high value. Solution: bigger model, more features, more data.

### Overfitting (model memorizing)

```
Loss ↑
     |  \                 ___
     |   \              _/        val loss going UP!
     |    \           _/
     |     \________/              train loss still dropping
     |              \______
     +----------------------→ Epoch
                    ↑
              "Sweet spot" — stop here!
```

Training loss keeps dropping but validation goes UP. The model is memorizing. Solution: stop training earlier, get more data, regularize.

### Learning rate too high

```
Loss ↑
     | /\  /\  /\  /\     (bouncing wildly)
     |/  \/  \/  \/  \
     |                \   
     +----------------------→ Epoch
```

Loss oscillates instead of dropping smoothly. Solution: decrease learning rate.

### Learning rate too low

```
Loss ↑
     |-\
     |   \                  (barely moving)
     |     \_
     |       \_
     |         \_
     +----------------------→ Epoch  (would take FOREVER to converge)
```

Loss drops very slowly. Solution: increase learning rate.

### Broken (loss is NaN or Inf)

```
Loss = NaN → STOP TRAINING
```

Almost always means:
- Learning rate way too high (try 10x smaller)
- Exploding gradients (need gradient clipping)
- Division by zero somewhere in the model
- Log of 0 somewhere

---

## 3. The Average Running Loss Trick

During training, we update weights after EVERY batch. If you plot raw batch losses, they look noisy:

```
Batch losses: [4.2, 3.8, 4.5, 3.2, 5.1, 2.9, 4.0, ...]   ← jumps around
```

To see the trend, we plot a **running average** (smoothed loss):

```python
# Simple moving average over last 50 batches
smooth_loss = running_mean(losses, window=50)
```

Rule of thumb: always smooth your loss curves before analyzing them.

---

## 4. Common Training Bugs and How to Spot Them

### Bug 1: Loss is NaN
- **Cause:** numerical explosion (exp of huge number, division by zero, log of negative)
- **Fix:** lower learning rate, gradient clipping, check model output for issues

### Bug 2: Loss doesn't change at all
- **Cause:** gradients not flowing (forgot `.backward()`, `requires_grad=False`, disconnected graph)
- **Fix:** print `param.grad` — if it's None or 0, gradients aren't flowing

### Bug 3: Loss INCREASES
- **Cause:** learning rate too high, or bug in loss direction (sign flip)
- **Fix:** decrease LR by 10x, check loss formula

### Bug 4: Loss is always the same random value
- **Cause:** `optimizer.zero_grad()` missing (gradients accumulating)
- **Fix:** add `optimizer.zero_grad()` before `loss.backward()`

### Bug 5: Training loss goes down, validation explodes
- **Cause:** overfitting
- **Fix:** stop training earlier, get more data, add dropout/regularization

---

## 5. Useful matplotlib Techniques for ML

### Multiple subplots side by side
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train_losses)
ax2.plot(val_losses)
```

### Log scale for loss curves
When loss drops from 10 → 0.01, a linear y-axis hides the later improvements. Use log scale:
```python
plt.yscale('log')
```

### Compare training runs
```python
plt.plot(losses_lr001, label='lr=0.001')
plt.plot(losses_lr01,  label='lr=0.01')
plt.plot(losses_lr1,   label='lr=0.1')
plt.legend()
```

### Grid and style
```python
plt.grid(True, alpha=0.3)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
```

---

## 6. The ML Debugging Checklist

When training misbehaves, check these in order:

1. **Is the loss decreasing at all?**
   - No → gradients aren't flowing (check requires_grad, zero_grad, backward)

2. **Is it NaN/Inf?**
   - Yes → numerical issue (LR too high, exp/log problems)

3. **Are training and validation losses diverging?**
   - Yes → overfitting (stop earlier, more data, regularize)

4. **Is the loss curve smooth or jumpy?**
   - Jumpy → LR too high, or batch size too small

5. **Did loss plateau too early?**
   - Yes → LR too low, model too small, or stuck in local minimum

6. **Does the model produce reasonable outputs?**
   - Check predictions on a few samples. Don't just look at numbers.

---

## Mental Model

Think of training like watching a plant grow:

- **Healthy curve:** steady growth, levels off when mature
- **Underfitting:** plant is stunted — needs more light/nutrients (bigger model/data)
- **Overfitting:** plant is growing but leaves are wilting — you're over-watering (too much training)
- **NaN/Inf:** plant died (numerical explosion)

The loss curve is the plant's health chart. Learn to read it.

---

## Summary

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss is NaN | LR too high, numerical blowup | LR × 0.1, gradient clip |
| Loss doesn't change | No gradients flowing | Check backward(), requires_grad |
| Loss increases | LR too high, sign flip | LR × 0.1, check loss formula |
| Loss stuck at same value | Missing zero_grad() | Add optimizer.zero_grad() |
| Jagged loss curve | LR slightly too high, small batches | Lower LR, bigger batch size |
| Val loss going up while train drops | Overfitting | Early stop, more data, regularize |
| Both losses plateau high | Underfitting | Bigger model, more features |

**Tomorrow:** We start building actual neural networks — a single neuron learning logic gates (Day 06).
