# Day 12: Project — Complete Sentiment Classifier

## Why This Matters

You've now learned the building blocks separately:

```
Day 4:    Tokenization
Day 8:    Training loop, optimizers, LR scheduling, checkpointing
Day 9:    nn.Module, layers, dropout
Day 10:   Bag of Words classifier
Day 11:   Word embeddings
```

Today: bring them ALL together into a single, polished project. This is the kind of thing you'd put on GitHub or use in production.

The goal isn't to learn new concepts — it's to **integrate** what you know into something that works end-to-end.

---

## What We'll Build

A complete sentiment classifier with:

1. **A larger, more realistic dataset** (~200 reviews instead of 30)
2. **Proper data pipeline** — Dataset, DataLoader, batches
3. **Three different models** (compared head-to-head):
   - BoW + Linear (baseline from Day 10)
   - Embedding + Mean Pool + Linear (Day 11 style)
   - Embedding + MLP (deeper)
4. **Production training loop** — train/val/test, batches, scheduler, early stopping
5. **Saved model artifacts** — weights + vocab + config
6. **Inference function** — load and use the model on new text
7. **Error analysis** — see WHICH reviews the model gets wrong

---

## The Production Pipeline

```
                                                                    
┌─────────┐    ┌────────────┐    ┌──────────┐    ┌─────────────┐
│  Raw    │ →  │  Tokenize  │ →  │  Encode  │ →  │  Pad/Trunc  │
│  text   │    │  to tokens │    │  to IDs  │    │ to fixed len │
└─────────┘    └────────────┘    └──────────┘    └─────────────┘
                                                          │
                                                          ▼
┌─────────┐    ┌────────────┐    ┌──────────┐    ┌─────────────┐
│ Predict │ ←  │   Linear   │ ←  │   Mean   │ ←  │  Embedding  │
│  class  │    │ classifier │    │  pool    │    │   lookup    │
└─────────┘    └────────────┘    └──────────┘    └─────────────┘
```

Each step is a clear responsibility. Real ML projects keep these separate.

---

## Project Structure

A real project would have these files:

```
project/
├── notes.md                    (you're reading this)
├── notebook.ipynb              (interactive notebook for development)
├── code/
│   ├── tokenizer.py            (vocab building + encode/decode)
│   ├── model.py                (the nn.Module classes)
│   ├── train.py                (training script)
│   └── predict.py              (inference script)
└── artifacts/
    ├── model.pth               (saved weights)
    ├── vocab.pkl               (saved vocabulary)
    └── config.json             (hyperparameters)
```

For this day's work, we'll put everything in the notebook for easy exploration. But you'll see how to organize it for a real project.

---

## Why Compare 3 Models?

When building a real ML system, you should always have:
- A **baseline** model (simple)
- A **target** model (what you're aiming for)
- A **stretch** model (more complex, see if it actually helps)

For us:
- **Baseline:** BoW + Linear (what works without embeddings)
- **Target:** Embedding + Mean + Linear (the obvious upgrade)
- **Stretch:** Embedding + MLP (does adding hidden layers help?)

This pattern — "always compare against a simple baseline" — is one of the most important habits in ML.

---

## Error Analysis — The Often-Skipped Step

Looking at where your model FAILS is more useful than looking at where it succeeds.

For sentiment, common failure modes:
- **Negation:** "not bad" → BoW thinks "bad" → wrongly predicts negative
- **Sarcasm:** "Oh great, another sequel" → model can't tell tone
- **Mixed reviews:** "Acting was good but plot was terrible"
- **Unknown words:** new movie titles, slang

By systematically inspecting failures, you learn what to fix next.

---

## The "Confidence" Calibration Question

When a model says "85% confident this is positive," is it actually right 85% of the time?

We won't fully tackle this today, but it's good to know:
- **Well-calibrated:** confidence predicts actual accuracy
- **Overconfident:** says 99% but is right 80%
- **Underconfident:** says 60% but is right 90%

Modern LLMs are often **overconfident** — they say things very confidently even when wrong.

---

## What Makes This a "Project"

Compared to previous notebooks, today is different in feel:
- **Longer code** — fewer toy snippets, more real flow
- **Reusable** — built so you can adapt it to OTHER text classification tasks
- **Documented** — every function has a docstring
- **Persistent** — model gets saved properly
- **Testable** — predictions can be made on arbitrary new text

Treat this as a template you'll adapt for other projects (spam detection, topic classification, intent classification, etc.).

---

## Summary

| Skill | Where it came from |
|-------|---------------------|
| Tokenization | Day 4 |
| Training loop with batches | Day 8 |
| Optimizer + scheduler | Day 8 |
| nn.Module + nn.Embedding | Day 9, 11 |
| Embeddings as features | Day 11 |
| Cross-Entropy loss | Day 8 |
| Train/val/test methodology | Day 5, 8 |
| Save/load checkpoints | Day 8, 9 |
| Error analysis | NEW today |
| Side-by-side model comparison | NEW today |

### Going forward:

```
Day 1-12:  Foundations + first real text classifier   ← YOU ARE HERE
Day 13:    Sequence models (RNN) — finally respect order!
Day 14:    Bigram language model
Day 15-18: Attention mechanism — the heart of transformers
Day 19-25: Build a real GPT
```

Today caps off the "feed-forward + embeddings" era. Starting Day 13, we step into sequence modeling — where the order of words matters. That's the path to real LLMs.
