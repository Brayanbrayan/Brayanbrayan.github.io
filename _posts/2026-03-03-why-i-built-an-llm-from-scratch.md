---
title: "LLM From Scratch: Architecture and Training Infrastructure"
date: 2026-07-16
categories: machine-learning deep-learning
---

# LLM From Scratch: Architecture and Training Infrastructure

*Personal project · 2026 · PyTorch · no pretrained weights · 2-layer, 2-head, 128-dim model*

---

I wrote a full report on the alignment side of this project earlier this year: SFT, PPO, GRPO, and DPO, trained on the same checkpoint and compared head to head. That post starts from an already-trained language model and asks what happens when you align it. This one is about how that model got built in the first place, and why some of the architecture decisions made in these early parts are the reason the alignment results were legible at all.

If you want the alignment story, it's [here](https://brayanbrayan.github.io/2026/04/02/rlhf-post-blog.html). This post is the five parts that came before it.

---

## Table of Contents

1. [Why build this in order](#1-why-build-this-in-order)
2. [Attention from first principles](#2-attention-from-first-principles)
3. [Training on real text](#3-training-on-real-text)
4. [Modernizing the architecture](#4-modernizing-the-architecture)
5. [Training infrastructure](#5-training-infrastructure)
6. [Mixture of Experts](#6-mixture-of-experts)
7. [What this hands off to the alignment study](#7-what-this-hands-off-to-the-alignment-study)
8. [What I'd do differently](#8-what-id-do-differently)

---

## 1. Why Build This in Order

The project is ten parts. This post covers the first five: the transformer itself, a working training loop, the architecture upgrades that bring it in line with what production models actually look like, the infrastructure to train at scale, and Mixture of Experts. Parts six through ten are SFT, reward modeling, PPO, GRPO, and DPO, and they're documented separately.

I built it in this order on purpose. Debugging multi-head attention is much easier once you've already built and tested a single head by hand. Debugging gradient accumulation is much easier once you already have a clean training loop without it. Each part produces something that runs before the next part starts, so when something breaks, the blast radius is small.

One decision from Part 2 shaped everything after it: the model that goes into the alignment study is small on purpose. Two transformer layers, two attention heads, 128-dimensional embeddings, roughly 1M parameters. That's not a stripped-down placeholder waiting to be scaled up. It's the config chosen so that SFT, PPO, GRPO, and DPO could each run to completion on a single Colab GPU in a reasonable amount of time, and so I could run enough phases and hyperparameter sweeps to actually see algorithmic behavior rather than a single snapshot. Part 3 shows what a larger 4-layer, 4-head, 256-dim config looks like as an architectural reference, but every part after that, including the entire alignment study, runs on the 2-2-128 model. Small on purpose, not small by accident.

---

## 2. Attention from First Principles

Before building multi-head attention, I built single-head attention by hand, with the dimensions written out at every step:

```
x:        (B, T, d_model)
Q, K, V:  each (B, T, d_k)   via Linear(d_model → d_k)
scores:   (B, T, T)          via (Q @ K^T) / sqrt(d_k)
mask:     upper triangle → -inf   (causal, no peeking at future tokens)
weights:  (B, T, T)          via softmax(scores)
output:   (B, T, d_k)        via weights @ V
```

The causal mask is the part worth sitting with. It's applied via `masked_fill(mask, float('-inf'))` before the softmax, so future positions get exactly zero attention weight after normalization. Get this wrong and the model can "see" tokens it's supposed to be predicting, and you won't necessarily notice from the loss curve. You'll just have a model that looks great during training and falls apart at inference, where there's no future to peek at.

Multi-head attention is mechanical once the single-head version is solid. A single `Linear(d_model, 3 * d_model)` projects Q, K, and V together, then a reshape splits them into heads:

```python
qkv = self.qkv_proj(x)                                # (B, T, 3 * d_model)
q, k, v = qkv.view(B, T, 3, n_head, d_head).unbind(dim=2)
q = q.transpose(1, 2)                                 # (B, n_head, T, d_head)
k = k.transpose(1, 2)
v = v.transpose(1, 2)

scale = 1.0 / (d_head ** 0.5)
scores = (q @ k.transpose(-2, -1)) * scale            # (B, n_head, T, T)
out = attn_weights @ v                                  # (B, n_head, T, d_head)
out = out.transpose(1, 2).contiguous().view(B, T, d_model)
```

That `.contiguous()` before the final `.view()` isn't decoration. Transposing produces a tensor that's no longer contiguous in memory, and `.view()` requires contiguity. Skip it and PyTorch throws an error that has nothing obviously to do with the actual problem, which is exactly the kind of thing you only learn by hitting it once.

The feed-forward block in this baseline version is the standard expand-then-contract pattern: project up to `4 * d_model`, apply GELU, project back down. Nothing unusual yet. That comes in Part 3.

The full block wraps attention and the FFN with pre-norm residual connections:

```python
def forward(self, x):
    x = x + self.attn(self.ln1(x))   # pre-norm, then residual
    x = x + self.ffn(self.ln2(x))
    return x
```

Pre-norm, meaning the layer norm runs before each sublayer rather than after, is the choice that matters here. Post-norm, which is what the original transformer paper used, needs much more careful initialization to train reliably at depth. Pre-norm lets the residual stream carry signal through the network without normalization sitting in the way of every skip connection, which is why it's the default in every modern large model.

---

## 3. Training on Real Text

Part 2 puts the architecture to work on real data with byte-level tokenization, meaning the vocabulary is just the 256 possible byte values. No subword merges yet, no vocabulary to build, just the raw bytes.

The data pipeline is a shifted window: input is `tokens[i:i+block_size]`, target is `tokens[i+1:i+1+block_size]`.

```python
x = buf[i : i + block_size]
y = buf[i+1 : i+1 + block_size]
```

That one-position shift is the entire causal language modeling objective. At every position, the model predicts the next token, and cross-entropy is averaged across all positions at once. It's a small piece of code carrying the whole idea.

This is also where the 2-layer, 2-head, 128-dim configuration first appears, trained on a small text file to confirm the loss actually goes down and the model actually produces something coherent before scaling anything up. Getting a tiny model training correctly first, and confirming that by reading its actual generations rather than just watching the loss curve, catches a category of bug that a loss curve alone won't: a model that's confidently learning the wrong thing.

---

## 4. Modernizing the Architecture

Part 3 takes every major piece of the baseline and swaps it for what production models actually use. Each swap had a reason I wanted to understand before writing the code, not after.

**RMSNorm instead of LayerNorm.** LayerNorm centers the activations (subtracts the mean) and scales them (divides by standard deviation), then applies a learned scale and shift. RMSNorm drops the mean subtraction and the shift entirely:

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight
```

Half the learned parameters, no mean computation, a cleaner gradient. In a pre-norm transformer the residual stream already has its mean kept in check by the residual connections themselves, so the centering LayerNorm does turns out to be redundant. I used `eps=1e-8` here, tighter than LayerNorm's usual `1e-5`, since hidden states at 128 dimensions are small enough that the looser value starts to matter.

**RoPE instead of learned positional embeddings.** Rotary embeddings don't add a position vector to the input. Instead, they rotate the query and key vectors inside each attention layer, using precomputed sine and cosine values:

```python
class RoPECache:
    def __init__(self, head_dim, max_pos, base=10000.0):
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2) / head_dim))
        freqs = torch.outer(torch.arange(max_pos), inv_freq)
        self.cos, self.sin = torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    return interleave(xr1, xr2)
```

The property that matters: because both Q and K get rotated, their dot product at any pair of positions depends only on the distance between them, not their absolute positions. A model trained with RoPE on sequences up to 256 tokens handles longer sequences far better than one with learned embeddings, which have no idea what to do past the maximum position they were trained on. The cache grows dynamically if it needs to.

**SwiGLU instead of GELU.** The baseline FFN is one projection, one nonlinearity, one projection back down. SwiGLU adds a gate:

```python
class SwiGLU(nn.Module):
    def __init__(self, dim, mult=4):
        self.w1 = nn.Linear(dim, mult * dim)
        self.w2 = nn.Linear(dim, mult * dim)
        self.w3 = nn.Linear(mult * dim, dim)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        value = self.w2(x)
        return self.w3(gate * value)
```

The gate is input-dependent: for a given token, some intermediate features get amplified and some get suppressed, based on what's actually in that token's representation. GELU applies the same fixed nonlinearity everywhere regardless of content. SwiGLU costs a third weight matrix, and it buys learned routing capacity in exchange.

**KV cache.** Without one, generating token T requires recomputing K and V for every prior token at every step, which makes generation O(T²). A cache stores past K and V and appends only the new token's contribution each step, bringing it down to O(T):

```python
@dataclass
class KVCache:
    k: torch.Tensor   # (B, H, T_so_far, D)
    v: torch.Tensor
```

For longer contexts than a fixed cache can hold cheaply, a rolling variant keeps the first few tokens (an attention sink, since early tokens tend to absorb a disproportionate share of attention weight across many layers) plus a sliding window of recent tokens, capping memory regardless of total sequence length.

**Grouped Query Attention.** Fewer K/V heads than Q heads, with each K/V head shared across a group of queries:

```python
self.wq = nn.Linear(d_model, n_head * d_head)
self.wk = nn.Linear(d_model, n_kv_head * d_head)   # fewer heads here
self.wv = nn.Linear(d_model, n_kv_head * d_head)
```

At inference, K and V get expanded back out via `repeat_interleave` to match the query head count. With `n_head=8` and `n_kv_head=2`, the K/V projections and the KV cache both shrink by 4x, at a small quality cost that's negligible at the scale where Llama, Mistral, and Qwen actually use it.

---

## 5. Training Infrastructure

Part 4 is less conceptually interesting and more load-bearing. This is the difference between a training script that works once and one you can actually run for a long time without babysitting it.

**BPE tokenization** replaces the byte-level tokenizer from Part 2. Instead of 256 raw byte tokens, BPE iteratively merges the most frequent adjacent pairs until it hits a target vocabulary, here 8,000 tokens, trained on the same data the model trains on. Common subwords and whole words become single tokens, which is a lot more for the model to work with per token than a stream of individual bytes.

**Gradient accumulation** simulates a larger batch than fits in memory by accumulating gradients over several forward-backward passes before the optimizer actually steps:

```python
for step, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

The loss gets divided by `accumulation_steps` before `.backward()` so the accumulated gradient ends up the same magnitude it would've been from one large batch, not N times bigger. `set_to_none=True` skips zeroing out memory for parameters with no gradient that step, which is a small but free speedup.

**Mixed precision** runs the forward and backward pass in FP16 while keeping master weights in FP32:

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = model(xb, yb)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
scaler.step(optimizer)
scaler.update()
```

`GradScaler` exists because FP16 gradients can underflow to exactly zero before they ever reach the optimizer. Scaling the loss up before `.backward()` keeps them in a representable range, and `unscale_` before clipping matters because clipping needs to compare against the true gradient magnitude, not the inflated one.

**The learning rate schedule** is warmup followed by cosine decay:

```
warmup:  lr(t) = base_lr * (t / warmup_steps)
decay:   lr(t) = 0.5 * base_lr * (1 + cos(π * progress))
```

Warmup keeps the first few steps from taking huge jumps while the model's weights are still random and the loss surface is steep. Cosine decay brings the rate smoothly toward zero by the end, which in practice lands at a better final loss than a flat or linearly decaying rate.

**Checkpointing** saves model state, optimizer state, the current step, and the best validation loss so far, which matters more than it sounds like it should when your training environment is Colab and sessions end without warning. Losing optimizer state on resume means the momentum and adaptive learning rate terms restart from scratch, and the training trajectory after resuming won't match what it would've been uninterrupted. Saving it means it will.

---

## 6. Mixture of Experts

Part 5 swaps the dense FFN in some layers for a set of expert FFNs with a learned router deciding which tokens go where.

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts, expert_dim, top_k=2):
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, expert_dim), nn.SiLU(),
                          nn.Linear(expert_dim, d_model))
            for _ in range(num_experts)
        ])

    def forward(self, x):
        gate_probs = F.softmax(self.gate(x), dim=-1)
        top_k_probs, top_k_idx = gate_probs.topk(self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (top_k_idx == i).any(dim=-1)
            if mask.any():
                weight = top_k_probs[top_k_idx == i]
                out[mask] += weight.unsqueeze(-1) * expert(x[mask])
        return out
```

Each token routes to its top 2 experts out of the total pool, with the gating weights renormalized after selection so they sum to 1.

Left alone, the router tends to collapse: it finds one or two experts that work reasonably well early on and keeps sending everything there, leaving the rest of the parameter budget unused. The fix is an auxiliary load balancing loss, the same one from the Switch Transformer paper:

```python
def load_balancing_loss(gate_probs, top_k_indices, num_experts):
    tokens_per_expert = F.one_hot(top_k_indices, num_experts).float().sum(-2).mean([0, 1])
    router_prob_per_expert = gate_probs.mean([0, 1])
    return num_experts * (tokens_per_expert * router_prob_per_expert).sum()
```

Perfectly uniform routing gives `tokens_per_expert = router_prob_per_expert = 1/num_experts` for every expert, which makes this loss equal to 1.0. Any deviation pushes it higher. Add this to the main loss with a small coefficient, around 0.01, and the router gets penalized for playing favorites without the penalty swamping the actual training objective.

Not every layer needs to be MoE. Dense layers near the bottom handle general low-level features; MoE layers higher up handle the kind of routing where specialization pays off. That split, dense low, sparse high, is the same pattern Mixtral and DeepSeek use.

---

## 7. What This Hands Off to the Alignment Study

By the end of Part 5, there's a transformer with RMSNorm, RoPE, SwiGLU, GQA, and KV caching, a training loop with gradient accumulation, mixed precision, and proper LR scheduling, a BPE tokenizer trained on the target data, and MoE layers with load-balanced top-k routing, all built without a single pretrained weight.

The model that carries into Parts 6 through 10 is the same 2-layer, 2-head, 128-dim, ~1M parameter config from Part 2. Every architectural upgrade from Part 3 is implemented and available, but the alignment study runs on the small config specifically so that four different post-training methods could each be trained to completion, evaluated, retuned, and retrained, on free compute, in a timeframe where I could actually run enough experiments to see something real. That's what produced the result in the alignment post: DPO starting behind SFT and finishing first, PPO starting first and finishing last, GRPO recovering from actual group collapse once the group size and generation temperature were fixed. None of that comparison happens on a model too large to run five times.

The full writeup of what happened there, including exactly what group collapse looks like in the group standard deviation logs and why DPO's reward margin exploded to 599 before it was fixed, is in the [alignment study](https://brayanbrayan.github.io/2026/04/02/rlhf-post-blog.html).

---

## 8. What I'd Do Differently

Single-GPU training only, no data or pipeline parallelism, which caps this approach well below the size of anything resembling a modern LLM. That's a known and accepted limit of the project, not something to fix within it.

No gradient checkpointing. At 128 dimensions and 2 layers, memory was never the bottleneck, so the added complexity wasn't worth it here, but it's the first thing I'd add if I scaled the model up meaningfully.

The MoE load balancing coefficient (0.01) was set once and not swept. I'd want to check whether a different value changes the collapse behavior before trusting that number as a default rather than a guess that happened to work.

No distributed training, no quantization, no speculative decoding. All reasonable to skip for a from-scratch learning project, all the obvious next additions if this stopped being that and started being something meant to run at real scale.

---

*Built entirely from scratch in PyTorch · No pretrained weights · No trainer APIs · Single Colab GPU*
