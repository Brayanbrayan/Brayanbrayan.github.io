# Where Do You Put the Critic? Value Network Architecture for LLM Reinforcement Learning

**Brayan** · May 2026 · *ML Research Blog*

---

> *Note: VAML is an open-source research project by [gabe00122](https://github.com/gabe00122). The core ideas, architecture, and research hypothesis are entirely his. I came across this project on Discored EleutherAi Communoty and was immediately fascinated by the discussion. I contributed a GRPO baseline implementation, which forms the latter part of this writeup. What follows is my attempt to document and explain what is going on in this codebase in a way that is understanble and introduces the research idea.*

---

## Abstract

Standard Reinforcement Learning from Human Feedback (RLHF) practice places value heads on the final transformer layer of language models. VAML (Value Approximation for LLMs) challenges this convention, hypothesizing that final-layer representations are contaminated by the next-token prediction objective and therefore poorly suited for long-term value estimation. Instead, VAML taps intermediate "latent" representations from within the base model's transformer stack, feeding them into a separate 12-layer value network. The project uses Wordle as a structured test environment chosen for its clear multi-step credit assignment and unambiguous feedback signals and trains using PPO with Generalized Advantage Estimation (GAE). A curious empirical finding has emerged: the value network's predictions form a sharp step function, remaining flat during guessing and jumping discontinuously the moment environment feedback tokens appear in context. This blog documents the architecture, the mathematics, and my contribution of a GRPO baseline, exploring what this project reveals about how we should think about value estimation in LLM training.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background: Key Terms](#2-background)
3. [The Problem with Standard RLHF Architectures](#3-the-problem)
4. [Architecture: Tapping the Latent Space](#4-vaml-architecture)
5. [The Wordle Environment](#5-wordle-environment)
6. [Training Dynamics: PPO with GAE](#6-training-dynamics)
7. [The Step Function Phenomenon](#7-step-function)
8. [Introducing the GRPO Baseline](#8-grpo-baseline)
9. [Current Status and Future Directions](#9-future)
10. [References](#10-references)

---

## 1. Introduction

When people talk about RLHF the technique behind ChatGPT, Claude, and most modern aligned language models they tend to focus on the reward model or the policy. The value network is treated almost as scaffolding necessary for PPO to work, but not particularly interesting in itself.

VAML poses an almost overly straightforward question: *Does it matter where you attach the value head inside the model?*

The conventional answer is no. You take the LLM, bolt a small linear projection onto the final hidden state, and call it a critic. This is what OpenAI's InstructGPT does [1], what most RLHF implementations do, and what almost every open-source RL-for-LLMs library assumes.

VAML's answer is: *actually, yes, it matters quite a lot. And we have been doing it wrong.*

The intuition is this. A transformer's final layer is not a neutral summary of what the model knows. It is a representation that has been shaped, over billions of training steps, to answer one specific question: *what token comes next?* When you bolt a value head onto that layer, you are not reading the model's understanding of the world. You are reading a compressed, task-biased representation that may have discarded exactly the strategic information you need.

VAML proposes an alternative: reach *inside* the model, extract representations from the middle layers where strategic reasoning is believed to live, and route them into a separate, purpose-built value network. This document traces that idea from concept to implementation, and describes my contribution of a GRPO baseline to allow controlled comparison.

---

## 2. Background: Pinning Down the Terminology

Before engaging with the architecture, precise definitions matter here, because three terms (Value Function, Value Network, and Value Head) are frequently used interchangeably in practice but refer to meaningfully different things.

### 2.1 The Value Function (Mathematical Concept)

The value function is a theoretical object. It is the expected discounted return from state $s$ under policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k r_{t+k+1} \;\middle|\; S_t = s \right]$$

This is the quantity we want to approximate. It is not a neural network; it is the ground truth we are trying to learn. $\gamma \in [0, 1]$ is the discount factor controlling how much future rewards are weighted relative to immediate ones.

### 2.2 The Value Network vs. The Value Head

A **Value Network** is a dedicated neural network whose entire purpose is to approximate $V^\pi(s)$. In VAML, this is a 12-layer transformer, a substantial standalone model with its own parameters.

A **Value Head** is a small linear projection bolted onto an existing model's final hidden state. It is cheap, ubiquitous, and the conventional implementation in RLHF:

```python
# Standard implementation — the value head pattern
hidden_states = transformer(input_ids)      # shape: [batch, seq_len, hidden_dim]
final_layer   = hidden_states[:, -1, :]    # take last layer output only
value         = nn.Dense(1)(final_layer)   # project to single scalar
```

The distinction matters: a value head *borrows* the LLM's representations; a value network *builds its own*, using the LLM's internals as input rather than its output. In modern LLM RL, value heads are almost universally preferred because full value networks are expensive to train. VAML argues this is a false economy.

### 2.3 The Three Models in RLHF

A properly specified RLHF setup contains three distinct networks, not two:

| Model | When Trained | Purpose | Signature |
|---|---|---|---|
| **Reward Model** | Before RL (then frozen) | Score complete responses | (prompt, response) → scalar |
| **Policy Network** | During RL | Generate tokens | state → token distribution |
| **Value Network / Head** | During RL | Estimate future reward | state → expected return |

The reward model and the value critic are *not* the same thing. The reward model scores finished sequences; the critic scores partial sequences at every token step. This distinction is often glossed over, but it is fundamental to understanding what VAML is modifying.

### 2.4 Why Do We Need a Critic at All?

The critic exists to solve the **credit assignment problem**. Consider a Wordle game:

```
Turn 1: Guess "CRANE" → [⬜][🟨][⬜][⬜][🟨]
Turn 2: Guess "SHOUT" → [⬜][🟩][🟩][🟩][🟩]
Turn 3: Guess "SHORT" → [🟩][🟩][🟩][🟩][🟩]  ✓ WIN
```

If the final reward is +10, which of the three guesses was responsible? CRANE eliminated several letters. SHOUT locked four positions. SHORT just finished the job. Without a critic, all three receive identical credit signals.

With a critic that estimates $V(s)$ at each turn, we can compute **advantages**: how much better each action was than expected:

$$A(\text{CRANE}) = V(s_1) - V(s_0) = 7.0 - 5.0 = +2.0$$
$$A(\text{SHOUT}) = V(s_2) - V(s_1) = 9.5 - 7.0 = +2.5$$
$$A(\text{SHORT}) = V(s_3) - V(s_2) = 10.0 - 9.5 = +0.5$$

These advantages tell the policy optimizer exactly how much to reinforce each action, and in which direction.

---

## 3. The Problem with Standard RLHF Architectures

### 3.1 What the Final Layer Actually Encodes

A transformer pretrained on next-token prediction develops a specific internal structure through gradient descent. Empirical analyses of transformer representations [2, 3] consistently find a rough functional hierarchy:

- **Early layers (1–10):** Syntax, entity recognition, basic factual associations
- **Middle layers (11–20):** Semantic understanding, relational reasoning, contextual inference
- **Final layers (21–36):** Task-specific formatting, output distribution shaping for next-token prediction

The final layers are under sustained optimization pressure to encode *what token comes next*, not *what is the long-term value of this state*. These are different representational goals, and gradient descent does not automatically produce representations that serve both.

### 3.2 The Contamination Hypothesis

VAML's author frames this as a contamination problem. When you attach a value head to the final layer:

```
Pretraining objective: minimize CrossEntropy(predicted_token, actual_token)
                                  ↓  (over hundreds of billions of tokens)
Final layer representation: compressed, task-biased toward token selection
                                  ↓
Value head: "What is V(s)?"
```

The value head is trying to extract long-range strategic information from a representation that has been optimized to suppress everything except what is needed for token selection. For Wordle specifically: if the model is mid-way through typing "CRAN" and about to type "E", the final layer's representation is dominated by the question *"should the next token be E, N, I, or something else?"*

What the value critic needs to know is something else entirely:

- How many candidate words remain in the solution space?
- Have we covered enough vowels and common consonants?
- Is this guess maximally informative given the current constraint set?

The final layer may not preserve these quantities; it has been trained to discard them in favor of token probability distributions.

### 3.3 Why Standard Solutions Are Insufficient

The standard response is that joint training of policy and value on the LLM backbone will gradually reshape representations to serve both objectives. There are three reasons this argument fails in practice:

**Pretraining dominance.** The base model has seen hundreds of billions of tokens optimizing for next-token prediction. A few thousand RL update steps cannot fundamentally reshape what the final layer encodes. The pretraining gradient has a massive head start.

**Gradient conflict.** If you allow value loss to backpropagate into the base model, you create a tug-of-war between the LM objective and the value objective. Most implementations freeze the base model or use LoRA precisely to avoid this, which means the final layer's representations are fixed to serve next-token prediction and nothing else.

**Representational specificity.** Even when not frozen, there is no guarantee gradient descent will find a representation that simultaneously serves next-token prediction and value estimation well. The two objectives may require fundamentally different geometric structures in representation space.

---

## 4. VAML's Architecture: Tapping the Latent Space

### 4.1 The Core Innovation

Rather than reading the final layer, VAML collects the hidden state from *every* transformer layer during the forward pass and routes all of them into a dedicated value network that processes them in parallel with the policy.

The base model is **Qwen3-4B** [4], with the following key dimensions:

| Parameter | Value |
|---|---|
| Hidden size | 2,560 |
| Number of layers | 36 |
| Attention heads | 32 (GQA with 8 KV heads) |
| Head dimension | 128 |
| FFN intermediate size | 9,728 |
| Vocabulary size | 151,936 |
| Precision | bfloat16 |

The modified forward pass collects all 37 intermediate representations (embedding + 36 layer outputs):

```python
# model/qwen3.py — the modified forward pass
def __call__(self, tokens, positions, carry=None, *, rng_key):
    x = self.embeddings(tokens)

    latents = [x]                                    # Start: embedding output
    for layer in self.layers:
        x, _ = jax.checkpoint(layer)(x, positions)
        latents.append(x)                            # Collect EVERY layer output

    # Route all 37 latents to value network (runs in parallel with policy head)
    if self.value_net is not None:
        value_repr, _, rng_key = self.value_net(latents, positions, rng_key=rng_key)

    # Standard policy head (unchanged)
    x = self.final_norm(x)
    logits = x @ self.embeddings.embedding.T
    return logits, value_repr, carry, rng_key
```

The system architecture looks like this:

```
Input Tokens
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│  QWEN3 BASE MODEL  (36 layers)                             │
│                                                            │
│  Embed → L1 → L2 → L3 → ... → L15 → ... → L36            │
│    │      │    │    │           │           │              │
│   [L0]  [L1] [L2] [L3]  ... [L15]  ... [L36]             │
│    └──────┴────┴────┴──────────┴───────────┘              │
│                  Latent Collection (37 tensors)            │
│                           │                               │
│                           ├──────────────────────────┐    │
│                           │                          │    │
│  → Final Norm → LM Head → Token Logits (Policy)     │    │
└────────────────────────────────────────────────────────────┘
                                                      │
┌─────────────────────────────────────────────────────┘
│  VALUE NETWORK  (12 transformer layers)
│
│  Sample every 3rd latent:
│  [L0, L3, L6, L9, L12, L15, L18, L21, L24, L27, L30, L33, L36]
│                                 ↓
│    [VL1]─[VL2]─[VL3]─...─[VL12]
│      ↑     ↑     ↑           ↑
│     [L0]  [L3]  [L6]      [L33]   ← latents injected via cross-attention
│                                 ↓
│              HL-Gauss Head → distribution over 51 reward bins
└──────────────────────────────────────────────────────────────
```

### 4.2 The ValueBackbone

The 12-layer value network downsamples the 36 collected latents evenly, sampling every third layer:

```python
# model/value_network.py — ValueBackbone forward pass (simplified)
def __call__(self, latents, positions, *, rng_key):
    x, *layer_latents = latents                          # Unpack embedding + 36 outputs

    take_every = len(layer_latents) // len(self.layers)  # 36 // 12 = 3
    layer_latents = layer_latents[::take_every][:len(self.layers)]
    # Sampled: [L3, L6, L9, L12, L15, L18, L21, L24, L27, L30, L33, L36]

    for layer, latent in zip(self.layers, layer_latents):
        x, _, rng_key = layer(x, latent, positions, rng_key=rng_key)
        # Each value layer: x = TransformerBlock(x + ValueNetEncode(latent))

    x = self.final_norm(x)
    value_repr = self._head(x)    # HL-Gauss distribution
    return value_repr, None, rng_key
```

Each value layer receives the corresponding Qwen3 latent via a gated injection:

$$\mathbf{x}_{t+1} = \text{TransformerBlock}\!\left(\mathbf{x}_t + \text{ValueNetEncode}(\mathbf{h}_{\text{base}})\right)$$

The `ValueNetEncode` module uses a gated activation (SiLU) to selectively pass features from the Qwen3 representation into the value network's processing stream.

### 4.3 The Stop-Gradient Barrier

This is the most important architectural detail. The value network must not backpropagate gradients into the base model through the latent connection. If it did, the value loss would reshape Qwen3's representations, potentially degrading its language modeling ability and creating an uncontrolled feedback loop.

```python
# model/value_network.py — ValueNetEncode
class ValueNetEncode(nnx.Module):
    def __call__(self, x, *, rng_key):
        x = jax.lax.stop_gradient(x)   # Hard barrier: zero gradients into Qwen3
        x = self._normalize(x)
        gate = self._up_gate(x)
        x = self._encode_up(x)
        x = jax.nn.silu(x) * gate      # Gated activation for selective feature passing
        x = self._encode_down(x)
        return x, rng_key
```

The gradient flow is therefore:

```
Value loss → Value Network (12 layers) ← STOP_GRADIENT ─ Qwen3 latents
Policy loss → LoRA adapters in Qwen3                    ← Qwen3 latents
```

The base model's weights are updated only by the policy gradient (through LoRA); the value network updates only its own 12 layers.

### 4.4 Memory Management: Gradient Checkpointing

Storing 37 intermediate tensors per forward pass is expensive. For a batch of 8 sequences with sequence length 256 and hidden dimension 2560:

$$\text{Memory} = 37 \times 8 \times 256 \times 2560 \times 2\ \text{bytes (bfloat16)} \approx 375\ \text{MB}$$

This is just for the latent collection, before optimizer state. `jax.checkpoint()` mitigates this with a memory-compute tradeoff: during the forward pass, only layer *outputs* are stored; internal activations (QKV projections, attention scores, FFN intermediates) are discarded and recomputed on-demand during backpropagation.

Result: approximately 50% memory reduction at the cost of approximately 33% slower backward pass. For a memory-constrained research setup, this is the right tradeoff.

### 4.5 HL-Gauss: Distributional Value Estimation

Rather than predicting a scalar $V(s) \in \mathbb{R}$, the final value head outputs a probability distribution over 51 evenly-spaced bins spanning $[-0.1, 1.1]$:

```json
"head": {
  "type": "hl_gauss",
  "min": -0.1,
  "max": 1.1,
  "n_logits": 51,
  "sigma": 0.05
}
```

The expected value is the weighted sum over bins:

$$\hat{V}(s) = \sum_{i=1}^{51} p_i \cdot b_i, \quad b_i = -0.1 + \frac{1.2(i-1)}{50}$$

The training loss is KL divergence between the predicted distribution and a Gaussian centered on the GAE return target:

$$\mathcal{L}_{\text{value}} = D_{\text{KL}}\!\left(\mathcal{N}\!\left(\hat{G}_t,\, \sigma^2\right) \;\Big\|\; \hat{p}_\phi(b)\right)$$

The distributional output is motivated by the step-function finding described later. Before receiving feedback, there is genuine uncertainty about future reward. A scalar value collapses this uncertainty to a point estimate; a distribution preserves it. The width of the predicted distribution encodes the value network's confidence: narrow when the situation is clear, wide when it is not.

### 4.6 Dual Optimizers

Policy and value are trained by separate optimizers at deliberately different rates:

```json
"policy_optimizer": {
  "lr": 1e-5,
  "schedule": "warmup_cosine"
},
"value_optimizer": {
  "lr": 1e-4,
  "multi_step": 4,
  "schedule": "warmup_cosine"
}
```

The value network trains at $10\times$ the policy learning rate and takes 4 gradient steps per policy step. The rationale: the value network starts from random initialization and must rapidly bootstrap accurate return estimates; the policy is already a competent language model and needs only conservative adjustments to adapt to Wordle.

### 4.7 LoRA for Parameter-Efficient Policy Updates

The 3.7B-parameter Qwen3 base is not directly fine-tuned. Instead, Low-Rank Adaptation [5] inserts small trainable matrices into the attention and MLP layers:

$$W_{\text{eff}} = W_{\text{frozen}} + \underbrace{B \cdot A}_{\text{rank-}r\text{ update}}, \quad A \in \mathbb{R}^{d \times r},\; B \in \mathbb{R}^{r \times d}$$

With rank $r = 32$ and hidden dimension $d = 2560$:

$$\text{Parameters per layer} = 2 \times (d \times r) = 2 \times 2560 \times 32 = 163{,}840$$
$$\text{Total LoRA parameters} = 36\ \text{layers} \times 163{,}840 \approx 23\ \text{M}$$

This is approximately 0.6% of the full model, a 160× reduction in trainable parameters, while keeping optimizer state requirements at around 200 MB instead of 30 GB.

---

## 5. The Wordle Environment

### 5.1 Why Wordle?

Wordle is not an arbitrary choice of environment. It satisfies a specific set of properties that make credit assignment difficult and measurable:

**Sparse, structured feedback.** You learn nothing until you submit a complete five-letter guess. Every token within a guess carries zero reward. This stresses the critic's ability to assign value across a long sequence of unrewarded steps.

**Multi-turn horizon.** The game takes up to six guesses. A good first guess (high information coverage) may not pay off until turns two or three. The value network must bridge this temporal gap.

**Unambiguous feedback.** Green, yellow, and gray tiles give exact logical information about the secret word. There is no noise in the feedback signal.

**Bounded episodes.** Games end in at most six guesses, making training tractable without truncation.

### 5.2 The Rust Environment

The environment is implemented in Rust and compiled to Python via PyO3. This reflects a deliberate engineering tradeoff: environment simulation requires no gradients, and Rust is orders of magnitude faster than Python for tight simulation loops.

```rust
// src/wordle.rs — Two-tier architecture for efficient parallel execution
pub struct WordleShared {
    settings: WordleSettings,
    words: Vec<String>,     // Shared immutably across all instances
    guess_re: Regex,
}

struct WordleInstance {
    shared: Arc<WordleShared>,   // Arc = atomic reference count, zero-copy sharing
    rng: SmallRng,
    secret_word_index: usize,
    got_yellow: [bool; 5],       // Track per-position discoveries
    got_green: [bool; 5],
    guesses: usize,
}
```

The `Arc<WordleShared>` pattern allows 32 parallel environment instances to share the word list and regex without duplication.

### 5.3 The Reward Signal

The reward function is designed to be sparse, incremental, and bounded:

```rust
// src/wordle.rs — Reward calculation after each guess
let mut reward = 0.0;

for (i, f) in feedback.iter().enumerate() {
    // Reward for first discovery of a letter in any position
    if (f == &'Y' || f == &'G') && !self.got_yellow[i] {
        self.got_yellow[i] = true;
        reward += 0.05;
    }
    // Additional reward for first positional lock (green)
    if f == &'G' && !self.got_green[i] {
        self.got_green[i] = true;
        reward += 0.05;
    }
}

// Win bonus
if guess == secret_word {
    reward += 0.5;
}
```

Maximum episode reward: $5 \times 0.10 + 0.50 = 1.0$. Rewards are assigned *after* feedback tokens appear, not during token generation. This timing is the direct cause of the step-function phenomenon discussed below.

### 5.4 The Token-Level View

From the model's perspective, a single Wordle turn produces a sequence like:

```
Tokens:       [C]  [R]  [A]  [N]  [E]  [\n]  [⬜] [🟨] [⬜] [⬜] [🟨]
                                              ↑ Environment feedback begins
Rewards:      0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.05 0.0  0.0  0.05
Policy mask:  T    T    T    T    T    T    F    F    F    F    F
                                              ↑ Model stops generating here
```

The model generates the guess tokens (policy mask = True). The environment generates the feedback tokens (policy mask = False). Advantages are computed only over policy-generated tokens, but the value network estimates $V(s_t)$ everywhere, including during the guess, where all rewards are zero.

---

## 6. Training Dynamics: PPO with GAE

### 6.1 Generalized Advantage Estimation

VAML uses Generalized Advantage Estimation [6] to compute advantages from the value network's predictions. GAE introduces a bias-variance tradeoff parameter $\lambda$:

$$A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

With $\lambda = 0$: pure TD(0), high bias, low variance. With $\lambda = 1$: Monte Carlo returns, low bias, high variance. The config uses `gae_lambda = 1.0`, effectively Monte Carlo returns, which makes sense given the sparse reward structure (few intermediate signals to bootstrap from).

The JAX implementation processes the trajectory in reverse using `jax.lax.scan`, avoiding Python-level loops:

```python
# update_step.py — GAE via reverse scan
def calculate_advantages(rewards, values, discount, td_lambda):
    def _body(acc, xs):
        reward_t, value_tp1, discount_t, lambda_t = xs
        acc = reward_t + discount_t * ((1 - lambda_t) * value_tp1 + lambda_t * acc)
        return acc, acc

    _, targets = jax.lax.scan(
        _body,
        init=jnp.zeros_like(values[:, 0]),
        xs=(rewards, values[:, 1:], discount, td_lambda),
        reverse=True,                    # Process from end to start
    )
    advantages = targets - values[:, :-1]
    return advantages, targets
```

### 6.2 Turn-Aware Discounting

At the boundary between one Wordle turn and the next, where environment feedback ends and the model begins generating a new guess, the discount factor switches:

```python
# Detect transitions from environment token to policy token
turn_boundaries = ~policy_mask[:, :-1] & policy_mask[:, 1:]

# Apply reduced discount at turn boundaries
td_discount = jnp.where(turn_boundaries, turn_discount=0.9, gae_discount=1.0)
```

This prevents the GAE estimator from "looking backward" across turns. Without it, the value estimate for the second turn's opening token would be influenced by the reward structure of the first turn, a form of reward contamination across turn boundaries.

### 6.3 The PPO Objective

The complete PPO loss combines three terms:

$$\mathcal{L} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}} - \beta_{\text{ent}} \cdot H(\pi)$$

**Clipped policy loss** with asymmetric bounds ($\varepsilon_{\text{low}} = 0.2$, $\varepsilon_{\text{high}} = 0.28$):

$$\mathcal{L}_{\text{policy}} = -\mathbb{E}_t \left[ \min\!\left( \rho_t A_t,\; \text{clip}(\rho_t, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}) \cdot A_t \right) \right]$$

where $\rho_t = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$ is the probability ratio.

**Distributional value loss:**

$$\mathcal{L}_{\text{value}} = \mathbb{E}_t\!\left[ D_{\text{KL}}\!\left(\mathcal{N}(G_t^{\text{GAE}}, \sigma^2) \;\Big\|\; \hat{p}_\phi(b)\right) \right]$$

**Entropy regularization** to maintain exploration:

$$H(\pi) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$$

---

## 7. The Step Function Phenomenon

### 7.1 What Was Observed

The key empirical finding so far is the temporal shape of the value predictions. Across training, the value estimates exhibit a sharp discontinuity rather than a smooth evolution:

```
Token:  [C]   [R]   [A]   [N]   [E]   [⬜]  [🟨]  [⬜]  [⬜]  [🟨]
        ←── Policy generates guess ──→  ←── Environment feedback ──→

Value:  5.02  5.01  5.03  5.01  5.04  5.10  7.30  7.45  7.48  7.50
                                              ↑
                                 Sharp jump at feedback
```

During the guess phase (C, R, A, N, E), the value estimate is nearly flat. At the first feedback token, it jumps, and continues adjusting as each tile is revealed. This is the step function.

### 7.2 Two Interpretations

**Interpretation A: The model is reasoning (the good outcome):**

The value network has learned a world model of Wordle encoded in the middle layers. Before feedback, there is *genuine* epistemic uncertainty about what the environment will reveal. A correct Bayesian agent would maintain high-entropy value estimates until information arrives, then update sharply and accurately.

Under this interpretation, the step function is correct behavior. The flat region reflects genuine uncertainty; the jump reflects accurate Bayesian updating on new information. The value network is modeling information gain, not failing to anticipate reward.

**Interpretation B: The model is pattern-matching (the bad outcome):**

The value network learned a shallow heuristic: when yellow or green tiles appear in the context, increase the value estimate. It has learned to react to the *literal tokens* representing feedback, not to infer anything about the underlying game state during the guess phase.

Under this interpretation, the middle layers do not encode Wordle reasoning. The step function is a symptom of failed credit assignment; the model has learned to condition on reward signals rather than anticipate them.

### 7.3 How to Distinguish the Two

**Probing classifiers** are the cleanest diagnostic. At each token position during the guess phase (before any feedback), extract the hidden state from layer 15 and train a linear probe to predict:

- The number of remaining candidate words (a proxy for information state)
- The set of ruled-out letters
- The information entropy of the remaining word distribution

If probes achieve significantly above-chance accuracy, the middle layers carry structured game-state representations, supporting Interpretation A. If accuracy is near chance, the representations do not encode Wordle reasoning.

**Discount factor experiments** provide a complementary test. With $\gamma = 0.99$, the model should plan multiple turns ahead. With $\gamma = 0.5$, greedy single-turn optimization suffices. If changing $\gamma$ produces no behavioral difference, the model is not performing multi-step planning regardless of what the latent representations contain.

Both experiments were planned but have not yet been executed. The project is currently paused.

---

## 8. Introducing the GRPO Baseline

*This section describes my contribution to the VAML project.*

### 8.1 Why a Baseline?

An experimental hypothesis without a comparison point is not a falsifiable claim. To evaluate whether VAML's deep-latent value architecture provides genuine improvement over simpler alternatives, we need a controlled baseline that uses the same policy (Qwen3 with LoRA) and the same environment (Wordle) but eliminates the value network entirely.

GRPO (Group Relative Policy Optimization) [7], introduced by the DeepSeek team, provides exactly this. It removes the critic and replaces it with group-relative reward normalization, an approach that has demonstrated strong performance on mathematical reasoning tasks while being significantly cheaper to run.

The experimental question is direct: does VAML's 12-layer critic provide benefit over a zero-parameter baseline estimator?

### 8.2 GRPO: The Algorithm

GRPO's insight is that you can estimate advantages without a critic if you generate multiple rollouts for the same prompt and normalize returns within each group:

$$A_i^{(g)} = \frac{R_i - \mu_g}{\sigma_g + \epsilon}, \quad \mu_g = \frac{1}{k}\sum_{i=1}^k R_i^{(g)}, \quad \sigma_g = \sqrt{\frac{1}{k}\sum_{i=1}^k (R_i^{(g)} - \mu_g)^2}$$

For this to produce meaningful comparisons in Wordle, all rollouts within group $g$ must play the *same* secret word. Otherwise, the variance in returns reflects word difficulty rather than policy quality, and the normalization becomes meaningless.

### 8.3 Seed-Based Environment Grouping

The core implementation challenge is ensuring that $k$ environments within a group receive the same Wordle word deterministically. The Rust environment was extended with a grouped factory:

```rust
// src/env.rs — Grouped environment initialization
pub fn new_grouped(
    num_per_group: usize,
    group_seeds: Vec<u64>,
    settings: S::Settings,
) -> Self {
    // For each seed, create num_per_group environments
    // All environments in group_i share seed_i → same word selection
    let mut instances = Vec::new();
    for seed in &group_seeds {
        let mut rng = StdRng::seed_from_u64(*seed);
        for _ in 0..num_per_group {
            instances.push(E::new_with_rng(&mut rng, settings.clone()));
        }
    }
    Self { instances }
}
```

A new `WordleEnvGrouped` PyO3 class exposes this to Python. The original `WordleEnv` is unchanged; backward compatibility is preserved.

### 8.4 The Grouped Buffer

JAX/XLA requires statically known shapes for JIT compilation. A buffer that groups rollouts dynamically at runtime would trigger recompilation on every batch. The solution is pre-allocation with a fixed three-dimensional shape:

```python
# python/vaml/agent/grpo_buffer.py
class GRPOBuffer:
    """
    Fixed-shape buffer for GRPO grouped rollouts.
    Shape: [num_groups, rollouts_per_group, max_seq_len]
    Static shape is required for XLA JIT compilation.
    """
    def __init__(self, num_groups=4, rollouts_per_group=8, max_seq_len=4096):
        shape = (num_groups, rollouts_per_group, max_seq_len)
        self.context   = np.zeros(shape, dtype=np.int32)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.rewards   = np.zeros(shape, dtype=np.float32)
        self.masks     = np.zeros(shape, dtype=np.bool_)

        # Metadata (not passed to JAX)
        self.group_seeds  = np.zeros(num_groups, dtype=np.uint32)
        self.current_group = 0
        self.current_slot  = 0
```

With `num_groups = 4` and `rollouts_per_group = 8`, the total collection size is 32 parallel environments per update step, comparable to the PPO setup.

### 8.5 JAX-Compatible Advantage Computation

The advantage computation is designed to be JIT-compilable with no dynamic shapes and no data-dependent control flow:

```python
# python/vaml/grpo_loss.py
@jax.jit
def compute_grpo_advantages(rewards, masks):
    """
    rewards: [num_groups, rollouts_per_group, seq_len]
    masks:   [num_groups, rollouts_per_group, seq_len]
    Returns: advantages with same shape
    """
    # Step 1: Episode returns — sum rewards over each rollout's sequence
    episode_returns = jnp.sum(rewards * masks, axis=-1)
    # shape: [num_groups, rollouts_per_group]

    # Step 2: Per-group normalization — static axis, XLA-friendly
    group_mean = jnp.mean(episode_returns, axis=1, keepdims=True)  # [groups, 1]
    group_std  = jnp.std(episode_returns,  axis=1, keepdims=True) + 1e-8
    normalized = (episode_returns - group_mean) / group_std
    # shape: [num_groups, rollouts_per_group]

    # Step 3: Broadcast to token level
    advantages = normalized[:, :, None] * masks
    # shape: [num_groups, rollouts_per_group, seq_len]

    return advantages
```

No dynamic indexing. No data-dependent branches. XLA compiles this once and reuses the kernel across all batches.

### 8.6 The GRPO Loss Function

The full objective combines the PPO clipped surrogate with a stronger KL penalty and entropy bonus:

$$\mathcal{L}_{\text{GRPO}} = \underbrace{-\min\!\left(\rho A,\; \text{clip}(\rho,\, 1-\varepsilon,\, 1+\varepsilon) \cdot A\right)}_{\text{clipped surrogate}} + \underbrace{\beta_{\text{KL}} \cdot D_{\text{KL}}(\pi_{\text{old}} \| \pi_\theta)}_{\text{stability penalty}} - \underbrace{\beta_{\text{ent}} \cdot H(\pi_\theta)}_{\text{exploration bonus}}$$

with $\beta_{\text{KL}} = 0.05$ (versus PPO's typical 0.01) and $\beta_{\text{ent}} = 0.01$. The stronger KL penalty compensates for the absent value baseline: without a critic to reduce advantage variance, the policy is more prone to making large, potentially destabilizing updates.

### 8.7 Architecture Comparison

| Component | PPO with VAML | GRPO Baseline |
|---|---|---|
| Value network | 12-layer transformer | None |
| Advantage estimator | GAE with HL-Gauss critic | Group mean/std normalization |
| Memory footprint | ~400 MB | ~200 MB |
| Forward pass cost | Qwen3 + ValueNet | Qwen3 only |
| Advantage computation | $O(T)$ reverse scan | $O(k)$ mean/std |
| KL coefficient | 0.01 | 0.05 |
| Estimated time per update | ~3.2 s | ~1.8 s |

### 8.8 Files Delivered

The implementation adds six new artifacts and modifies three existing ones:

| File | Type | Purpose |
|---|---|---|
| `src/env.rs` | Modified | Grouped environment factory |
| `src/wordle.rs` | Modified | `WordleEnvGrouped` PyO3 class |
| `src/lib.rs` | Modified | Export `WordleEnvGrouped` |
| `python/vaml/agent/grpo_buffer.py` | New | Fixed-shape grouped buffer |
| `python/vaml/grpo_loss.py` | New | JIT-compiled loss and advantages |
| `python/vaml/agent/grpo_agent.py` | New | Grouped rollout collection |
| `python/vaml/train_grpo.py` | New | Training entry point |
| `configs/grpo.json` | New | Hyperparameter configuration |

Both training pipelines are now independently runnable:

```bash
# PPO with deep-latent value network (original)
python python/vaml/train_rl.py configs/value-net.json

# GRPO with group-relative advantages (baseline)
python python/vaml/train_grpo.py configs/grpo.json
```

---

## 9. Current Status and Future Directions

### 9.1 Where the Project Stands

The project is currently **paused** due to compute constraints. The implementation is complete and verified; large-scale experiments have not yet been run. What exists:

- A fully functional PPO pipeline with VAML's deep-latent value architecture
- A GRPO baseline with correct seed-based environment grouping and JIT-compiled advantage computation
- Validated code that compiles, runs, and produces outputs consistent with expectations on small test runs

What does not yet exist: the controlled experimental results that would confirm or refute the core hypothesis.

### 9.2 Open Questions

**The primary question** (does deep-latent value estimation outperform final-layer alternatives in sample efficiency and final win rate?) remains unanswered. The comparison requires running all three configurations (VAML-PPO, standard PPO, GRPO) for identical episode counts under identical compute budgets.

**The step-function question** is arguably the more scientifically interesting of the two. If probing classifiers reveal that layer 15 activations encode structured Wordle game-state information (remaining candidates, ruled-out letters, entropy), that is a mechanistic interpretability finding independent of training performance. It would suggest that mid-layer representations are genuinely richer for reasoning tasks, a result that generalizes beyond this project.

**The scalability question** is entirely untested. All architecture decisions are tuned for Qwen3-4B. Whether the latent-tapping hypothesis holds for 7B or 14B models, where the pretraining-RL objective gap may be different, is unknown.

### 9.3 What Would Constitute Success

For VAML's hypothesis to be confirmed, three things would need to hold simultaneously:

1. Probing classifiers decode game-state information from middle-layer activations at above-chance accuracy during the guess phase (before feedback).
2. Value estimates anticipate feedback rather than purely reacting to it; the step function softens over training.
3. VAML-PPO reaches target win rate in fewer environment interactions than both standard PPO (final-layer value head) and GRPO.

If any of these fails, the hypothesis is partially or fully refuted. That is also a meaningful outcome.

### 9.4 Directions Worth Pursuing

**Short term:**
- Probing classifier experiments on layer 15 vs. layer 36 activations
- Controlled benchmark: VAML-PPO vs. standard PPO vs. GRPO on win rate and sample efficiency
- Hyperparameter sweep on GRPO group size (2, 4, 8) and KL coefficient

**Medium term:**
- Adaptive KL penalty (auto-adjust based on observed divergence from target)
- Extension to the arithmetic environment already implemented in Rust
- Test on a language task rather than a game to assess domain generality of the latent hypothesis

**Long term:**
- If probes succeed: write up the mechanistic interpretability finding as a standalone contribution
- If VAML outperforms GRPO significantly: architectural comparison paper
- Hybrid approaches: using group structure to generate a distributional baseline, then refining with a lightweight critic

### 9.5 A Reflection

What drew me to this project was how cleanly it posed a question that most RLHF practice treats as settled. The placement of the value head is not a hyperparameter people typically sweep. It is a design decision frozen in place since InstructGPT, inherited by nearly every open-source RLHF implementation without question.

VAML asks whether that decision is actually well-motivated. The step-function finding suggests that something interesting is happening in the gap between the model's internal reasoning and its observable behavior, and that we do not yet fully understand what the value network is actually learning.

Whether the hypothesis is confirmed or refuted, the question deserves an answer.

---

## 10. References

[1] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35, 27730–27744.

[2] Tenney, I., Das, D., & Pavlick, E. (2019). BERT rediscovers the classical NLP pipeline. *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, 4593–4601.

[3] Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT. *Advances in Neural Information Processing Systems*, 35, 17359–17372.

[4] Qwen Team. (2025). Qwen3 Technical Report. *arXiv preprint arXiv:2505.09388*.

[5] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*.

[6] Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. *arXiv preprint arXiv:1506.02438*.

[7] Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., ... & Guo, D. (2024). DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*.

---

*VAML is an open-source research project by gabe00122. The core architecture, research hypothesis, and experimental design are entirely his work. This blog reflects my attempt to document and understand the project, and describes my contribution of the GRPO baseline implementation. All errors in interpretation are my own.*

*The GRPO baseline code is part of the VAML repository. The project is paused pending compute access for large-scale experiments.*
