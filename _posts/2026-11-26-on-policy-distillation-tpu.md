---
layout: post
title: "Reproducing on-policy distillation on a single TPU chip"
date: 2026-11-26
---

# Reproducing on-policy distillation on a single TPU chip

Nov 26, 2026

*Personal project · 2026 · JAX, Tunix · Qwen3-0.6B student, Qwen3-1.7B teacher · GSM8K*

---

This project started as a reproduction. Thinking Machines Lab's post on [on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) describes a way to train a small model from a larger one: the student writes its own answers and the teacher scores every token. I wanted to run that loop myself, end to end, on hardware I could get for free, which meant one Colab TPU v5e chip with 15.75 GiB of memory and no local GPU.

It did not stay a reproduction. I added SFT and GRPO baselines at the same token budget so I could say something about efficiency, and then each surprising result produced another experiment: two more learning rates, a larger adapter, a warm start from SFT, a second seed. Two of my early results were wrong because of bugs in my setup, and finding those took longer than running the experiments.

Where I ended up. A Qwen3-0.6B student scores 65.5% on GSM8K (the first 1,312 test problems, 640-token answer cap), and its Qwen3-1.7B teacher scores 82.0%. After about a million tokens of training, six runs land between 66.9% and 68.1%: SFT on teacher answers, SFT on only the correct ones, distillation from two seeds, distillation warm-started from SFT, and GRPO at a lower learning rate, which uses no teacher at all. That is 1.4 to 2.6 points above the untrained student, and every one of those intervals includes zero. Two seeds of the same distillation setup differ by 1.2 points, so I cannot rank the methods. The per-token gap between student and teacher fell by about 30% and reproduced across seeds, while accuracy barely followed.

Code, logs and per-problem results are in the [repository](https://github.com/Brayanbrayan/on-policy-distillation).

## The method

Off-policy distillation trains the student on text the teacher wrote. At test time the student continues from its own text, which can drift somewhere the teacher never showed it. Reinforcement learning has the opposite shape: the training text is the student's own, but the feedback is one reward for the whole answer. On-policy distillation keeps the student's own text and replaces the single reward with feedback on every token.

For each token the student sampled, take the teacher's log-probability of that token and the student's log-probability of it, both conditioned on the same prefix:

```
A_t = log p_teacher(x_t | prefix) - log p_student(x_t | prefix)
```

Averaged over samples from the student, the negative of this is the reverse KL divergence between student and teacher at that position. Using `A_t` as a per-token advantage in a policy-gradient update pushes the student's distribution towards the teacher's on the states the student actually visits. The Thinking Machines post uses a discount factor of zero, so each token is credited only for itself and not for what follows. The teacher writes nothing. It reads.

The post argues that this dense signal carries far more information per sample than one reward per answer, and reports a large compute saving over RL. I wrote the loop from the post's description. Their implementation runs on Tinker, their training service, and I did not use it. Their experiments also use larger models and AIME-level math, so nothing here reproduces their numbers. What I tested is whether the efficiency story shows up at 0.6B parameters, one chip and GSM8K.

## The setup

| Component | Choice |
| --- | --- |
| Hardware | Colab TPU v5e, one chip, 15.75 GiB |
| Stack | Tunix 0.1.7 (JAX, Flax NNX), JAX 0.11.2, bf16 weights |
| Student | Qwen3-0.6B, 1.11 GiB |
| Teacher | Qwen3-1.7B, 3.21 GiB |
| Trained parameters | LoRA rank 16, alpha 32, on the q, k, v, o, gate, up and down projections of every layer: 10.09M parameters |
| Optimizer | AdamW, peak learning rate 1e-4, warmup, cosine decay to 10% of peak, gradient clipping at 1.0 for distillation and GRPO |
| Training prompts | GSM8K train questions (7,473), prompted for a final answer in `\boxed{}`, non-thinking mode |
| Evaluation | First 1,312 GSM8K test problems, greedy decoding, answers capped at 640 tokens |

On-policy distillation needs the teacher's probability for the exact token IDs the student sampled, so the two models must share a vocabulary. I first considered a Gemma teacher with a Qwen student, but Gemma 3's vocabulary has about 262k entries and Qwen3's about 152k. A same-family Gemma pair would have worked: the Gemma 3 4B and 1B tokenizers are identical. Tunix 0.1.7 could not load either, though. It raised `NotImplementedError` and said it supports Gemma 3 only from GCS, so I moved to Qwen3. Qwen3-0.6B and Qwen3-1.7B encode text identically (151,669 entries).

One Colab-specific trap. Installing Tunix pins a `libtpu` version that Colab's preinstalled `torch-tpu` backend rejects the moment anything imports torch, and `tensorboardX` does so indirectly. Setting `TORCH_DEVICE_BACKEND_AUTOLOAD=0` before the first import avoids the crash. Nothing here runs torch on the TPU.

All three methods train the same adapter, so the comparison is about the training signal and not the number of trainable parameters.

SFT trains on 3,712 teacher answers to GSM8K training questions, 1.0M tokens, sampled at temperature 1.0 and left unfiltered. The loss covers the answer tokens, stop token included. Two epochs at an effective batch of 32 give 232 optimizer steps.

Distillation takes 16 training questions per step, one student answer each, for 230 steps and about 1.0M student-written tokens. The student samples at temperature 1.0 with answers capped at 384 tokens. The teacher then scores the same text, the advantage is the quantity above with no gradient through it, and the loss is the usual policy-gradient form:

```python
lp = token_logps(student, prompt, answer, pad_id, eos_id)   # student log-probs, with gradient
adv = stop_gradient(teacher_lp - lp)                         # one advantage per token
loss = -(adv * lp * mask).sum() / mask.sum()                 # mask: answer tokens only
```

There is one update per batch, so every answer was written by the exact weights being updated.

GRPO takes 4 questions with 4 answers each per step, also for 230 steps. The reward is 1 when the final boxed number is correct, the advantage is the reward normalised within the group of four, and it is applied to every token of the answer. There is no KL penalty and no teacher.

## Fitting it on one chip

The two models take 4.32 GiB in bf16, which leaves about 11 GiB. Most of that goes to the KV cache. Tunix's sampler sizes its cache with the model config's dtype, which is float32 for Qwen3 here, so the cache costs 28 layers x 2 x 8 KV heads x 128 x 4 bytes, about 224 KiB per token per sequence. Only the student writes text, so only the student needs a cache. The teacher reads finished text in one forward pass.

A batch of 32 with a 512-token cache (16,384 positions, about 3.5 GiB) fit, peaking at 8.40 GiB. A batch of 64 with the same cache, and a batch of 32 with a 1,024-token cache, are both 32,768 positions. Both failed to load the decode program, which asked for 7.26 GB. The requirement scales with batch times cache length, so I fixed shapes early. Every new shape also cost about 24 seconds of compilation. Student sampling ran at about 1,700 tokens per second at batch 16 and 2,600 at batch 32.

The logits are the other memory problem. With a 152k vocabulary, 32 sequences of 512 tokens produce about 9.3 GiB of float32 logits by my arithmetic, so scoring and the gradient pass run in micro-batches of 4. A distillation step takes about 5.8 seconds: 3.6 sampling, 0.3 for the teacher, 1.9 for the update. The teacher's pass is about 5% of a step. At a 3x parameter ratio, the expensive part is the student writing, not the teacher reading.

Before trusting the loss I compared the sampler's per-token log-probabilities with a teacher-forced pass over the same sequences. The mean absolute difference was 0.002 nats per token, the maximum about 0.05, and 128-token sums agreed within 0.1 nats. That also confirmed my padding layout: prompts padded on the left, answers on the right. I still compute both the student's and the teacher's log-probs with the same Tunix function (`compute_per_token_logps`), so the advantage and the gradient share numerics. The sampler's values are for monitoring only.

The sampler also strips the stop token from the tokens it returns. If I scored only those tokens, the student would never be graded on when to stop. So I append `<|im_end|>` to any answer that ended before the cap, and leave answers that hit the cap without one. I saw this behaviour in one short test and did not check the case of an answer that ends exactly at the cap.

## The first results were wrong twice

### The sampler was greedy

I found this through GRPO. In a 20-step smoke test the reward column only ever showed 0.25, 0.5, 0.75 or 1.0, and the gradient norm was exactly 0.000 at every step. The four answers to each question were identical, so every group had zero variance and no learning signal.

In Tunix 0.1.7 the sampler's `temperature` defaults to 0.0. Passing `temperature=1.0` with no `top_p` or `top_k` still decoded greedily. Across 8 prompts and 64 tokens, zero tokens differed from temperature 0, reversing the batch order changed nothing, and `seed=1` and `seed=2` gave the same output. With `top_p=1.0`, four copies of one prompt gave four different answers, and different seeds gave different outputs. I did not trace why in the source. What I did see is that the sampler derives its random key from a fixed default seed when none is passed, so I also pass a new seed on every step:

```python
out = sampler(input_strings=prompts, max_generation_steps=384, max_prompt_length=256,
              temperature=1.0, top_k=None, top_p=1.0, seed=run_seed * 100000 + step,
              eos_tokens=eos)
```

Each script now fails at startup if identical prompts return identical outputs.

GRPO was the cheap place to find it. The same bug had already spoiled my first distillation run, whose rollouts were the student's greedy answers, and my first SFT dataset, which was the teacher's greedy output and not samples. The clue was in the distillation log, and I explained it away at the time. The estimator `log p_student - log p_teacher`, averaged over tokens drawn from the student, is a KL divergence and cannot be negative in expectation. In the second half of that run it averaged about -0.01, with single steps down to -0.07. Greedy tokens are not draws from the student, so the identity does not hold. After the fix the estimate stayed above 0.14 on every step of every later run.

### GRPO that never stopped

With sampling fixed, the first full GRPO run rewarded a correct final boxed number and nothing else. Within about 65 steps the share of answers that stopped on their own fell from 0.95 to zero, and it stayed there. Every answer ran to the 384-token cap.

![Mean answer length and the share of answers that stop on their own, for distillation and the first GRPO run](https://brayanbrayan.github.io/images/opdtpublog/length_and_finish.png)

*Figure 1. Left: mean answer length in tokens. Right: share of answers that stop on their own. Both are averaged over each step's 16 training answers and smoothed over 10 steps. Blue is distillation (seed 0), orange is the first GRPO run, whose reward ignored stopping.*

That run still scored 46.4% on my 192-problem monitor, because a correct number can appear before the model rambles on. My explanation, which I did not test, is that the reward cannot tell a finished answer from an unfinished one, and the stop token only appears in answers that finish, so nothing pushed it up while "keep going" got rewarded along with the correct number.

Requiring an answer to finish before it earns reward fixed the stopping, and at learning rate 1e-4 the student found a shortcut. Its answers shrank to 169 tokens on average, against 259 for the untrained student, with 96% of them boxed and finished. Accuracy on the full test set fell 28.7 points to 36.9%. It had traded reasoning for finishing. At learning rate 3e-5, with the same reward, answers kept their length (232 tokens) and accuracy ended at 68.0%.

## Measuring it properly

My first exam was the first 192 test problems with answers capped at 384 tokens. It was the wrong instrument, and it took me too long to see it.

A single accuracy on 192 problems carries about plus or minus 7 points. Two distillation seeds with identical settings were 13 points apart at step 50 (43.8% and 56.8%). I had read the first as a sign the learning rate was too high.

![Accuracy on the 192-problem monitor against student-generated tokens for distillation and GRPO runs](https://brayanbrayan.github.io/images/opdtpublog/accuracy_vs_tokens_final.png)

*Figure 2. The 192-problem monitor (384-token cap) against student tokens generated, for the distillation seeds and learning rates and the two GRPO runs. Each point carries a 95% interval of about plus or minus 7 points. Compare the two distillation seeds at the first checkpoint.*

The cap mattered as well. Raising it from 384 to 640 tokens added about 8 points for the teacher (74.1% to 82.0% on all 1,319 test problems) and 4 for the student (61.4% to 65.5%). At 384 tokens, 16.2% of the teacher's answers were cut off, against 1.8% at 640. A cap that hits two models by different amounts distorts any comparison between them.

The two errors compound. At step 230, GRPO at learning rate 3e-5 read 70.8% on the 192-problem monitor, within two points of the teacher's 72.9% on that slice. On the full set at the 640 cap it scored 68.0%, which is 2.4 points above the untrained student. The untrained student itself scores 58.3% on the first 192 problems at the short cap and 65.5% on the full set at the long one.

So the final protocol is this. Every checkpoint is evaluated greedily on the first 1,312 test problems (a multiple of the batch size of 16) at a 640-token cap, and compared against the untrained student on the same problems with a paired bootstrap over problems (2,000 resamples). Per-problem results are saved. The teacher is measured on all 1,319. At this size an interval is about plus or minus 2.5 points wide, so a difference smaller than that is not resolved, and two seeds of one setup differ by about a point.

## Results

| Run | pass@1 | Change vs untrained student (95% interval, points) |
| --- | --- | --- |
| Untrained student | 65.5% | n/a |
| SFT, all teacher answers | 67.8% | +2.3 (-0.1 to +4.8) |
| SFT, correct teacher answers only | 67.2% | +1.6 (-0.8 to +4.0) |
| Distillation, seed 0 | 66.9% | +1.4 (-1.1 to +4.0) |
| Distillation, seed 1 | 68.1% | +2.6 (-0.2 to +5.1) |
| Distillation, warm start from SFT | 67.3% | +1.8 (-0.8 to +4.3) |
| Distillation, lr 3e-5 | 65.6% | +0.1 (-2.4 to +2.5) |
| GRPO, lr 3e-5 | 68.0% | +2.4 (-0.2 to +5.1) |
| GRPO, lr 1e-4 | 36.9% | -28.7 (-31.3 to -25.9) |
| Teacher, Qwen3-1.7B | 82.0% | +16.6 (+13.9 to +19.2) |

*Table 1. First 1,312 GSM8K test problems, greedy decoding, 640-token cap. Both GRPO rows require an answer to finish to earn reward. The teacher's pass@1 is over all 1,319 problems and its paired change is over the first 1,312.*

![Paired change in accuracy against the untrained student for each run, with 95% intervals](https://brayanbrayan.github.io/images/opdtpublog/final_paired_change.png)

*Figure 3. Change in accuracy against the untrained student, in points, with paired 95% bootstrap intervals. The right panel zooms on the student-training runs.*

The teacher is 16.6 points ahead of the student, so there is room. The student-training runs used a small part of it. SFT, filtered SFT, two seeds of distillation, the warm start and GRPO at 3e-5 all sit within 1.2 points of each other, and I cannot separate them.

GRPO at 3e-5 changed how I read the rest. It uses no teacher, and it lands in the same cluster. Whatever produces the roughly two-point lift, if it is a lift, is not specific to the teacher's signal. The shared ingredients are the GSM8K training questions, the prompt format and a LoRA update. I did not run the control that would separate them, such as SFT on the student's own samples.

Filtering SFT to correct answers did not help (+1.6 against +2.3). About 20% of the teacher's answers were wrong, and I expected removing them to matter. The filtered run also trained on 745k answer tokens per epoch against 1.005M, which is a confound.

Compute is the one place where the methods differ clearly.

| Run | Tokens generated | Compute on the chip |
| --- | --- | --- |
| SFT | 1.00M by the teacher | about 12 min sampling, 5 min training |
| SFT, correct answers only | reuses the same data | 4.4 min training |
| Distillation | 1.01M by the student, 1.02M scored by the teacher | 24 to 25 min |
| GRPO, lr 3e-5 | 0.85M | 21 min |
| GRPO, lr 1e-4 | 0.68M | 19 min |
| Exam, per checkpoint | n/a | about 8 min |

*Table 2. Measured on one v5e chip. SFT sampling ran at about 1,380 tokens per second.*

SFT matched distillation for about 17 minutes in all against 24. The efficiency advantage I went looking for did not appear at this scale.

## What the student-teacher gap did

The one clean effect in the logs is the reverse KL itself.

![Mean per-token reverse KL estimate over 230 distillation steps](https://brayanbrayan.github.io/images/opdtpublog/reverse_kl.png)

*Figure 4. The estimate `log p_student - log p_teacher` averaged over all answer tokens in each step (light line) and its 10-step mean (orange), distillation seed 0.*

Averaged over blocks of ten steps, it fell from about 0.34 nats per token at the start to 0.26 around step 105 and 0.24 at the end, and then stopped falling. Seed 1 gave 0.36, 0.25 and 0.23. A 28% drop that reproduces across seeds is the method working as described. For scale, 0.24 nats per token over a 280-token answer is about 67 nats per answer, so a large gap remains in absolute terms.

The warm start makes the picture sharper. SFT alone reached a gap of about 0.24 in five minutes, which is where cold-start distillation ends after 230 steps. Distillation started from the SFT weights began at 0.24 and ended near 0.21. Its accuracy was 67.3%, indistinguishable from the others.

I read this as the student picking up the teacher's phrasing and structure faster than its reasoning, and the gap measure mostly tracking the first. That is an interpretation. I never separated style from reasoning in the data.

## Changes that did not help

I changed one thing at a time against the baseline, and none of the changes helped.

Learning rate 3e-5 brought the gap down faster early (about 0.25 against 0.32 over steps 41 to 50), reached the same plateau, and scored +0.1 on the full set. Learning rate 3e-4 was worse. I stopped it at step 50 because the gap over steps 41 to 50 averaged 0.41 against 0.32 for the baseline, and I did not evaluate it.

Rank 64 with alpha 128 (40.4M trainable parameters) was worse in training. The gap rose from 0.44 to 0.91 by step 21 and ended near 0.33. Gradient norms before clipping were 10 to 25, against 4 to 8 for rank 16. On the 192-problem monitor it scored 43.8% at step 230, where the untrained student scores 58.3%. I did not evaluate it on the full set. It changed step size together with capacity, so it says nothing clean about capacity. A fair capacity test would lower the learning rate alongside the rank, and I did not run it.

## What I would and would not claim

The loop works as the post describes it. The estimator behaves like a KL divergence once sampling is real, the gap falls, and the teacher's pass costs about 5% of a step.

At this scale and budget I cannot rank SFT, distillation and GRPO, and the teacher-free GRPO run landing in the same cluster means I cannot attribute the lift to the teacher. SFT was cheaper than distillation for the same accuracy. That is weak evidence against the efficiency story, from one model pair, one dataset, about a million tokens and mostly one seed per setting, so it says little about the post's setting.

I checked that `top_p=1.0` gives varied outputs and that the reverse KL behaves like a KL. I did not test the sampler's output distribution against the model's probabilities beyond one weak check on a prompt whose first token has probability 0.999. The LoRA weights are stored in bf16, and I did not test whether fp32 master copies change anything.

## Not run

Three times more teacher data. A larger rollout batch at the same token budget. A Qwen3-4B teacher, whose memory fit I did not measure. Twice the training steps. The capacity test described above. The control of SFT on the student's own samples. A Gemma teacher, which Tunix 0.1.7 blocked. The top-K truncated reverse-KL variant from [Revisiting On-Policy Distillation](https://arxiv.org/abs/2603.25562), which argues that the sampled-token estimator I used is biased relative to the sequence-level objective but has lower variance.

## Code and data

Everything is in [github.com/Brayanbrayan/on-policy-distillation](https://github.com/Brayanbrayan/on-policy-distillation). The training scripts are `scripts/train_sft.py`, `scripts/train_opd.py` and `scripts/train_grpo.py`. The exam is `scripts/eval_ckpt.py`, which writes per-problem results to `results/final/`. The run logs are under `results/runs/`, and `docs/setup-notes.md` records the measurements behind the memory and speed numbers.
