---
layout: post
title: "Choosing what to train on: Thompson sampling for GRPO data selection"
date: 2026-10-02
categories: [machine-learning, post-training, rl]
---

I had 5,760 rollouts to spend across 120 tasks and no idea which tasks would teach the model anything.

The model is five numbers, one skill level each, between 0 and 1. Its update rule is fixed and the learning rate is given, so neither was mine to change. I controlled one decision, repeated 5,760 times: which task gets the next rollout. The score is the model's pass rate on a held-out evaluation set I never see, and a run prints no score while it trains. The only feedback is the pass or fail of the rollouts I already spent.

This post covers the scheduler I wrote, in two layers. The first assumes every task is safe to train on. The second handles the case where some tasks push the model's skills down instead of up. Both live in one file of roughly 150 lines of Python.

## The setup

The numbers: 120 tasks, a group size of 8, a learning rate of 0.030, a slope of 6.0. Spread evenly over the bank, the budget is six epochs, since 120 tasks × 8 rollouts × 6 = 5,760.

A task demands some level of each skill it needs, and it passes only when every one of those demands is met. Pass probability is a product of sigmoids, one per skill: `sigmoid(6.0 × (prof[s] - demand[s]) + 2.9444)`. A demand is the level at which that skill alone passes 95% of the time, and shortfalls multiply, so being a little short on three skills costs more than being a little short on one. I see each task's shape, meaning its demands relative to the largest one. The scale, which is how hard the task actually is, stays hidden. So does the model's starting proficiency.

## Why only some tasks teach anything

GRPO scores each rollout against its group: reward minus the group mean. If all eight rollouts agree, every advantage is zero and the group produces no gradient. The environment condenses this into one number per closed group, the spread, `k(8-k)/64`, where k is the number of passes.

| passes (k) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| spread | 0 | 0.109 | 0.188 | 0.234 | 0.25 | 0.234 | 0.188 | 0.109 | 0 |

The path from a rollout to a change in the model has several stages, and I found them easy to blur together. Eight rollouts on one task make a group. When the group closes, its spread becomes credit, `LR × spread × share × 8`, split across the skills the task demands in proportion to how much it demands of each. The credit is banked. The model applies it only at the next cadence boundary, when each skill moves by `credit × (1 - prof)`. Between boundaries the model is frozen.

For a task with pass probability p, the expected spread of a group is `0.875 × p(1-p)`, which peaks at p = 0.5. So the target is a pass rate near 0.5, not a high one. A task the model always passes teaches nothing, and neither does one it always fails. Only the tasks near the frontier move the model.

## Why round-robin and greedy both fall short

Round-robin is the baseline. Take the bank in order, spend one group of 8 on each task, wrap around until the budget runs out. Every task gets six groups. Coverage is guaranteed, but the policy treats every task as equally likely to teach something, and that is false. Most of the 720 groups close flat, because most tasks sit well below or well above the model's current skill, and nothing in the policy notices which ones did.

The obvious fix is greedy: track a pass-rate estimate per task and always sample the one closest to 0.5. After one group that estimate rests on eight outcomes. A task that fails its first group by bad luck drops out and stays out. A task the model can't solve yet gets no second look once it can.

Round-robin never learns from its data. Greedy learns too much from too little of it. I needed a policy that explores while it is uncertain and exploits as evidence accumulates.

## Why Thompson sampling

Each task is an arm of a bandit. Its payoff is the closeness of its pass rate to 0.5, and I only learn that payoff by pulling the arm.

Epsilon-greedy picks a random arm with probability epsilon and the best current estimate otherwise. It works, but exploration is uniform: it spends as much on arms I already know are dead as on arms I know nothing about. Epsilon is also a constant I have to tune. Too low and I never find the frontier. Too high and the budget goes to dead tasks.

Upper confidence bound adds a bonus for uncertainty to each estimate and picks the highest total. It has good regret guarantees, but the bonus is a formula with a scaling constant, and it assumes payoffs hold still. Here they drift upward as the model learns.

Thompson sampling keeps a distribution over each arm's payoff, draws one sample from each, and picks the arm with the best draw. Exploration comes out of the variance. An uncertain arm has a wide distribution, so its draws swing high often enough to get picked. A confident arm has a narrow one and gets picked on merit. There is no exploration constant. Pass/fail outcomes also pair with a Beta distribution, whose update is a single increment, so the whole method fits in a few lines.

## The scheduler

Each task keeps a Beta posterior over its current pass rate, starting at Beta(1,1), a uniform prior with no information. A pass increments alpha and a fail increments beta. The posterior mean follows the observed rate, and its width follows how much evidence I have.

```python
def __init__(self, tasks, budget, cadence, group, seed):
    self.tids = [t["task_id"] for t in tasks]
    self.group = int(group)
    self.rng = random.Random(seed)
    self.alpha = {t: 1.0 for t in self.tids}
    self.beta = {t: 1.0 for t in self.tids}
    self.n = {t: 0 for t in self.tids}
    self.min_visits = self.group
```

To choose a task, I draw one sample p from each posterior and score it with `1 - 2|p - 0.5|`. That function is a tent peaking at 0.5. It is not the exact spread function, but it peaks in the same place as the expected spread, so maximizing it pushes toward the same tasks.

```python
def _score(self, tid):
    p = self.rng.betavariate(self.alpha[tid], self.beta[tid])
    return 1.0 - abs(p - 0.5) * 2.0
```

What Thompson samples here is tasks, not groups. Each draw is one plausible value of a task's current pass rate. The scheduler never decides anything about groups directly. It decides which task gets the next rollout, and a group is what a task's rollouts add up to.

Two examples. After its first group, a task that went 0 for 8 has a Beta(1,9) posterior with a mean near 0.1, so its draws score about 0.2. A task that went 4 for 8 has Beta(5,5), whose draws cluster around 0.5 and score about 0.75 on average. The second task wins most draws, but the first still gets picked now and then, because a single draw from a posterior with this much spread can land well above its mean.

```python
def choose(self):
    # visit floor: every task gets one group before the bandit takes over
    for tid in self.tids:
        if self.n[tid] < self.min_visits:
            return tid
    best, best_score = None, -1e9
    for tid in self.tids:
        s = self._score(tid)
        if s > best_score:
            best_score = s
            best = tid
    return best

def observe(self, task_id, passed):
    if passed:
        self.alpha[task_id] += 1.0
    else:
        self.beta[task_id] += 1.0
    self.n[task_id] += 1
```

The one piece that isn't Thompson sampling is the visit floor. Every task gets one full group before the bandit starts, so the scheduler never ranks a task it has not measured. The floor costs a full epoch, 960 of the 5,760 rollouts. It also has a limit: it guarantees one look, not a second chance. A task that fails its floor group lands at Beta(1,9) and is sampled only occasionally from then on.

## When some tasks poison the model

The scheduler above assumes every task is safe. In the harder version, some tasks land their credit with the sign flipped, so training on them pushes down the very skills the task demands. Nothing marks them. A poisoned group and a healthy group look identical from the outside, with the same size, the same spread and the same pass/fail outcomes. The difference only appears downstream, when the credit lands and the model gets better or worse on other tasks. The share of poisoned tasks is drawn fresh each run, anywhere from 0 to 40% of the bank, and zero is possible.

This breaks the bandit framing. A bandit assumes each arm has a payoff to maximize. Here some arms carry a negative payoff I can't observe directly, and pulling them damages the model in ways that persist. Detection becomes a credit-assignment problem sitting on top of the selection problem.

The right signal is cross-task: train on task A and watch whether unrelated tasks get worse. That needs a skill-attribution matrix mapping each task to the skills it shares with the others, and I didn't build one. I used a proxy. The model is learning, so a healthy task's pass rate should drift upward with the global trend. A task whose rate lags that trend is either poisoned or hard, and the proxy cannot tell which.

## The poisoned scheduler

The base policy stays the same. On top of it sit a suspicion score per task and two exponentially weighted moving averages: one per task with a short memory of roughly the last four rollouts, and one global across all tasks. Every rollout updates both.

Once a task has at least two groups of evidence (16 rollouts) and its average sits at least 0.10 below the global rate, its suspicion grows by 0.05 per rollout. Otherwise it decays by 0.02. Growth outweighs decay on purpose, so one good window does not clear a task. The arithmetic makes the threshold concrete: suspicion rises whenever a task sits below the line in more than about 29% of its rollouts. When suspicion passes 0.5 and the task has at least three groups of evidence (24 rollouts), it is quarantined.

```python
def observe(self, task_id, passed):
    super().observe(task_id, passed)
    y = 1.0 if passed else 0.0
    self.ewma[task_id] = (1 - self.ewma_alpha) * self.ewma[task_id] + self.ewma_alpha * y
    self._observe_global(passed)

    if self.n[task_id] >= 2 * self.group:
        if self.ewma[task_id] < self._global_rate - 0.10:
            self.suspicion[task_id] += 0.05
        else:
            self.suspicion[task_id] = max(0.0, self.suspicion[task_id] - 0.02)

    if self.suspicion[task_id] > 0.5 and self.n[task_id] >= 3 * self.group:
        self.quarantined.add(task_id)
```

A false positive costs a healthy task its share of the budget, so I added two guardrails.

The first is a soft penalty ahead of the hard exclusion. In `choose`, a task's score drops by twice its suspicion. The base score runs from 0 to 1, so at a suspicion of 0.5 the penalty equals the best score a task can earn, which is exactly where quarantine starts. Below that, mild suspicion tilts the decision without removing the task.

```python
def choose(self):
    for tid in self.tids:
        if tid in self.quarantined:
            continue
        if self.n[tid] < self.min_visits:
            return tid
    best, best_score = None, -1e9
    for tid in self.tids:
        if tid in self.quarantined:
            continue
        s = self._score(tid) - 2.0 * self.suspicion[tid]
        if s > best_score:
            best_score = s
            best = tid
    return best
```

The second is a retest. At the halfway point every quarantined task gets one more chance: it leaves quarantine, gets a probation group, and has its suspicion halved rather than reset. A task that was poisoned crosses the threshold again quickly. A task that was only hard, and has since caught up, stays in.

```python
def choose(self):
    if sum(self.n.values()) >= self._retest_at:
        for tid in list(self.quarantined):
            if tid not in self._retested:
                self._retested.add(tid)
                self.quarantined.discard(tid)
                self.probation[tid] = self.group
                self.suspicion[tid] *= 0.5
                return tid
    # ... then probation, then visit floor, then bandit
```

No threshold is right for both a clean bank and one that is 40% poisoned, and I don't know which I'm in. The policy is a compromise tuned for the middle. On a clean bank the guardrails cost a few rollouts on retests. On a heavily poisoned bank some poison gets through.

## Where it falls short

The proxy is the largest gap. A task that lags the global rate looks the same whether it is poisoned or just hard, and the scheduler punishes both. The cross-task signal would separate them, and I would build the skill-attribution matrix first.

The Beta counts also only accumulate. As the model improves, outcomes from early in the run keep their full weight, so a task's posterior lags its true pass rate. That is why a task that failed early needs several lucky draws before its estimate moves.

## What I took from it

Pass rate is the wrong target when the learning signal comes from disagreement. The target is the frontier, and finding it is a different problem from maximizing performance.

Thompson sampling works here because it acts on what I don't know. A point estimate throws away the fact that I'm unsure, while a posterior keeps it and the policy uses it.

Detection without the right signal ends up as a proxy. I kept the per-task version because it was cheap and said plainly that the cross-task version is the correct one.

The code is on my GitHub if you want to read the implementation.
