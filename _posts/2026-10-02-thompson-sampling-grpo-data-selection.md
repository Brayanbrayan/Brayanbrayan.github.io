---
layout: post
title: "Choosing what to train on: Thompson sampling for GRPO data selection"
date: 2026-10-02
categories: [machine-learning, post-training, rl]
---

You have 120 tasks, a budget of 5,760 rollouts, and a group size of 8. That's roughly six epochs over the dataset. The model's update rule is fixed, the learning rate is given, the slope is given. The only thing you control is which task the next rollout goes to.

Your score is the model's average pass rate on a held-out evaluation set you never see. You don't know how hard any task is. You don't know the model's current skill on any of the five skills. You don't know which tasks are worth training on and which are already solved or out of reach. All of that arrives incrementally, as a consequence of the sampling decisions you already made.

This is the scheduling problem in post-training, stripped to its bones. It's the same shape as picking what to spend GPU hours on when you have more data than compute: you can't afford to run everything, so you have to choose, and you have to choose without knowing what you're choosing between.

This post is about how I solved it. Two layers: a scheduler for the case where every task is safe to train on, and a second layer for when some tasks actively damage the model. Both are in the same file, roughly 150 lines of Python.

## The constraint that shapes everything

Before the scheduler, the thing it has to schedule around.

GRPO learns from disagreement within a group. Sample a group of rollouts on a task, score them, and the advantage of each rollout is its reward minus the group mean. If every rollout in the group earns the same reward, every advantage is zero and the group contributes no gradient. The group's spread term `k(8-k)/64`, where k is the number of passes, captures this. It's zero at k=0 and k=8, and peaks at k=4.

The practical consequence: a task only produces learning when its current pass rate is near 0.5. A task the model always passes is a solved task; the budget spent on it buys nothing. A task the model always fails is an out-of-reach task; same waste. Only tasks near the frontier move the model.

In a normal training run, this doesn't matter much, because you run epochs over the dataset and the frontier tasks produce gradient wherever they happen to be. With a fixed budget and a task bank larger than you can exhaustively sample, it matters enormously. Every rollout spent on a solved or unreachable task is a rollout not spent on the frontier, and the total number of frontiers you can find is capped by the budget.

So the scheduler's job is: find tasks whose current pass rate is near 0.5, and spend as much of the budget there as possible. That's it. Everything else is a mechanism for doing that under uncertainty.

## What a normal scheduler looks like

The baseline is uniform round-robin. Take the task bank in order, spend 8 rollouts on each (one group), then move to the next, wrap around, repeat until the budget runs out. Six epochs over 120 tasks at group size 8 is exactly 5,760 rollouts.

This is the right thing to do if you know nothing and have no way to learn anything. It treats every task as equally likely to be informative, and it guarantees coverage: every task gets exactly six groups, so no task is starved.

It's also badly wasteful. Most of those 720 groups will close flat, because most tasks are either well below or well above the model's current skill. The baseline spends the same budget on tasks that were never going to teach anything as it does on tasks sitting right at the frontier. The waste is structural, and it doesn't self-correct because the policy has no mechanism for noticing which groups went flat and adjusting.

A slightly better naive policy: keep a running pass-rate estimate per task and sample the task whose estimate is closest to 0.5. This is greedy. It exploits the current best estimate and never explores past it. If a task's estimate looks mediocre after one group, the policy never revisits it, even though one group is nowhere near enough evidence to write it off. A task that got unlucky on its first group is permanently excluded, and a task that would become frontier later (once the model picks up the skills it needs) never gets another chance.

The gap between round-robin and greedy is the gap between not learning from your data and over-learning from too little of it. You need something in between: explore when uncertain, exploit when confident, and shift from one to the other as evidence accumulates.

## Why Thompson sampling

The problem is a multi-armed bandit. Each task is an arm with an unknown payoff, and the payoff here isn't the pass rate. It's the *closeness of the pass rate to 0.5*. You want to pull the arm whose current payoff is highest, but you only learn the payoff by pulling.

There are three standard families for this.

### Epsilon-greedy

With probability epsilon, pick a random arm; otherwise pick the arm with the best current estimate. Simple, and it works. The problem is that exploration is uniform and unconditional: when the policy explores, it explores every arm with equal probability, including ones you've already ruled out. And the exploration rate is a fixed constant you have to tune. Too low and you never find the frontier; too high and you spend most of the budget on tasks you already know are dead.

### Upper confidence bound (UCB)

Pick the arm with the highest upper confidence bound on its payoff (the estimate plus a bonus for how uncertain it is). This is a principled exploration policy, and it's optimal in several regret senses. But the bonus term is a formula: it depends on the number of pulls and a scaling constant, and getting it right requires knowing something about the reward distribution. It's brittle when the reward is non-stationary, which it is here, because the model is learning and every task's pass rate is drifting upward over time.

### Thompson sampling

Maintain a probability distribution over each arm's payoff, sample from it, pick the arm whose sample is best. Exploration falls out of the variance of the distribution: uncertain arms have wide distributions and their samples vary a lot, so they occasionally get picked. Confident arms have narrow distributions and get picked (or not) on merit. No exploration parameter to tune. No bonus formula. And because the posterior is updated Bayesianly as evidence comes in, the exploration decays naturally and adapts to non-stationarity better than UCB's fixed bonus.

Thompson sampling was the right fit for three reasons. The reward is non-stationary (the model learns, pass rates drift), and Thompson adapts more gracefully than UCB. The natural distribution for pass/fail is Beta, which has closed-form updates and is trivial to implement. And the exploration is uncertainty-driven rather than uniform, which matters when the budget is tight and most arms are known to be bad.

## The scheduler

Each task gets a Beta posterior over its current pass rate, starting from Beta(1,1), a uniform prior, no information. A pass increments the first parameter, a fail increments the second. The posterior mean tracks the observed rate and the posterior width tracks how much evidence you have.

To choose a task, draw one sample from each task's posterior and score the sample by `1 - 2|p - 0.5|`. That function peaks at p=0.5 and falls off symmetrically. It's not the exact spread function, but it has the same shape (same peak, same direction) so maximizing it maximizes expected spread. The task with the highest score gets the next rollout.

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

```python
def _score(self, tid):
    p = self.rng.betavariate(self.alpha[tid], self.beta[tid])
    return 1.0 - abs(p - 0.5) * 2.0
```

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

The one piece that isn't Thompson sampling is the visit floor. Every task gets one full group before the bandit can deprioritize it. A task that starts flat-fail gets a posterior centered near zero, and the bandit will basically never revisit it. But flat-fail now doesn't mean flat-fail forever (the model is learning, and a task out of reach in epoch 1 might be on the frontier by epoch 4). The visit floor is cheap insurance against permanently writing off a task based on one unlucky group. It's the difference between a policy that's optimal given its beliefs and one that's robust to its beliefs being wrong.

## When some tasks fight back

The scheduler above assumes every task is safe. The harder version of the problem is when some tasks land their credit with the sign flipped. Training on them actively pushes the model's skills down, on the very skills the task demands. And nothing marks them. A poisoned group and a healthy group look identical from the outside: same size, same spread, same pass/fail outcomes. The difference only shows up downstream, when the credit lands and the model either improves or degrades on other tasks.

This breaks the bandit framing. A bandit assumes each arm has a payoff you're trying to maximize. Here, some arms have negative payoff you can't observe directly, and pulling them damages the model in ways that persist. The detection problem is a credit-assignment problem layered on top of the selection problem.

The signal I use is a proxy, and the weakness is worth stating plainly. The model is learning, so every healthy task's pass rate should drift upward over time, tracking the model's overall improvement. A task whose rate lags the global trend is either poisoned or just genuinely hard. The proxy can't distinguish those two cases. A stronger signal would be cross-task: train on task A, watch whether unrelated tasks get worse. That requires knowing which tasks are unrelated, a skill-attribution matrix mapping tasks to the skills they share, which I didn't build in the time I had. The per-task proxy captures some of the signal at much lower complexity, and that was the tradeoff.

## The poisoned scheduler

The base policy is the same. On top of it, a suspicion score per task, and two exponentially weighted moving averages: one per task tracking its recent pass rate, one global tracking the model's recent pass rate overall.

Every rollout updates both EWMAs. If a task has at least two groups of evidence and its EWMA sits at least 0.10 below the global rate, suspicion grows by 0.05. Otherwise it decays by 0.02. The asymmetry is deliberate: accusations are expensive to undo, so sustained evidence should build suspicion quickly and a single good window shouldn't fully exonerate a task.

Once suspicion crosses 0.5 and the task has at least three groups of evidence, it's quarantined.

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

Two guardrails keep the quarantine from being too aggressive, because a false positive costs a healthy task its budget.

First, suspicion is a soft penalty before it's a hard exclusion. In choose, a task's score is reduced by twice its suspicion. The base score is in [-1, 1] and suspicion is in [0, 1], so the penalty is large enough to override any frontier score. A task with mild suspicion still gets sampled occasionally; only sustained evidence pushes it into the quarantined set.

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

Second, every quarantined task gets one more chance at the halfway point, with its suspicion halved rather than reset. A genuinely poisoned task re-quarantines after a few more groups; a falsely accused one stays out. Half-measures on re-entry, not full exoneration.

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

There's no threshold that's right for both a bank with no poison and a bank with 40% poison, and the poison fraction is redrawn every run. The policy is a compromise tuned for the middle. If the bank is clean, the guardrails cost a little budget on re-tests. If the bank is heavily poisoned, some poison slips through. Both are acceptable; neither is optimal.

## What I took away

The objective isn't what it looks like. "Pass rate" is the wrong target when the learning signal comes from disagreement. The right target is the frontier, and finding the frontier is a different problem than maximizing performance.

Uncertainty is a resource. Thompson sampling works because it treats what you don't know as something to act on, not something to average away. A point estimate throws away the information that you're unsure. A posterior keeps it, and the policy uses it.

Every detection problem is a proxy problem until you've built the infrastructure for the real signal. The per-task poison proxy is worse than the cross-task version. Naming the gap is more useful than pretending the proxy is complete.

The code is on my GitHub if you want to read the implementation.
