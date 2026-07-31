---
title: "Evaluating Qwen3-VL-2B-Instruct:"
date: 2026-07-30
categories: machine-learning computer-vision
---

# Evaluating Qwen3-VL-2B-Instruct: Building a Pipeline That Doesn't Lose Work

*Personal project · 2026 · PyTorch · Qwen3-VL-2B-Instruct · MMMU benchmark*

---

Most of my work this year has been on the language modeling side: building transformers from scratch, running alignment experiments, comparing PPO against GRPO against DPO. I spent time on the multimodal picture, loading and evaluating a vision-language model locally.

I used Qwen3-VL-2B-Instruct and MMMU as the benchmark: college-level questions spanning multiple disciplines, with images embedded directly into the question. The interesting engineering problem here wasn't squeezing out the highest possible accuracy. It was building a pipeline that survives the real conditions of running a 2B parameter model on constrained, unreliable compute: memory limits, session disconnects, malformed outputs, all of it. This post is about that pipeline, what broke, and what the results actually showed once it worked.

---

## Table of Contents

1. [ modules](#1-five-modules-one-job-each)
2. [Loading the model](#2-loading-the-model-without-guessing-at-device-placement)
3. [Bugs ](#3-three-bugs-that-compounded-into-one-cascade)
4. [Resumability](#4-resumability-and-what-done-actually-means)
5. [compute ](#5-running-on-compute-that-disappears)
6. [Results](#6-results)
7. [What I'd do differently](#7-what-id-do-differently)

---

## 1. Five Modules, One Job Each

The pipeline splits into five files, each with exactly one responsibility and, more importantly, exactly one way it can be wrong.

`data.py` only knows about MMMU's schema: parsing the stringified option lists with `ast.literal_eval` rather than `eval` for safety, and pulling whichever of the seven possible image fields (`image_1` through `image_7`) are actually populated for a given sample, since most questions have one image but some have several. It has no idea how inference works.

`model.py` only knows about loading weights and running a forward pass. It has no idea what MMMU is.

`parser.py` only knows how to pull a letter out of free text.

`pipeline.py` is the one file that knows the other three exist. Its job is to move data between them and log the result. `metrics.py` never touches the model or the dataset directly, it only reads the JSONL file `pipeline.py` already wrote.

The point of drawing the boundaries this way is that a bug in how I parse MMMU's option format can't corrupt how the model loads, and a bug in device placement can't corrupt how a dataset record gets read. Each file fails in isolation, which made debugging the actual failures a lot faster than it would have been in one monolithic script.

---

## 2. Loading the Model Without Guessing at Device Placement

The model loads through `AutoModelForImageTextToText` rather than importing a specific model class by name. Every checkpoint on Hugging Face ships with a config file declaring its own architecture, and the Auto class reads that config and dispatches to the matching implementation automatically. Swap the model ID to a different VLM later and `model.py` needs zero changes.

Device selection falls back gracefully: CUDA if available, then MPS for Apple Silicon, then CPU. Precision is chosen per device, bfloat16 on CUDA, float16 on MPS, float32 on CPU, since defaulting to float32 everywhere roughly doubles the memory footprint of a checkpoint that's already 4.3GB.

```python
def get_dtype(device):
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32
```

`low_cpu_mem_usage=True` streams weights into memory during loading instead of allocating duplicate buffers, which matters more than it sounds like it should when you're working with a machine that doesn't have much headroom to begin with.

The processor gets explicit `min_pixels` and `max_pixels` bounds, `256*28*28` to `1280*28*28`, to cap how many visual patch tokens Qwen3-VL generates per image. Without that bound, a high-resolution MMMU image can drive the attention computation during vision prefill to allocate several gigabytes on its own, entirely separate from the model weights. That's not a hypothetical, it's the specific thing that pushed one of my early runs past available memory before a single token of text had been generated.

---

## 3. Three Bugs That Compounded Into One Cascade

The first full run against all 100 samples came back with `inference_errors: 100` and `completed: 0`. Every single sample failed. Here's what was actually going on, in the order I found it.

**The `torch_dtype` deprecation.** A recent `transformers` release renamed `torch_dtype` to `dtype` in `from_pretrained`. The old name still loaded the model, but only with a warning, and it left real ambiguity about whether the precision I thought I was setting was actually being honored. Renaming the parameter removed the warning and confirmed the model's dtype attribute matched what I expected.

**The dict that didn't have a `.to()` method.** This one is worth walking through carefully, because the failure mode is easy to miss. A model on the GPU needs every input tensor on the GPU too, PyTorch won't run a computation across two different devices. The plan was one line: call `apply_chat_template`, chain `.to(device)` directly onto whatever came back. That pattern works in a lot of example code because in those cases the return value is a wrapper object built with its own `.to()` method that knows how to move every tensor it's holding.

With `return_dict=True` and `return_tensors="pt"` in this specific call, what actually came back was a plain Python dictionary. A plain dict has no `.to()` method, it's just a container. The call crashed instantly:

```
AttributeError: 'dict' object has no attribute 'to'
```

This happened before the model ever ran, before any real computation started, which is exactly why the earliest failed samples logged inference times near zero, the crash was at input preparation, not inference. The fix was splitting the single chained call into two explicit steps: render the chat template as text only (`tokenize=False`), then build tensors in a separate `processor()` call, then iterate the resulting dict and call `.to(device)` on each tensor value individually rather than on the container.

**Prompt text leaking into the logged response.** `generate()` returns the full sequence, the original input tokens plus everything newly generated, concatenated together. My first attempt at trimming the prompt off used string slicing based on the prompt's character length. That broke because the prompt string, built with `tokenize=False`, still contained the chat template's structural markup, while the final decoded output, built with `skip_special_tokens=True`, had that same markup stripped out. Two strings of different lengths representing conceptually "the same" prompt, sliced against each other. The fix was trimming in token space instead: slicing `generated_ids` down to only the newly generated portion, using the input token count, before decoding anything at all.

The three bugs compounded in a specific way. The dtype issue created uncertainty about precision but wasn't itself the crash. The `.to()` crash is what caused every sample to fail identically, at the same line, before generation ever started, which is why the failure pattern was completely uniform rather than a mix of different exceptions. The prompt-trimming bug wasn't part of this crash at all, it only became visible once the device bug was fixed and real completions started coming through.

---

## 4. Resumability, and What "Done" Actually Means

The spec I was working against required the pipeline to survive being killed mid-run and resumed without redoing completed work or silently skipping anything. That shaped `pipeline.py` from the start rather than getting bolted on afterward.

Every sample writes a record to `trajectories.jsonl` the moment it's attempted, append mode, with an explicit flush, never held in memory and written at the end. On startup, the pipeline reads whatever's already there and builds a set of completed sample IDs from it.

The interesting design decision is what "completed" means. A sample counts as done once it has any record at all, success or error, not only on success. If a specific image reliably causes an out-of-memory error, retrying that same sample on every resume would stall the pipeline indefinitely rather than move forward. Logging the failure once and moving on is the correct behavior, and it's a different definition of "done" than the one `metrics.py` uses downstream, where a sample only counts toward `completed` if it actually produced a usable result. Same word, two different jobs, kept deliberately separate.

I tested this by killing the process mid-run with a keyboard interrupt and restarting it. It picked up from the next unprocessed sample, no re-computation, no duplicate records.

---

## 5. Running on Compute That Disappears

My development machine has 4GB of RAM. Loading a 4.3GB checkpoint in float32 on CPU, the default without an explicit dtype, needs roughly 8.5GB, and the OS killed the process before it could finish loading. The math checks out cleanly: the checkpoint is stored in bfloat16, 2 bytes per parameter at 2B parameters, and float32 doubles that.

I moved execution to Colab's T4 GPU with zero changes to any file in `src/`, only where the code runs, not what it does. That surfaced a second problem: Colab's local disk is ephemeral across a full session recycle, not just a soft restart. Both the downloaded model cache and `results/trajectories.jsonl` would be gone if the session died and came back fresh, which would have quietly broken the resumability guarantee I'd just finished building and testing.

The fix was mounting Google Drive and setting `HF_HOME` to a path on it before the model ever loads, plus cloning the repo itself onto Drive so `results/` persists there too:

```python
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
```

With both the weights cache and the results file on persistent storage, resumability held not just within one live session but across a session that died completely and got reconnected from scratch.

---

## 6. Results

```
Total samples: 100
Completed: 100
Inference errors: 0
Overall accuracy: 27.0%
Parse failure rate: 41.0%
Avg inference time: 18.88s
Total runtime: 1900.6s

Per-subject accuracy:
  Art: 52.0%
  Biology: 36.0%
  Architecture & Engineering: 12.0%
  Accounting: 8.0%
```

Art scored highest by a wide margin, and the reason is visible directly in the response pattern: most Art questions are short factual identification ("who painted this," "what is this called"), and the model typically answers in a single word or letter, leaving little room for a parse failure.

Accounting and Architecture/Engineering scored lowest, and here the interesting finding isn't the low accuracy itself, a 2B parameter model attempting multi-step financial and engineering calculations underperforming is not surprising on its own. It's what a chunk of the 41% parse failure rate turned out to actually be. Pulling the raw logged responses behind the failures, several end mid-word or mid-calculation, cut off before ever stating a concluding letter. One response trails off at "...in the trans," clearly heading toward "transmembrane." Another stops at a bare markdown header with nothing after it. This is response truncation from the `max_new_tokens` limit, not the model failing to reach an answer, and it concentrates almost entirely in the subjects that answer with long step-by-step reasoning before committing to a letter. Art rarely truncates because its answers are short enough to finish naturally well inside the token budget.

That's a diagnosis with a visible mechanism sitting in the data, not a guess about why the number is what it is.

---

## 7. What I'd Do Differently

Raise `max_new_tokens` for the reasoning-heavy subjects specifically, since the truncation evidence points there directly. I started a rerun to test this and lost it partway through to a Colab memory issue, so it stays a well-evidenced hypothesis rather than a confirmed fix.

Fix a random seed before generation. Qwen3-VL's default generation config samples rather than decoding greedily, so the exact text of any individual response, and therefore the exact accuracy number, can shift slightly between runs on the same code and the same environment. Everything else about the pipeline, the dependency versions, the sample selection, the parsing and logging logic, is fully reproducible through the pinned lockfile. The model's raw text output on a given sample currently isn't, bit for bit.

Run more than once. Every per-subject percentage here comes from a single pass, and with 25 samples per subject, individual numbers carry real noise that repeated trials would average out.

No batching. Each sample runs as its own forward pass, which is the honest, simple version and the one I'd optimize first if inference speed became the actual bottleneck rather than a secondary concern.

---

*Local inference, no API calls · Qwen3-VL-2B-Instruct · MMMU benchmark, 100-sample subset · uv-managed Python project*
