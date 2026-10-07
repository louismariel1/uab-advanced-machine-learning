## Practical 4 — Chunk 3: Exam Compression

### 1\. Main purpose of this chunk

This part evaluates **LoRA vs Dense Fine-Tuning (FT)** in three main areas:

- **Model/state-dict size**
- **Optimizer state size**
- **Training speed**
- Then it combines everything into a **final LoRA vs Dense conclusion**.

The best LoRA model is taken as **`r = 32`** in this chunk.

---

## 2\. Model `state_dict` size

### Dense FT

All model parameters are trainable, so the model stores the original full-rank weights.

### LoRA

LoRA:

\[ W' = W + BA \]

where:

- (W) = original pretrained weight, frozen
- (A) = low-rank matrix
- (B) = low-rank matrix
- (r) = LoRA rank

**Important exam point:**

> LoRA does **not necessarily reduce the complete model****`state_dict`****size**, because the original frozen weights are still stored. It additionally stores the LoRA matrices (A) and (B).

Therefore:

- Dense → original weights
- LoRA → original frozen weights **+** LoRA adapters

So the LoRA state dict can actually be **slightly larger**.

### Key distinction

**Trainable parameter count ≠ total stored model parameters.**

This is a very important concept for the exam.

---

## 3\. Optimizer state size

The optimizer is different.

The code uses:

```
filter(lambda p: p.requires_grad, model.parameters())
```

Therefore the optimizer only receives **trainable parameters**.

For AdamW, optimizer state includes things such as:

- `exp_avg`
- `exp_avg_sq`
- `step`

The frozen parameters don't need optimizer moments.

Therefore:

> **LoRA dramatically reduces optimizer-state storage because only the small A/B adapter matrices and other trainable parameters are optimized.**

### Easy exam explanation

Dense FT:

\[ \\text{Many trainable parameters} \\Rightarrow \\text{large optimizer state} \]

LoRA:

\[ \\text{Few trainable parameters} \\Rightarrow \\text{small optimizer state} \]

---

## 4\. Why the optimizer state is reduced

Suppose a weight matrix has:

\[ W \\in \\mathbb{R}^{d_{out}\\times d_{in}} \]

Dense FT trains all:

\[ d_{out}d_{in} \]

parameters.

LoRA instead trains:

\[ A\\in\\mathbb{R}^{r\\times d\_{in}} \]

and

\[ B\\in\\mathbb{R}^{d\_{out}\\times r} \]

So LoRA trains:

\[ r(d_{in}+d_{out}) \]

parameters.

When:

\[ r \\ll d_{in},d_{out} \]

this is much smaller than:

\[ d_{out}d_{in} \]

Hence the optimizer state is also much smaller.

---

# 5\. Training-time experiment

The code measures the time for **50 training batches**.

For each batch it performs:

1. `zero_grad()`
2. Forward pass
3. Loss calculation
4. Backward pass
5. Optimizer step

CUDA synchronization is used:

```
torch.cuda.synchronize()
```

This is important because GPU operations are normally asynchronous.

### Exam point

> CUDA synchronization makes the timing measurement more reliable because it forces previously queued GPU operations to finish before recording the time.

---

# 6\. Is LoRA necessarily faster?

**Not necessarily.**

This is a subtle but important point.

Although LoRA has fewer trainable parameters, the original frozen network still has to perform its forward computation.

The LoRA forward pass is approximately:

\[ y = Wx + BAx \]

instead of simply:

\[ y = Wx \]

So LoRA introduces additional operations for the adapter:

\[ xA^T \]

followed by:

\[ (xA^T)B^T \]

Therefore LoRA can sometimes be:

- slightly faster,
- approximately the same speed,
- or even slower,

depending on the model, hardware and batch size.

### Important distinction

> **Parameter efficiency does not automatically mean computational speed-up.**

---

# 7\. Rank (r)

The rank controls the capacity of the LoRA adapter.

### Small (r)

Example:

\[ r=2 \]

Advantages:

- Very few trainable parameters
- Very small optimizer state
- Strong parameter efficiency

Disadvantage:

- Lower adaptation capacity

### Large (r)

Example:

\[ r=32 \]

Advantages:

- More adaptation capacity
- Potentially better target-domain accuracy

Disadvantages:

- More trainable parameters
- Larger optimizer state
- More LoRA computation

### Core trade-off

\[ \\boxed{\\text{Higher }r \\Rightarrow \\text{more capacity but less parameter efficiency}} \]

---

# 8\. Rank sweep performed

The experiment tests:

\[ \\boxed{r\\in{2,4,8,16,32}} \]

Everything else should remain fixed:

- Same pretrained model
- Same target dataset
- Same train/validation split
- Same learning rate
- Same number of epochs
- Same L2 regularization
- Same frozen convolutional layers

Therefore, the main changing variable is **LoRA rank**.

---

# 9\. What to look for in the rank results

You need to compare:

| Rank | Trainable parameters | Accuracy |
| --- | --- | --- |
| 2 | Lowest | Depends on experiment |
| 4 | ↑ | Depends on experiment |
| 8 | ↑ | Depends on experiment |
| 16 | ↑ | Depends on experiment |
| 32 | Highest | Depends on experiment |

Don't memorise a particular accuracy unless it is actually printed by your notebook.

The code identifies the best rank with:

```
best_lora_idx = rank_results['best_val_acc'].idxmax()
```

So the **actual best rank is determined experimentally**, not theoretically.

---

# 10\. Dense baseline comparison

The most important comparison is:

\[ \\text{LoRA accuracy} - \\text{Dense accuracy} \]

If positive:

> LoRA outperformed dense fine-tuning.

If zero:

> LoRA matched dense fine-tuning.

If negative:

> LoRA was slightly worse than dense fine-tuning.

But if LoRA achieves similar accuracy with dramatically fewer trainable parameters, it is still highly useful.

---

# 11\. Why LoRA can approach dense FT accuracy

Dense FT learns an unrestricted update:

\[ W' = W + \\Delta W \]

where (\\Delta W) can have full rank.

LoRA restricts the update to:

\[ \\Delta W = BA \]

and therefore:

\[ \\operatorname{rank}(\\Delta W)\\le r \]

So LoRA assumes that the useful adaptation can be represented by a **low-rank update**.

If this assumption is good for the task, LoRA can achieve accuracy close to dense FT while training far fewer parameters.

---

# 12\. The three most important storage concepts

### Complete model storage

LoRA may **not save much**:

\[ \\boxed{ \\text{LoRA model storage} \\approx \\text{original weights}+\\text{adapter weights} } \]

### Trainable parameters

LoRA saves a lot:

\[ \\boxed{ \\text{LoRA trainable parameters} \\ll \\text{Dense trainable parameters} } \]

### Optimizer state

LoRA also saves a lot:

\[ \\boxed{ \\text{LoRA optimizer state} \\ll \\text{Dense optimizer state} } \]

---

# 13\. Final conclusion to remember

The central conclusion of Practical 4 is:

> **LoRA is parameter-efficient rather than necessarily model-storage-efficient or dramatically faster.**

It freezes the original model and learns low-rank adapter matrices.

This gives:

- ✅ Far fewer trainable parameters
- ✅ Much smaller optimizer state
- ✅ Potentially similar target-domain accuracy
- ✅ Easy to vary adaptation capacity using rank (r)
- ❌ Complete `state_dict` may not be smaller
- ❌ Wall-clock training may not improve dramatically
- ❌ Very small (r) can reduce adaptation capacity

---

## 14\. Exam-ready answer: “Why does LoRA reduce optimizer memory?”

**Answer:**

LoRA freezes the pretrained weights and only trains the low-rank adapter matrices (A) and (B). AdamW stores optimizer states such as first- and second-moment estimates for trainable parameters. Since the frozen weights are excluded from the optimizer, their optimizer states are not allocated. Therefore LoRA can dramatically reduce optimizer memory compared with dense fine-tuning.

---

## 15\. Exam-ready answer: “Why can LoRA have a larger state dict?”

**Answer:**

The complete LoRA state dict still contains the original pretrained weights because freezing a parameter does not remove it from the model. LoRA additionally stores the adapter matrices (A) and (B). Therefore the complete LoRA state dict can be slightly larger than the dense model's state dict, even though LoRA has far fewer trainable parameters.

---

## 16\. Exam-ready answer: “Why isn't LoRA necessarily faster?”

**Answer:**

Although LoRA reduces the number of parameters requiring gradient computation and optimizer updates, the frozen pretrained layers still participate in the forward pass. The LoRA layer also adds extra matrix multiplications for (A) and (B). Consequently, the reduction in trainable parameters does not necessarily produce a large wall-clock speed-up.

---

## 17\. Exam-ready answer: “What does the rank (r) control?”

**Answer:**

The LoRA rank (r) controls the capacity of the low-rank weight update. A larger (r) provides a more expressive update and increases the number of trainable parameters, while a smaller (r) gives greater parameter efficiency but may limit adaptation capacity.

---

## 18\. One-minute memorisation sheet

\[ \\boxed{W'=W+BA} \]

- (W): frozen pretrained weight
- (A): (r\\times d\_{in})
- (B): (d\_{out}\\times r)
- (r): LoRA rank

Trainable LoRA parameters:

\[ \\boxed{r(d_{in}+d_{out})} \]

Dense parameters:

\[ \\boxed{d_{in}d_{out}} \]

Therefore:

\[ \\boxed{\\text{LoRA trainable params} \\ll \\text{Dense params}} \]

**LoRA advantages:** fewer trainable parameters + much smaller optimizer state.

**LoRA limitation:** complete model state dict still contains frozen weights.

**Higher****(****r****)****:** more capacity, more parameters.

**Lower****(****r****)****:** less capacity, fewer parameters.

**Speed:** not guaranteed to improve because frozen layers still execute and LoRA adds adapter computation.

**Main idea:** \[ \\boxed{\\text{LoRA = parameter-efficient fine-tuning, not necessarily storage-efficient or faster fine-tuning}} \]
