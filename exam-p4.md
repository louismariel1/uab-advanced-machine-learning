# Practical 4 — LoRA: Complete Exam Compression

This is the **merged exam-focused version of Chunks 1–3**. It combines the theory, implementation, rank sweep, state/optimizer analysis, training efficiency, and final experimental conclusions.

---

## 1\. Practical objective

The practical investigates **parameter-efficient fine-tuning** using **LoRA (Low-Rank Adaptation)** and compares it with dense fine-tuning.

The overall pipeline is:

\[ \\boxed{ \\text{CIFAR-10 pre-training} \\rightarrow \\text{Dense fine-tuning baseline} \\rightarrow \\text{LoRA fine-tuning} \\rightarrow \\text{Rank/state/efficiency analysis} } \]

The main question is:

> **Can we adapt a pretrained model to a new task while training dramatically fewer parameters, without losing too much accuracy?**

---

# 2\. Transfer learning setup

The model is first pretrained on:

- **Source domain:** CIFAR-10

It is then adapted to:

- **Target domain:** a 5-class subset of CIFAR-100.

The model is a `ConvMLP` containing:

- convolutional feature-extraction layers;
- large fully connected/MLP layers;
- a final classifier.

The target classifier must be adapted because the target task has a different number of classes.

### Core idea

Instead of learning the target task entirely from scratch:

\[ \\text{pretrained representation} \\rightarrow \\text{target-domain adaptation} \]

This is **transfer learning**.

---

# 3\. Dense fine-tuning baseline

Dense fine-tuning provides the reference point for LoRA.

In this practical, some early convolutional layers are frozen, while the MLP layers and classifier remain trainable.

Conceptually:

\[ W\\rightarrow W+\\Delta W \]

where the trainable layers directly learn a full-sized update.

### Important distinction

Freezing some layers is **not the same thing as LoRA**.

Dense fine-tuning still directly trains the complete weight matrices of the layers selected for training.

### Exam answer

> Dense fine-tuning directly updates the full-sized trainable weight matrices, whereas LoRA freezes pretrained weights and learns only a low-rank adaptation.

---

# 4\. Why is full/dense fine-tuning expensive?

Suppose a weight matrix is:

\[ W\\in\\mathbb{R}^{d_{out}\\times d_{in}} \]

The number of weights is:

\[ d_{out}d_{in} \]

For a square matrix:

\[ d^2 \]

Training these parameters also requires:

- gradients;
- optimizer state;
- backpropagation computation;
- additional GPU memory.

For AdamW, optimizer state can be especially significant because it maintains quantities such as:

- `exp_avg`;
- `exp_avg_sq`;
- step information.

### Exam answer

> Dense fine-tuning is expensive because many parameters require gradients and optimizer states. As model size increases, training memory and computation can become very large.

---

# 5\. The central LoRA idea

LoRA asks:

> Do we really need to learn an arbitrary full-sized update (\\Delta W)?

Instead:

\[ \\boxed{\\Delta W\\approx BA} \]

where:

\[ A\\in\\mathbb{R}^{r\\times d\_{in}} \]

and:

\[ B\\in\\mathbb{R}^{d\_{out}\\times r} \]

with:

\[ \\boxed{r\\ll d} \]

The original pretrained weight matrix (W) is frozen.

Therefore:

\[ \\boxed{W'=W+BA} \]

This is the single most important equation in the practical.

---

# 6\. LoRA forward pass

The original layer performs approximately:

\[ y=Wx+b \]

LoRA adds the low-rank adaptation:

\[ \\boxed{y=Wx+BAx+b} \]

Depending on whether the implementation uses row or column vectors, the code may appear as:

\[ xW^T+xA^TB^T \]

These are equivalent descriptions under different tensor conventions.

### Implementation concept

```
Input x
   │
   ├──────────────→ Frozen original layer → Wx
   │
   └→ A: d_in → r → B: r → d_out → BAx
                                      │
                                      ↓
                                  Wx + BAx
```

### Exam explanation

1. Pass (x) through the frozen original layer.
2. Project (x) into a low-dimensional (r)-dimensional space using (A).
3. Project it back using (B).
4. Add the resulting update to the original layer output.

---

# 7\. LoRA parameter count

For:

\[ W\\in\\mathbb{R}^{d_{out}\\times d_{in}} \]

dense fine-tuning requires:

\[ \\boxed{d_{out}d_{in}} \]

trainable weight parameters.

LoRA requires:

\[ A:r\\times d\_{in} \]

and:

\[ B:d\_{out}\\times r \]

Therefore:

\[ \\boxed{ N_{LoRA}=r(d_{in}+d\_{out}) } \]

For a square (d\\times d) matrix:

\[ \\boxed{N\_{LoRA}=2rd} \]

while dense training requires:

\[ d^2 \]

Therefore:

\[ \\frac{N_{LoRA}}{N_{dense}} = \\frac{2rd}{d^2} = \\boxed{\\frac{2r}{d}} \]

Since (r\\ll d):

\[ 2rd\\ll d^2 \]

---

# 8\. Numerical example

Suppose:

\[ d=10,000,\\qquad r=10 \]

Dense update:

\[ d^2=100,000,000 \]

LoRA:

\[ 2rd=2(10)(10,000)=200,000 \]

Ratio:

\[ \\frac{200,000}{100,000,000}=0.002 \]

Therefore only:

\[ \\boxed{0.2%} \]

of the full update parameters are required.

That is a:

\[ \\boxed{500\\times} \]

reduction for this particular matrix.

---

# 9\. What exactly is frozen and trained?

### Frozen

- Original pretrained weight matrices (W)
- Frozen convolutional layers
- Other pretrained parameters not explicitly made trainable

### Trainable

- LoRA (A) matrices
- LoRA (B) matrices
- New/adapted classifier parameters

Conceptually:

\[ \\boxed{W:\\text{ frozen}} \]

\[ \\boxed{A,B:\\text{ trainable}} \]

The practical checks this using things such as:

```
requires_grad == False
```

for the original layer weights.

---

# 10\. AbstractLayer: important conceptual step

The practical introduces an `AbstractLayer`.

It freezes:

\[ W \]

and introduces a full-sized trainable matrix:

\[ \\Delta W \]

The layer becomes:

\[ \\boxed{Wx+\\Delta Wx+b} \]

But:

\[ W+\\Delta W=\\tilde W \]

so this is mathematically equivalent to ordinary dense fine-tuning.

This establishes the conceptual progression:

\[ \\boxed{ W+\\Delta W \\quad\\rightarrow\\quad W+BA } \]

The first represents a **full-sized update**.

The second represents the **LoRA low-rank approximation**.

---

# 11\. Why initialise (B=0)?

LoRA commonly uses:

- (A): random initialization;
- (B): zero initialization.

Initially:

\[ B=0 \]

therefore:

\[ BA=0 \]

and:

\[ Wx+BAx+b=Wx+b \]

So the LoRA model initially behaves exactly like the pretrained model.

### Exam answer

> (B) is initialized to zero so that the LoRA update is initially zero. This prevents the randomly initialized adapter from immediately perturbing the pretrained model.

---

# 12\. Why isn't (A) also zero?

If both (A) and (B) were zero, the adapter would initially contain no useful random structure.

Instead:

\[ A=\\text{random} \]

and:

\[ B=0 \]

gives:

\[ BA=0 \]

initially while allowing useful learning dynamics once (B) starts changing.

### Highest-priority fact

\[ \\boxed{B=0\\Rightarrow BA=0\\text{ initially}} \]

---

# 13\. LoRA does NOT make the pretrained model low-rank

This is a very common exam trap.

LoRA does **not** assume:

\[ W\\approx W\_{low-rank} \]

Instead:

\[ \\boxed{\\Delta W\\approx BA} \]

The pretrained (W) can remain full rank.

### Correct statement

> LoRA assumes the **required adaptation/update** has a low-dimensional structure, not that the pretrained model itself is low rank.

---

# 14\. Why not directly compress (W)?

If we replace:

\[ W \]

with a low-rank approximation:

\[ W\_{low-rank} \]

we may introduce approximation error into the pretrained representation.

LoRA instead preserves:

\[ W \]

and learns:

\[ BA \]

as an additional adaptation.

Therefore:

\[ \\boxed{\\text{LoRA compresses the update, not necessarily the pretrained model}} \]

---

# 15\. Rank (r)

The LoRA rank determines the capacity of the adapter.

The practical tests:

\[ \\boxed{r=2,4,8,16,32} \]

### Small (r)

- fewer trainable parameters;
- lower memory usage;
- lower optimizer state;
- lower adapter computation;
- less expressive;
- possible underfitting.

### Large (r)

- more trainable parameters;
- more expressive adaptation;
- potentially better accuracy;
- more memory;
- more computation.

The central trade-off is:

\[ \\boxed{ r\\uparrow \\Rightarrow \\text{capacity}\\uparrow,\\quad \\text{cost}\\uparrow } \]

---

# 16\. Rank sweep experiment

Each rank is trained using a **fresh model**.

This is important.

For each:

\[ r\\in{2,4,8,16,32} \]

the experiment should use:

- same pretrained weights;
- same dataset;
- same train/validation split;
- same epochs;
- same learning rate;
- same regularization;
- only the rank changes.

### Why?

Otherwise one rank could inherit training progress from another, making the comparison unfair.

### Same random seed

The seed improves:

- reproducibility;
- experimental control;
- comparability between configurations.

---

# 17\. Choosing the best rank

The best rank is selected according to validation accuracy:

\[ \\boxed{ r^\*=\\arg\\max\_r \\text{Validation Accuracy}(r) } \]

The practical finds the best configuration from the rank sweep.

In the provided final analysis:

\[ \\boxed{r=32} \]

is the selected/best LoRA model.

---

# 18\. Accuracy vs parameter efficiency

Accuracy alone is not enough.

The purpose of LoRA is parameter-efficient adaptation.

Two important plots are:

### Accuracy vs rank

\[ r\\rightarrow\\text{validation accuracy} \]

Shows how adaptation capacity affects performance.

### Accuracy vs trainable parameters

\[ \\text{trainable parameters} \\rightarrow \\text{validation accuracy} \]

This is particularly important because we want:

\[ \\boxed{\\text{high accuracy with few trainable parameters}} \]

---

# 19\. Trainable fraction

The practical can calculate:

\[ \\boxed{ \\text{trainable fraction} = \\frac{\\text{trainable parameters}} {\\text{total parameters}} } \]

For example:

\[ \\frac{10,000}{100,000}=0.1 \]

means:

\[ \\boxed{10%} \]

of the model is trainable.

---

# 20\. What does LoRA actually save?

LoRA primarily reduces:

\[ \\boxed{\\text{trainable parameters}} \]

This leads to reductions in:

- gradient storage;
- optimizer state;
- backpropagation work;
- task-specific trainable parameters;
- training memory.

But be precise:

> LoRA does not necessarily reduce the storage required for the complete pretrained model.

---

# 21\. Model state\_dict size

A model's `state_dict` can contain the original pretrained weights.

Therefore, even though LoRA has very few **trainable** parameters, its complete state dict can still contain:

\[ \\boxed{W+A+B} \]

rather than only:

\[ A+B \]

So LoRA does **not automatically mean a smaller complete model state dict**.

In the practical:

\[ \\boxed{\\text{LoRA }r=32\\text{ state dict can be slightly larger than dense FT}} \]

because the pretrained weights remain stored and LoRA adds adapter matrices.

### Very important distinction

**Trainable parameter efficiency ≠ complete model storage compression.**

---

# 22\. Saving only LoRA adapters

If the pretrained model is already available, we can save only:

\[ \\boxed{A,B} \]

for the adapted layers.

Then the conceptual checkpoint becomes:

\[ \\boxed{ \\text{shared pretrained model} + \\text{small task-specific adapters} } \]

This is extremely useful when many tasks share the same base model.

---

# 23\. Optimizer state

The optimizer only needs state for parameters being optimized.

Dense fine-tuning:

\[ \\boxed{\\text{many trainable parameters}} \]

therefore:

\[ \\boxed{\\text{large optimizer state}} \]

LoRA:

\[ \\boxed{\\text{only }A,B\\text{ and classifier are trainable}} \]

therefore:

\[ \\boxed{\\text{much smaller optimizer state}} \]

For AdamW, optimizer state includes tensors such as:

- `exp_avg`;
- `exp_avg_sq`;
- step-related state.

The practical explicitly measures the tensor values stored in the optimizer.

---

# 24\. Why optimizer-state savings are important

This is one of the strongest practical advantages of LoRA.

Frozen parameters do not require optimizer moments.

Therefore:

\[ \\boxed{ \\text{fewer trainable parameters} \\Rightarrow \\text{fewer optimizer states} \\Rightarrow \\text{lower training memory} } \]

### Exam-safe distinction

> The complete LoRA model state dict may still contain the frozen pretrained weights, but its optimizer state is much smaller because the optimizer only tracks trainable parameters.

---

# 25\. Training efficiency

The practical measures training time for:

- Dense fine-tuning
- LoRA (r=32)

using a fixed number of batches.

It measures:

\[ \\boxed{\\text{average training time per batch}} \]

and compares:

\[ \\text{Dense batch time} \]

against:

\[ \\text{LoRA batch time} \]

---

# 26\. Is LoRA always faster?

\[ \\boxed{\\text{No}} \]

This is a very important exam point.

LoRA reduces the amount of trainable parameters, but its forward pass adds:

\[ xA^T \]

followed by:

\[ (xA^T)B^T \]

in addition to the original:

\[ xW^T \]

Therefore LoRA can introduce extra forward computation.

### Safe exam answer

> LoRA reduces gradient and optimizer computation because most weights are frozen, but it adds low-rank adapter operations to the forward pass. Therefore it is not guaranteed to be faster in wall-clock time.

Actual timing depends on:

- hardware;
- batch size;
- layer dimensions;
- implementation;
- LoRA rank.

---

# 27\. Why does larger (r) increase computation?

LoRA computes:

\[ xA^T \]

where the intermediate representation has dimension:

\[ r \]

and then:

\[ (xA^T)B^T \]

Therefore:

\[ \\boxed{ r\\uparrow \\Rightarrow \\text{parameters}\\uparrow \\Rightarrow \\text{adapter computation}\\uparrow } \]

---

# 28\. GPU memory measurements

The practical may use:

```
torch.cuda.memory_allocated(0)
torch.cuda.memory_reserved(0)
torch.cuda.max_memory_reserved(0)
```

### Meanings

- `memory_allocated` → memory currently used by live tensors;
- `memory_reserved` → memory reserved by PyTorch;
- `max_memory_reserved` → peak reserved memory.

### `empty_cache()`

```
torch.cuda.empty_cache()
```

releases unused cached memory.

It does **not** delete live models or tensors.

For fair memory comparisons, the practical recommends starting from a clean/restarted environment.

---

# 29\. LoRA scaling

LoRA can scale the update, commonly using a factor related to:

\[ \\frac{\\alpha}{r} \]

where (\\alpha) is a scaling hyperparameter.

Conceptually, scaling helps control the effective magnitude of the LoRA update when the rank changes.

### Exam takeaway

> LoRA scaling controls the magnitude of the low-rank update and helps make changes in rank less likely to arbitrarily change the update scale.

---

# 30\. Biases

For:

\[ Wx+b \]

LoRA primarily modifies the weight update.

The bias is a vector rather than a large matrix, so it does not offer the same parameter-saving opportunity.

In this practical, the bias can be frozen depending on the implementation.

### Exam point

> LoRA mainly targets large weight matrices where low-rank factorisation provides substantial parameter savings.

---

# 31\. Dense FT vs LoRA — master comparison

| Property | Dense FT | LoRA |
| --- | --- | --- |
| Pretrained (W) | Updated | Frozen |
| Adaptation | Full (\\Delta W) | Low-rank (BA) |
| Trainable parameters | Large | Small |
| Gradients | Many | Few |
| Optimizer state | Large | Much smaller |
| Training memory | Higher | Lower |
| Forward computation | Standard | Extra adapter computation |
| Guaranteed faster? | — | No |
| Complete state dict | Large | Can still be large |
| Adapter-only checkpoint | No | Very small |
| Main advantage | Maximum unrestricted adaptation | Parameter efficiency |
| Main trade-off | Expensive | Rank limits adaptation capacity |

---

# 32\. State dict vs optimizer state — VERY IMPORTANT

Do not confuse these.

### Complete model state dict

Can contain:

\[ \\boxed{W+A+B} \]

Therefore LoRA may **not** be much smaller.

### Optimizer state

Only tracks trainable parameters:

\[ \\boxed{A+B+\\text{trainable classifier}} \]

Therefore LoRA is substantially smaller.

### Memorise:

\[ \\boxed{ \\text{LoRA saves optimizer state much more directly than complete state-dict size} } \]

---

# 33\. LoRA merging

After training:

\[ W'=W+BA \]

The adapter can be merged into the original weight.

Therefore:

\[ \\boxed{W\_{merged}=W+BA} \]

Inference can then use:

\[ \\boxed{y=W\_{merged}x} \]

rather than separately computing:

\[ Wx+BAx \]

### Benefit

After merging:

- no separate LoRA branch;
- no extra adapter computation;
- ordinary layer architecture;
- same mathematical output, assuming correct merging.

---

# 34\. Multiple-task scenario

Suppose one pretrained model must be adapted to 50 different tasks.

Dense fine-tuning would require a large task-specific set of updated parameters for every task.

LoRA can instead keep one shared:

\[ W \]

and store small task-specific:

\[ A_i,B_i \]

for each task.

Conceptually:

\[ \\boxed{ \\text{One base model} + \\text{many small adapters} } \]

This is a major practical advantage of LoRA.

---

# 35\. High-probability exam questions

### Q1. What problem does LoRA solve?

> LoRA makes fine-tuning more parameter- and memory-efficient by freezing pretrained weights and learning only a low-rank adaptation.

### Q2. What is the key LoRA equation?

\[ \\boxed{W'=W+BA} \]

### Q3. How many parameters does dense fine-tuning require?

For (d_{out}\\times d_{in}):

\[ \\boxed{d_{out}d_{in}} \]

### Q4. How many parameters does LoRA require?

\[ \\boxed{r(d_{in}+d_{out})} \]

### Q5. What does (r) represent?

> The rank of the LoRA adaptation and therefore its expressive capacity.

### Q6. What happens when (r) increases?

> Trainable parameters, memory and computation increase, while adaptation capacity generally increases.

### Q7. Are the original weights trained?

> No. They are frozen.

### Q8. Why is (B) initialized to zero?

> So (BA=0) initially and the LoRA model starts with the pretrained model's behaviour.

### Q9. Does LoRA mean (W) is low rank?

> No. The **update** (\\Delta W) is approximated as low rank; (W) itself can remain full rank.

### Q10. Why is optimizer state smaller?

> AdamW only maintains optimizer state for trainable parameters, and LoRA has far fewer trainable parameters.

### Q11. Does LoRA always produce a smaller complete state dict?

> No. The frozen pretrained weights may still be stored, and the LoRA matrices are added.

### Q12. Is LoRA always faster?

> No. It reduces backward/optimization work but introduces additional low-rank forward operations.

### Q13. How can LoRA be merged?

\[ \\boxed{W\_{merged}=W+BA} \]

### Q14. Why test multiple ranks?

> To find the best trade-off between validation accuracy, adaptation capacity, parameter count, memory and computation.

---

# 36\. Scenario questions

## Scenario 1 — Limited GPU memory

> You must adapt a huge pretrained model but have limited GPU memory. What should you use?

\[ \\boxed{\\text{LoRA}} \]

Reason:

\[ \\text{fewer trainable parameters} \\rightarrow \\text{fewer gradients} \\rightarrow \\text{smaller optimizer state} \\rightarrow \\text{lower training memory} \]

---

## Scenario 2 — Increasing rank improves accuracy

> (r=4) gives lower accuracy than (r=32), but (r=32) uses more memory. Explain.

Because:

\[ N_{LoRA}=r(d_{in}+d\_{out}) \]

Therefore:

\[ r\\uparrow \\Rightarrow \\text{more trainable parameters} \]

and more capacity can improve adaptation.

But:

\[ r\\uparrow \\Rightarrow \\text{memory/computation}\\uparrow \]

---

## Scenario 3 — Student says LoRA compresses (W)

Incorrect.

LoRA does:

\[ W+BA \]

not:

\[ W\\rightarrow W\_{low-rank} \]

The low-rank assumption applies to:

\[ \\boxed{\\Delta W} \]

not (W).

---

## Scenario 4 — LoRA changes predictions immediately

Check initialization.

Expected:

\[ B=0 \]

so:

\[ BA=0 \]

and:

\[ y_{LoRA}=y_{pretrained} \]

initially.

---

## Scenario 5 — LoRA state dict is not smaller

This is not necessarily a bug.

The state dict may contain:

\[ W+A+B \]

so the frozen base model still consumes storage.

The major storage saving is associated with:

- trainable parameters;
- optimizer state;
- adapter-only checkpoints.

---

# 37\. Final experiment results — what to say

The practical's final analysis compares the best LoRA model, (r=32), with dense fine-tuning.

The final report considers:

1. validation accuracy;
2. trainable parameters;
3. complete model state dict;
4. optimizer state;
5. training time.

The key interpretation is:

### Accuracy

The best tested LoRA configuration is:

\[ \\boxed{r=32} \]

Its accuracy is compared directly against the dense baseline.

### Trainable parameters

LoRA uses dramatically fewer trainable parameters:

\[ \\boxed{ \\text{Dense trainable parameters} \\gg \\text{LoRA trainable parameters} } \]

### Model storage

The complete LoRA state dict is **not necessarily smaller** because the frozen pretrained weights remain stored.

### Optimizer state

The LoRA optimizer state is substantially smaller because only LoRA and other trainable parameters are optimized.

### Training time

The measured wall-clock improvement may be modest because the frozen pretrained network still participates in the forward pass and LoRA adds adapter operations.

---

# 38\. The most important conceptual distinction

Memorise this table:

| Question | LoRA effect |
| --- | --- |
| Trainable parameters? | **Much smaller** |
| Gradient storage? | **Much smaller** |
| Optimizer state? | **Much smaller** |
| Complete model state dict? | **Not necessarily smaller** |
| Forward computation? | **Adds adapter operations** |
| Wall-clock training? | **Not guaranteed faster** |
| Inference after merge? | **Can have essentially no extra adapter overhead** |
| Pretrained weights? | **Frozen and preserved** |
| Adaptation? | **Low-rank****(****BA****)** |

---

# 39\. One-minute memorisation sheet

If you remember only this, remember:

\[ \\boxed{W'=W+BA} \]

- (W) = pretrained weight, **frozen**
- (A,B) = trainable LoRA matrices
- (r) = LoRA rank
- (r\\ll d)
- Dense parameters:

\[ \\boxed{d_{out}d_{in}} \]

- LoRA parameters:

\[ \\boxed{r(d_{in}+d_{out})} \]

- Square matrix:

\[ \\boxed{2rd\\text{ instead of }d^2} \]

- (B=0) initially:

\[ \\boxed{BA=0} \]

- Therefore the initial LoRA model behaves like the pretrained model.
- Higher (r) → more capacity, parameters and computation.
- Lower (r) → more efficient but potentially less expressive.
- LoRA reduces **trainable parameters, gradients and optimizer state**.
- It does **not necessarily reduce complete model state-dict size**.
- LoRA is **not guaranteed to be faster**.
- LoRA assumes the **adaptation** is low rank, not that (W) is low rank.
- After training:

\[ \\boxed{W\_{merged}=W+BA} \]

- Main goal:

\[ \\boxed{\\text{high target-task accuracy with very few trainable parameters}} \]

---

# 40\. Ultimate exam chain

The entire Practical 4 can be reduced to:

\[ \\boxed{ \\text{Pretrained model} \\rightarrow \\text{transfer learning} } \]

\[ \\boxed{ \\text{Dense FT} \\rightarrow \\text{large full weight updates} } \]

\[ \\boxed{ \\text{Freeze }W \\rightarrow \\text{learn only adaptation} } \]

\[ \\boxed{ \\Delta W \\rightarrow BA } \]

\[ \\boxed{ d_{out}d_{in} \\rightarrow r(d_{in}+d_{out}) } \]

\[ \\boxed{ r\\ll d \\rightarrow \\text{far fewer trainable parameters} } \]

\[ \\boxed{ \\text{fewer trainable parameters} \\rightarrow \\text{smaller gradients + optimizer state} } \]

but:

\[ \\boxed{ \\text{frozen }W\\text{ is still stored} \\rightarrow \\text{complete state dict may not shrink} } \]

and:

\[ \\boxed{ \\text{LoRA forward adds }BAx \\rightarrow \\text{not necessarily faster} } \]

finally:

\[ \\boxed{ W+BA \\rightarrow \\text{merge after training} } \]

### The single most important sentence to memorise

> **LoRA freezes the pretrained full-rank weights and learns a low-rank approximation of the required task-specific update, reducing trainable parameters and optimizer memory while preserving the pretrained model.**
