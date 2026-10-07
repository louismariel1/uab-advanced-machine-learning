# Practical 4 — LoRA: Exam Compression

This chunk is mainly about **transfer learning → dense fine-tuning → parameter-efficient fine-tuning → LoRA**. The most important exam theme is:

> **LoRA reduces the number of trainable parameters by representing the fine-tuning update as a low-rank matrix, while keeping the pretrained weights frozen.**

## 1\. Practical setup: what is being done?

The practical uses:

- **CIFAR-10** as the large **source domain**.
- A 5-class subset of **CIFAR-100** as the small **target domain**.
- A `ConvMLP` model:
- two convolutional layers for feature extraction;
- several large fully connected layers;
- a final classifier.
- The model is first **pre-trained on CIFAR-10**.
- It is then adapted to the target CIFAR-100 subset.

The important conceptual pipeline is:

\[ \\boxed{ \\text{Pre-training} \\rightarrow \\text{Dense fine-tuning baseline} \\rightarrow \\text{LoRA fine-tuning} } \]

The practical is therefore not primarily testing dataset knowledge. It is testing **how LoRA changes the fine-tuning problem**.

---

# 2\. Transfer learning setup

The model first learns a useful representation on the source task:

\[ \\text{CIFAR-10} \\rightarrow \\text{pretrained model} \]

Then the pretrained representation is transferred to a related target task:

\[ \\text{pretrained model} \\rightarrow \\text{CIFAR-100 subset} \]

The target has only five classes, so the final classifier must be replaced/reinitialised.

### Exam point

A pretrained model is useful because its earlier learned representations can be reused rather than learning everything from random initialisation.

---

# 3\. Dense fine-tuning baseline

The practical first establishes a baseline using ordinary fine-tuning.

The early convolutional layers are frozen:

```
conv1: frozen
conv2: frozen
```

while the MLP layers and new classifier remain trainable.

Therefore:

\[ \\boxed{\\text{Dense fine-tuning = directly learning full-sized weight updates}} \]

for the trainable layers.

This is the baseline against which LoRA is compared.

### Important distinction

Freezing layers reduces **training cost**, but does **not automatically reduce inference cost**.

During inference, the frozen layers still perform their forward computations.

---

# 4\. The key problem with dense fine-tuning

Suppose a layer has

\[ W\\in\\mathbb{R}^{d\\times d}. \]

Ordinary fine-tuning learns an update

\[ \\Delta W\\in\\mathbb{R}^{d\\times d}. \]

That requires:

\[ d^2 \]

trainable parameters.

For a large model, this becomes expensive because training requires more than storing the weights themselves.

Trainable parameters also lead to:

- gradients;
- optimiser states;
- additional memory;
- computation during backpropagation.

### Exam question

**Why is full fine-tuning expensive?**

**Answer:** Because every trainable weight can require gradient computation and optimiser state. For a large pretrained model, the number of trainable parameters can therefore dominate training memory and computation.

---

# 5\. The central LoRA idea

LoRA asks:

> Do we really need to learn an arbitrary full-sized update (\\Delta W)?

Instead of learning

\[ \\Delta W\\in\\mathbb{R}^{d\\times d}, \]

LoRA approximates it using two smaller matrices:

\[ \\boxed{\\Delta W\\approx BA} \]

where

\[ A\\in\\mathbb{R}^{r\\times d} \]

and

\[ B\\in\\mathbb{R}^{d\\times r}, \]

with

\[ \\boxed{r\\ll d}. \]

The original pretrained matrix (W) is frozen.

The forward computation becomes:

\[ \\boxed{ Wx+BAx+b } \]

rather than learning a new full matrix.

---

# 6\. The most important LoRA equation

Know this:

\[ \\boxed{ z=\\operatorname{ReLU}(Wx+BAx+b) } \]

where:

- (W) = frozen pretrained weights;
- (A,B) = trainable LoRA parameters;
- (BA) = low-rank approximation to the weight update;
- (b) = bias.

Compare with ordinary fine-tuning:

\[ z=\\operatorname{ReLU}(\\tilde W x+b) \]

where

\[ \\tilde W=W+\\Delta W. \]

LoRA instead uses:

\[ \\tilde W\\approx W+BA. \]

---

# 7\. Why does LoRA save parameters?

Full update:

\[ \\Delta W:d\\times d \]

requires:

\[ d^2 \]

parameters.

LoRA requires:

\[ A:r\\times d \]

and

\[ B:d\\times r. \]

Therefore:

\[ N\_{\\text{LoRA}} = rd+dr = 2rd. \]

The parameter ratio is:

\[ \\frac{2rd}{d^2} = \\boxed{\\frac{2r}{d}}. \]

Since

\[ r\\ll d, \]

we have:

\[ 2rd\\ll d^2. \]

### Exam-grade explanation

> LoRA reduces trainable parameters because it replaces a full (d\\times d) update with two matrices whose inner dimension is a much smaller rank (r). Instead of (d^2) trainable parameters, only (2rd) are required.

---

# 8\. Numerical example

Suppose:

\[ d=10,000,\\qquad r=10. \]

Full fine-tuning:

\[ d^2=100,000,000. \]

LoRA:

\[ 2rd = 2(10)(10,000) = 200,000. \]

Therefore:

\[ \\frac{200,000}{100,000,000} = 0.002 = \\boxed{0.2%}. \]

So only **0.2% of the parameters required for the full update** are trained for this matrix.

Equivalently, the LoRA update uses **500× fewer trainable parameters**.

---

# 9\. AbstractLayer: why is it introduced?

The practical first introduces an `AbstractLayer` before LoRA.

It freezes (W) and introduces a full-sized trainable matrix:

\[ \\Delta W. \]

The forward pass is:

\[ \\boxed{ Wx+\\Delta Wx+b } \]

This is mathematically equivalent to ordinary fine-tuning because:

\[ W+\\Delta W=\\tilde W. \]

The purpose is therefore **conceptual**.

It demonstrates that we can think of fine-tuning as:

> Keep the original model fixed and learn a separate update to its weights.

LoRA then makes this update much smaller:

\[ \\Delta W\\approx BA. \]

### Very useful conceptual progression

\[ \\boxed{ W+\\Delta W \\quad\\rightarrow\\quad W+BA } \]

The first is an equivalent abstraction of dense fine-tuning.

The second is the LoRA approximation.

---

# 10\. Dense fine-tuning vs LoRA

|  | Dense fine-tuning | LoRA |
| --- | --- | --- |
| Original (W) | Updated | Frozen |
| Update | Full (\\Delta W) | Low-rank (BA) |
| Trainable parameters | Large | Small |
| Gradient computation | Many parameters | Few parameters |
| Training memory | Higher | Lower |
| Optimiser state | Large | Much smaller |
| Main assumption | No low-rank restriction | Update has low-rank/intrinsic structure |

### Exam question

**What is the fundamental difference between dense fine-tuning and LoRA?**

**Answer:**

> Dense fine-tuning directly updates the full weight matrix. LoRA freezes the pretrained weight matrix and learns a low-rank approximation to its required update using two smaller trainable matrices.

---

# 11\. LoRA initialisation

A particularly important implementation detail:

\[ B=0 \]

initially, while (A) is randomly initialised.

Therefore:

\[ BA=0. \]

So initially:

\[ Wx+BAx+b = Wx+b. \]

Thus the LoRA model initially behaves exactly like the pretrained model.

### Why is this useful?

It prevents the randomly initialised adapter from immediately disrupting the pretrained model's behaviour.

### Exam question

**Why is****(****B****)****initialised to zero?**

**Answer:**

> Because (BA=0) initially, the LoRA adapter initially makes no change to the pretrained model's activations. The model therefore starts from the pretrained behaviour rather than from a randomly perturbed version of it.

---

# 12\. Why is (A) not also initialised to zero?

The practical uses:

- (A): random/He/Kaiming initialisation;
- (B): zeros.

This gives:

\[ BA=0 \]

at the beginning, while retaining useful stochastic structure in (A) for subsequent training dynamics.

### Exam-level takeaway

You do **not** need to memorise the exact Kaiming formula unless the implementation is examined.

What matters is:

\[ \\boxed{B=0\\Rightarrow BA=0\\text{ initially}} \]

---

# 13\. Biases

The practical freezes the bias as well.

For a linear layer:

\[ Wx+b \]

LoRA modifies the weight update but does not need a low-rank representation for the bias.

The notebook notes that the bias is already a vector rather than a large matrix and is therefore relatively small.

### Exam point

LoRA primarily targets **large weight matrices**, not necessarily every parameter in a layer.

---

# 14\. Rank (r)

The rank (r) is the key LoRA hyperparameter.

\[ A\\in\\mathbb{R}^{r\\times d}, \\qquad B\\in\\mathbb{R}^{d\\times r}. \]

Smaller (r):

- fewer trainable parameters;
- lower training cost;
- stronger compression of the update;
- potentially less adaptation capacity.

Larger (r):

- more trainable parameters;
- greater computational/memory cost;
- potentially greater adaptation capacity.

Therefore:

\[ \\boxed{\\text{smaller }r \\leftrightarrow \\text{efficiency}} \]

\[ \\boxed{\\text{larger }r \\leftrightarrow \\text{adaptation capacity}} \]

This is a classic **trade-off question**.

---

# 15\. LoRA does NOT mean the original model becomes low-rank

This is one of the most important conceptual points.

LoRA does **not** assume:

\[ W\\approx\\text{low-rank matrix}. \]

Instead, it assumes that the **required change to****(****W****)** can be represented approximately in a low-dimensional form:

\[ \\boxed{\\Delta W\\approx BA}. \]

So:

> **The pretrained weights do not have to be low rank. The adaptation/update is what is constrained to be low rank.**

This distinction is highly exam-worthy.

---

# 16\. Why not simply compress (W)?

If you directly low-rank-factorise the pretrained weights, you are changing the original model:

\[ W\\rightarrow W\_{\\text{low-rank}}. \]

This can introduce approximation error into the pretrained representation.

LoRA instead keeps:

\[ W \]

intact and learns:

\[ BA \]

as an additional adaptation.

Therefore:

\[ \\boxed{ \\text{LoRA preserves the pretrained weights and compresses the update} } \]

rather than compressing the original model itself.

---

# 17\. LoRA scaling

The original LoRA formulation can scale the update by approximately:

\[ \\frac{1}{r}. \]

Conceptually, this helps keep the effective update scale more consistent when changing (r).

For this practical, the notebook says the effect was not particularly important experimentally, but theoretically it is an important detail.

### Exam answer

> LoRA can scale the low-rank update according to the rank so that changing (r) does not arbitrarily change the effective update magnitude.

---

# 18\. What LoRA actually saves

Be precise.

LoRA primarily reduces:

\[ \\boxed{\\text{number of trainable parameters}} \]

which reduces:

- gradient storage;
- optimiser-state storage;
- backpropagation computation;
- task-specific parameters that must be trained/stored.

It does **not necessarily reduce the size of the frozen pretrained model itself**.

This distinction is important.

### Compare

**Quantisation:**

\[ \\text{represent existing weights more compactly} \]

**LoRA:**

\[ \\text{avoid training/storing a full update} \]

---

# 19\. Does LoRA automatically make inference faster?

Not necessarily.

During LoRA inference:

\[ Wx+BAx \]

contains an additional adapter computation.

Therefore, if implemented literally, LoRA can introduce some inference overhead.

However, because:

\[ W+BA \]

can be formed into a single effective weight matrix after training, the adapter can often be **merged** with the original weights.

Then inference can use:

\[ \\boxed{(W+BA)x} \]

with essentially the same architecture as the original layer.

### Exam question

**Why can LoRA have little or no inference overhead after training?**

**Answer:**

> The learned low-rank update (BA) can be merged into the pretrained weight matrix, producing (W+BA). The resulting layer can then perform the normal dense matrix multiplication without a separate LoRA branch.

---

# 20\. The most important practical comparison

### Dense fine-tuning

\[ W\\rightarrow W+\\Delta W \]

Train:

\[ d^2 \]

parameters.

### LoRA

\[ W\\text{ frozen} \]

and learn:

\[ \\Delta W\\approx BA. \]

Train:

\[ 2rd \]

parameters.

Thus:

\[ \\boxed{ d^2 \\quad\\rightarrow\\quad 2rd } \]

with (r\\ll d).

---

# 21\. High-probability exam questions

### Q1. What problem is LoRA designed to solve?

**Answer:** LoRA is designed to make fine-tuning large pretrained models more parameter- and computation-efficient by freezing the pretrained weights and learning only a small low-rank update.

---

### Q2. A layer has a (d\\times d) weight matrix. How many trainable parameters are needed for full fine-tuning versus LoRA?

**Answer:**

Full fine-tuning:

\[ d^2. \]

LoRA:

\[ 2rd. \]

The ratio is:

\[ \\frac{2r}{d}. \]

---

### Q3. Why does LoRA reduce training memory?

**Answer:** Only (A) and (B) are trainable. Therefore, gradients and optimiser states are required for far fewer parameters than in full fine-tuning.

---

### Q4. Why does LoRA freeze (W)?

**Answer:** The pretrained knowledge is retained in (W), while task-specific adaptation is represented separately by the low-rank update (BA).

---

### Q5. What happens initially when (B=0)?

**Answer:**

\[ BA=0 \]

so:

\[ Wx+BAx+b=Wx+b. \]

The LoRA model initially behaves exactly like the pretrained model.

---

### Q6. What does increasing (r) do?

**Answer:** It increases the number of trainable parameters and the expressive capacity of the adaptation, but also increases training memory and computation.

---

### Q7. Is LoRA equivalent to saying that (W) itself has low rank?

**Answer:** No. LoRA assumes that the **adaptation/update** can be represented in a low-rank form:

\[ \\Delta W\\approx BA. \]

The original (W) can remain full rank.

---

### Q8. Why is the `AbstractLayer` useful?

**Answer:** It demonstrates that ordinary fine-tuning can be viewed as freezing (W) and learning a separate full-sized update (\\Delta W). LoRA then replaces that expensive full update with a low-rank approximation (BA).

---

# 22\. Exam-grade scenario questions

### Scenario 1

> You need to fine-tune a huge pretrained model for 50 different tasks, but GPU memory is limited. Would you use full fine-tuning or LoRA?

**Reasoning:**

- Full fine-tuning creates a separate large update for each task.
- LoRA freezes the shared base model.
- Each task only needs its small (A,B) matrices.
- Therefore LoRA is much more memory-efficient for multiple task adaptations.

\[ \\boxed{\\text{Choose LoRA}} \]

---

### Scenario 2

> Increasing LoRA rank from (r=4) to (r=64) improves accuracy but substantially increases training memory. Explain.

Because:

\[ N\_{\\text{LoRA}}=2rd. \]

Increasing (r) increases trainable parameters linearly.

Therefore:

\[ r\\uparrow \\Rightarrow \\text{capacity}\\uparrow \]

but also:

\[ r\\uparrow \\Rightarrow \\text{memory/computation}\\uparrow. \]

---

### Scenario 3

> A student says: "LoRA compresses the pretrained model because (W) is replaced by a low-rank matrix." Is this correct?

**No.**

LoRA keeps (W) frozen and adds:

\[ BA. \]

It compresses the **learned adaptation**, not necessarily the pretrained weight matrix.

---

### Scenario 4

> A LoRA model initially produces different predictions from the pretrained model. What implementation detail might be wrong?

Check the initialisation.

The intended behaviour is:

\[ B=0 \]

so that:

\[ BA=0. \]

If both matrices are randomly initialised, the adapter changes the model immediately.

---

# 23\. What the code is trying to teach you

You do **not** need to memorise every line of the notebook.

Understand this transformation:

```
Original layer

       W
       ↓
      Wx
```

becomes:

```
LoRA layer

       frozen W ─────→ Wx
                         +
       A ─→ B ─────────→ BAx
                         ↓
                    Wx + BAx
```

Mathematically:

\[ \\boxed{ Wx \\rightarrow Wx+BAx } \]

with:

\[ \\boxed{ W\\text{ frozen},\\qquad A,B\\text{ trainable}. } \]

---

# 24\. The one-minute revision version

If you remember only one section, remember this:

> **LoRA is a parameter-efficient fine-tuning method.** Instead of updating a pretrained weight matrix (W) directly, LoRA freezes (W) and learns a low-rank update: \[ \\Delta W\\approx BA \] where \[ A\\in\\mathbb{R}^{r\\times d},\\qquad B\\in\\mathbb{R}^{d\\times r},\\qquad r\\ll d. \] Full fine-tuning requires (d^2) trainable parameters, whereas LoRA requires only (2rd). This dramatically reduces gradients, optimiser states, training memory and computation. The pretrained model is preserved, while (BA) learns the task-specific adaptation. (B) is typically initialised to zero so that (BA=0) initially and the LoRA model starts with the same behaviour as the pretrained model. A larger (r) provides more adaptation capacity but costs more memory and computation. LoRA can often be merged into (W+BA) for inference, avoiding separate adapter computation.

## 25\. Core exam chain

The entire practical can be reduced to this reasoning chain:

\[ \\boxed{ \\text{Full fine-tuning} \\rightarrow \\text{too many trainable parameters} } \]

\[ \\boxed{ \\text{Freeze }W \\rightarrow \\text{learn only the update} } \]

\[ \\boxed{ \\Delta W \\rightarrow BA } \]

\[ \\boxed{ d^2 \\rightarrow 2rd } \]

\[ \\boxed{ r\\ll d \\rightarrow \\text{much cheaper fine-tuning} } \]

And the key conceptual insight is:

\[ \\boxed{ \\text{We are assuming the required adaptation is low-dimensional, not that the pretrained model itself is low-rank.} } \]

This is the **highest-priority material from Chunk 1** for exam preparation.
