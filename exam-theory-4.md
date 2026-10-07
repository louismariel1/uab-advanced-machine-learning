Absolutely. I’ll treat this as **Lecture 4 — Exam Compression** and keep it consistent with the compression style we used for Lectures 1–3.

# Lecture 4 — Exam Compression: Model Compression & Parameter-Efficient Fine-Tuning

## 1\. The central problem

Modern neural networks are **overparameterised** and expensive in terms of:

- memory/storage
- computation
- GPU requirements
- energy
- training cost
- inference latency

The lecture asks:

> **Do we really need all the parameters, all the numerical precision, and all the trainable weights?**

The answer is often **no**.

The overall strategy is:

\[ \\boxed{\\text{Exploit redundancy}} \]

There are two major goals:

| Goal | Technique |
| --- | --- |
| Make an existing model smaller/cheaper | **Model compression** |
| Make adapting a large pretrained model cheaper | **PEFT** |

---

# 2\. Model compression vs PEFT

### Model compression

Goal:

\[ \\boxed{\\text{same model, cheaper}} \]

Examples:

- Quantisation
- Pruning
- Low-rank factorisation
- Separable convolutions

### Parameter-efficient fine-tuning (PEFT)

Goal:

\[ \\boxed{\\text{new task, cheaper adaptation}} \]

Examples:

- Layer freezing
- Adapters
- Prefix tuning
- LoRA

### Important distinction

> **Compression reduces the cost of the model itself; PEFT reduces the cost of adapting a pretrained model.**

---

# 3\. Quantisation

## Core idea

Neural-network weights often use more numerical precision than necessary.

Instead of:

\[ \\text{FP32}=32\\text{ bits} \]

we can use:

\[ \\text{FP16}=16\\text{ bits} \]

\[ \\text{INT8}=8\\text{ bits} \]

\[ \\text{INT4}=4\\text{ bits} \]

Therefore:

\[ \\boxed{\\text{many precise values}\\rightarrow\\text{fewer approximate values}} \]

### Memory reduction

For (10^9) parameters:

FP32:

\[ 10^9\\times32=32\\times10^9\\text{ bits} \]

INT8:

\[ 10^9\\times8=8\\times10^9\\text{ bits} \]

Therefore:

\[ \\boxed{\\text{FP32}\\rightarrow\\text{INT8}\\approx4\\times\\text{less weight storage}} \]

### Main trade-off

\[ \\boxed{ \\text{lower precision} \\leftrightarrow \\text{lower memory/computation} } \]

But quantisation introduces **quantisation error**.

---

# 4\. PTQ vs QAT

This is a **high-priority exam comparison**.

## Post-training quantisation — PTQ

Train first, quantise afterward:

\[ \\text{FP model} \\rightarrow \\text{trained model} \\rightarrow \\text{quantise} \]

Advantages:

- simple
- cheap
- no full retraining

Disadvantage:

- model wasn't trained to compensate for quantisation error
- potentially greater accuracy degradation

---

## Quantisation-aware training — QAT

Quantisation is simulated during training:

\[ \\text{weights} \\rightarrow \\text{quantise} \\rightarrow \\text{dequantise} \\rightarrow \\text{forward pass} \]

The model learns to compensate for quantisation noise.

Therefore:

\[ \\boxed{\\text{QAT usually gives better accuracy but costs more training}} \]

| PTQ | QAT |
| --- | --- |
| Cheap | More expensive |
| Simple | More complex |
| After training | During training |
| Potentially larger accuracy loss | Usually better accuracy |

### Exam answer

> **PTQ is cheaper and simpler because quantisation is applied after training, while QAT incorporates simulated quantisation during training so the model can adapt to quantisation errors.**

---

# 5\. Pruning

Quantisation asks:

> **Can we represent the weights with fewer bits?**

Pruning asks:

> **Do we need all the weights at all?**

Example:

\[ W= \\begin{bmatrix} 0.01&0.8\\ 0.002&-0.7 \\end{bmatrix} \]

Small weights can potentially be removed:

\[ W= \\begin{bmatrix} 0&0.8\\ 0&-0.7 \\end{bmatrix} \]

Thus:

\[ \\boxed{\\text{Pruning}=\\text{remove unnecessary parameters}} \]

---

# 6\. Why can pruning work?

Not every parameter contributes equally.

Some parameters may be:

- redundant
- near zero
- correlated with other neurons
- effectively unused

Therefore:

\[ \\boxed{\\text{large model}\\neq\\text{all parameters are essential}} \]

A model can sometimes lose a very large fraction of its parameters without catastrophic performance loss.

---

# 7\. Types of pruning

## Magnitude pruning

Remove weights with small absolute magnitude:

\[ \\boxed{|w|\\text{ small}\\Rightarrow\\text{prune}} \]

Simple and common.

---

## Similarity pruning

If two neurons behave almost identically:

\[ h_1(x)\\approx h_2(x) \]

they may be redundant.

Therefore:

\[ \\boxed{\\text{redundant neurons}\\rightarrow\\text{remove one}} \]

---

## Node/neuron pruning

Remove neurons whose weights or activations have very small norms:

\[ |w_i__|_2\\approx0 \]

---

## Structured pruning

Remove entire structures such as:

- neurons
- channels
- filters
- blocks

This is especially useful for real hardware.

### Why?

Unstructured pruning:

\[ \\text{random individual zeros} \]

may create sparse matrices that hardware cannot efficiently exploit.

Structured pruning:

\[ \\text{remove entire channel/filter} \]

creates smaller, regular computations.

Therefore:

\[ \\boxed{ \\text{structured pruning} \\rightarrow \\text{more practical speedups} } \]

### Exam distinction

> **Unstructured pruning removes individual weights; structured pruning removes regular groups such as neurons, channels, filters, or blocks. Structured sparsity is generally easier for hardware to exploit.**

---

# 8\. Iterative pruning

Aggressive pruning all at once can severely damage performance.

Instead:

\[ \\text{train} \\rightarrow \\text{prune} \\rightarrow \\text{fine-tune} \\rightarrow \\text{prune} \\rightarrow \\text{fine-tune} \\rightarrow\\cdots \]

This is **iterative pruning**.

### Intuition

> Remove parameters gradually and allow the network to compensate after each pruning step.

---

# 9\. Manifold hypothesis

Why can a network with millions/billions of parameters tolerate removing many of them?

One explanation is the **manifold hypothesis**.

The raw parameter space may be enormous:

\[ \\mathbb R^{1,000,000} \]

but useful solutions may occupy a much smaller effective region/manifold.

Therefore:

\[ \\boxed{ \\text{raw parameter dimensionality} \\gg \\text{effective dimensionality} } \]

This provides intuition for why overparameterised networks can contain substantial redundancy.

---

# 10\. Lottery Ticket Hypothesis

The Lottery Ticket Hypothesis asks:

> If a large network can be heavily pruned, why not simply train a small network from scratch?

The idea:

> A large randomly initialised network may contain a smaller **winning subnetwork** that, with an appropriate initialization/training process, can achieve comparable performance.

Conceptually:

\[ \\boxed{ \\text{large random network} \\rightarrow \\text{contains winning subnetwork} } \]

### Important intuition

A large network can contain useful subnetworks that are difficult to discover if we start directly with a small architecture.

---

# 11\. Combining compression techniques

Compression methods can be combined.

For example:

\[ \\boxed{\\text{Pruning}+\\text{Quantisation}} \]

gives:

- fewer parameters
- fewer bits per remaining parameter

This illustrates a general principle:

> **Different techniques exploit different types of redundancy and their benefits can compound.**

---

# 12\. PEFT — Parameter-Efficient Fine-Tuning

Now the problem changes.

Instead of asking:

> How can I make an existing model smaller?

we ask:

> **How can I adapt a huge pretrained model without updating all its parameters?**

Suppose:

\[ 100\\text{ billion parameters} \]

Full fine-tuning means updating all (100) billion.

But the pretrained model already contains useful knowledge.

Therefore:

\[ \\boxed{\\text{Only a small adaptation may be necessary}} \]

This is the motivation for **PEFT**.

---

# 13\. Layer freezing

The simplest PEFT method:

\[ \\boxed{\\text{freeze most layers}} \]

Only a subset remains trainable.

Example:

\[ \\underbrace{\\text{frozen}}_{\\text{no updates}} \\rightarrow \\underbrace{\\text{frozen}}_{\\text{no updates}} \\rightarrow \\underbrace{\\text{trainable}}\_{\\text{adapt}} \]

### Advantage

Less memory is needed because frozen parameters don't require:

- gradients
- optimizer states

### Disadvantage

Adaptation is restricted to selected layers.

---

# 14\. Adapters

Instead of changing the pretrained layers, insert small trainable modules.

\[ \\text{Frozen layer} \\rightarrow \\boxed{\\text{trainable adapter}} \\rightarrow \\text{Frozen layer} \]

Adapters often use a bottleneck:

\[ d\\rightarrow r\\rightarrow d \]

where:

\[ r\\ll d \]

The adapter learns a small residual correction:

\[ \\boxed{y=x+\\text{adapter}(x)} \]

### Key idea

> Keep the large pretrained model frozen and train only small task-specific modules.

---

# 15\. Prefix tuning

Prefix tuning also leaves the main model frozen.

Instead, learn a task-specific trainable prefix/context:

\[ \\boxed{ \\text{learned prefix}+\\text{input} \\rightarrow \\text{frozen model} } \]

Different tasks can have different prefixes.

Therefore:

\[ \\boxed{ \\text{one frozen model} + \\text{small task-specific parameters} } \]

---

# 16\. Spatially separable convolutions

Another way to reduce parameters is to replace a large transformation with smaller sequential transformations.

A (3\\times3) convolution:

\[ 9\\text{ parameters} \]

can sometimes be represented as:

\[ 3\\times1 \]

followed by:

\[ 1\\times3 \]

giving:

\[ 3+3=6 \]

instead of:

\[ 9 \]

So:

\[ \\boxed{9\\rightarrow6} \]

### Core principle

> **A large transformation may be approximated or represented as a composition of smaller transformations.**

This leads naturally to low-rank methods.

---

# 17\. Low-rank factorisation

Suppose:

\[ W\\in\\mathbb R^{d\\times d} \]

A normal matrix contains:

\[ d^2 \]

parameters.

Instead, approximate:

\[ \\boxed{W\\approx UV} \]

where:

\[ U\\in\\mathbb R^{d\\times r} \]

and:

\[ V\\in\\mathbb R^{r\\times d} \]

with:

\[ r\\ll d \]

Parameter count becomes:

\[ 2dr \]

instead of:

\[ d^2 \]

Therefore, if (r\\ll d), the representation is much smaller.

---

# 18\. SVD and truncated SVD

Singular Value Decomposition:

\[ \\boxed{A=U\\Sigma V^T} \]

If only a few singular values are important, keep the largest (k):

\[ \\boxed{ A\\approx U_k\\Sigma_kV\_k^T } \]

This is **truncated SVD**.

The idea is:

> Keep the most important directions and discard less-important directions.

---

# 19\. Critical distinction: weights do not have to be low-rank

This is a **major exam trap**.

It is tempting to say:

> Neural networks have low effective dimensionality, therefore their weight matrices must be low-rank.

That does **not necessarily follow**.

A trained weight matrix may not be sufficiently low-rank for aggressive direct SVD compression.

Therefore:

\[ \\boxed{ \\text{low intrinsic dimensionality} \\neq \\text{weight matrix is necessarily low-rank} } \]

---

# 20\. LoRA — the central concept

The key insight behind **LoRA** is:

> **The pretrained weights don't need to be low-rank. The fine-tuning update may be low-rank.**

Start with pretrained:

\[ W \]

Normal fine-tuning:

\[ W'=W+\\Delta W \]

LoRA assumes:

\[ \\boxed{\\Delta W\\approx BA} \]

where:

\[ A\\in\\mathbb R^{r\\times d} \]

\[ B\\in\\mathbb R^{d\\times r} \]

and:

\[ r\\ll d \]

Therefore, instead of training the full (d\\times d) update, train only:

\[ 2dr \]

parameters.

---

# 21\. Why LoRA saves so many parameters

Suppose:

\[ d=12,288,\\qquad r=8 \]

Full update:

\[ d^2=12,288^2 \]

approximately:

\[ 151\\text{ million parameters} \]

LoRA:

\[ 2dr = 2(12,288)(8) = 196,608 \]

So:

\[ \\boxed{ 151\\text{ million} \\rightarrow 196,608 } \]

This is an enormous reduction.

### Core LoRA philosophy

\[ \\boxed{ \\text{freeze pretrained model} + \\text{learn tiny low-rank correction} } \]

---

# 22\. LoRA initialisation

LoRA is commonly initialized so that:

\[ BA=0 \]

Therefore initially:

\[ W'=W \]

So the adapted model starts exactly like the pretrained model.

During training:

\[ BA\\rightarrow\\text{learned non-zero update} \]

Thus the model gradually adapts.

### Intuition

LoRA behaves like a learned residual correction:

\[ \\boxed{ \\text{new model} = \\text{original model} + \\text{small learned correction} } \]

---

# 23\. LoRA vs SVD — extremely important

Do **not** confuse these.

### SVD compression

You take an existing matrix:

\[ W \]

and approximate it:

\[ \\boxed{W\\approx BA} \]

Goal:

> **Compress the existing weights.**

### LoRA

Keep (W) intact and represent only the update compactly:

\[ \\boxed{W'=W+\\Delta W,\\qquad\\Delta W\\approx BA} \]

Goal:

> **Compress the fine-tuning update.**

### Memorise:

\[ \\boxed{\\text{SVD}=\\text{compress weights}} \]

\[ \\boxed{\\text{LoRA}=\\text{compress adaptation/update}} \]

This is arguably the **most important distinction in Lecture 4**.

---

# 24\. Why LoRA adds no inference overhead after merging

During training:

\[ W\_{\\text{effective}}=W+BA \]

After training, compute:

\[ W'=W+BA \]

and merge the result into the original weight matrix.

Therefore:

\[ \\boxed{\\text{LoRA can be merged into the model for inference}} \]

So inference can use an ordinary weight matrix.

### Advantage over adapters

Adapters remain additional modules unless similarly handled, while LoRA can be merged directly into the weights.

---

# 25\. LoRA modularity

Suppose we have one base model:

\[ W \]

We can train different LoRA updates:

\[ \\Delta W\_{\\text{French}} \]

\[ \\Delta W\_{\\text{medical}} \]

\[ \\Delta W\_{\\text{style}} \]

etc.

Instead of storing several complete model copies:

\[ \\boxed{ \\text{one base model} + \\text{many tiny LoRA modules} } \]

This makes task-specific adaptation highly storage-efficient.

---

# 26\. QLoRA

The lecture combines two ideas:

### Quantisation

Store the large pretrained model using low precision.

### LoRA

Train a small low-rank adaptation.

Therefore:

\[ \\boxed{ \\text{QLoRA} = \\text{Quantisation} + \\text{LoRA} } \]

Conceptually:

\[ \\boxed{ \\text{cheap base-model storage} + \\text{cheap task adaptation} } \]

This is particularly useful for adapting very large language models with limited hardware.

---

# 27\. The complete Lecture 4 progression

Memorise this chain:

### Question 1 — Too many bits?

\[ \\boxed{\\text{Quantisation}} \]

\[ 32\\text{ bits}\\rightarrow8/4\\text{ bits} \]

### Question 2 — Too many parameters?

\[ \\boxed{\\text{Pruning}} \]

Remove unnecessary parameters.

### Question 3 — Why can parameters be removed?

\[ \\boxed{ \\text{Redundancy} + \\text{manifold hypothesis} + \\text{lottery-ticket intuition} } \]

### Question 4 — Too many parameters to fine-tune?

\[ \\boxed{\\text{PEFT}} \]

Freeze most parameters.

### Question 5 — Need adaptation throughout the network?

\[ \\boxed{\\text{Adapters / Prefix tuning}} \]

Train small task-specific components.

### Question 6 — Can the update itself be compact?

\[ \\boxed{\\text{LoRA}} \]

\[ \\Delta W\\approx BA \]

### Question 7 — Can these ideas be combined?

\[ \\boxed{\\text{QLoRA}} \]

\[ \\text{Quantisation}+\\text{LoRA} \]

---

# 28\. High-yield comparison table

| Technique | What is reduced? | Core idea |
| --- | --- | --- |
| **Quantisation** | Numerical precision | Fewer bits per parameter |
| **Pruning** | Number of parameters | Remove unnecessary weights |
| **Magnitude pruning** | Small weights | Remove low-( |
| **Similarity pruning** | Redundant neurons | Remove similar features |
| **Structured pruning** | Channels/filters/blocks | Hardware-friendly removal |
| **Layer freezing** | Trainable parameters | Freeze most of the model |
| **Adapters** | Trainable parameters | Small bottleneck modules |
| **Prefix tuning** | Trainable parameters | Learn task-specific prefixes |
| **Low-rank factorisation** | Matrix parameters | (W\\approx UV) |
| **SVD** | Existing matrix representation | Keep dominant singular directions |
| **LoRA** | Fine-tuning update | (\\Delta W\\approx BA) |
| **QLoRA** | Storage + adaptation | Quantisation + LoRA |

---

# 29\. Critical exam distinctions

### Quantisation vs pruning

**Quantisation:**

> Keep parameters, reduce precision.

\[ \\boxed{\\text{same parameters, fewer bits}} \]

**Pruning:**

> Remove parameters.

\[ \\boxed{\\text{fewer parameters}} \]

---

### PTQ vs QAT

**PTQ:**

\[ \\boxed{\\text{train}\\rightarrow\\text{quantise}} \]

**QAT:**

\[ \\boxed{\\text{simulate quantisation during training}} \]

QAT is more expensive but generally preserves accuracy better.

---

### Unstructured vs structured pruning

**Unstructured:**

\[ \\text{individual weights} \]

**Structured:**

\[ \\text{neurons/channels/filters/blocks} \]

Structured pruning is generally more hardware-friendly.

---

### Compression vs PEFT

**Compression:**

\[ \\boxed{\\text{make model cheaper}} \]

**PEFT:**

\[ \\boxed{\\text{make adaptation cheaper}} \]

---

### SVD vs LoRA

**SVD:**

\[ \\boxed{W\\approx BA} \]

compress existing weights.

**LoRA:**

\[ \\boxed{W'=W+\\Delta W,\\quad\\Delta W\\approx BA} \]

compress the fine-tuning update.

---

### LoRA vs QLoRA

**LoRA:**

\[ \\boxed{\\text{low-rank adaptation}} \]

**QLoRA:**

\[ \\boxed{\\text{quantised base model + LoRA}} \]

---

# 30\. Likely exam questions

### Q1. Why is quantisation useful?

**Answer:** Quantisation reduces the number of bits used to represent each parameter, reducing memory/storage and potentially computation, at the cost of introducing approximation error.

### Q2. Compare PTQ and QAT.

**Answer:** PTQ applies quantisation after training and is cheap and simple but can cause larger accuracy degradation. QAT simulates quantisation during training, allowing the model to adapt to quantisation error, usually giving better accuracy at higher training cost.

### Q3. What is pruning?

**Answer:** Pruning removes parameters or structures considered unnecessary, reducing model size and potentially computation.

### Q4. Why is structured pruning useful?

**Answer:** It removes regular structures such as channels or filters, producing computations that existing hardware can exploit more efficiently than arbitrary sparse individual weights.

### Q5. Why prune iteratively?

**Answer:** Gradual pruning followed by fine-tuning allows the model to compensate for removed parameters and generally avoids the severe performance degradation that can result from pruning too aggressively at once.

### Q6. What is the Lottery Ticket Hypothesis?

**Answer:** A large randomly initialized network may contain a smaller subnetwork—a winning ticket—that can achieve comparable performance when appropriately trained.

### Q7. What is PEFT?

**Answer:** Parameter-efficient fine-tuning adapts a pretrained model by updating only a small subset of parameters rather than the entire model.

### Q8. What is an adapter?

**Answer:** A small trainable bottleneck module inserted into a frozen pretrained network. It learns a task-specific residual correction while the original model remains frozen.

### Q9. What is low-rank factorisation?

**Answer:** It represents a large matrix approximately as a product of smaller matrices, e.g. (W\\approx UV), reducing the number of parameters when the rank (r) is much smaller than the matrix dimension.

### Q10. Why doesn't direct low-rank compression necessarily work well for neural-network weights?

**Answer:** Although neural networks may have low effective dimensionality, their trained weight matrices are not necessarily sufficiently low-rank for aggressive low-rank approximation without significant performance loss.

### Q11. What is the central idea of LoRA?

**Answer:** LoRA freezes the pretrained weights and represents the fine-tuning update as a low-rank matrix:

\[ \\boxed{\\Delta W\\approx BA} \]

with (r\\ll d).

### Q12. How is LoRA different from SVD compression?

**Answer:** SVD compresses an existing weight matrix (W), whereas LoRA keeps (W) intact and represents the task-specific update (\\Delta W) in low-rank form.

### Q13. Why can LoRA be merged at inference?

**Answer:** After training, (W+BA) can be computed once and stored as the new weight matrix, so no separate LoRA computation is required during inference.

### Q14. What is QLoRA?

**Answer:** QLoRA combines quantisation of the pretrained base model with LoRA-based low-rank fine-tuning, reducing both model-storage requirements and adaptation cost.

---

# 31\. Must-memorise formulas

### Quantisation

\[ \\boxed{ \\text{FP32}\\rightarrow\\text{INT8/INT4} } \]

---

### Low-rank factorisation

\[ \\boxed{ W\\approx UV } \]

where:

\[ U\\in\\mathbb R^{d\\times r}, \\qquad V\\in\\mathbb R^{r\\times d}, \\qquad r\\ll d \]

Parameter count:

\[ \\boxed{d^2\\rightarrow2dr} \]

---

### SVD

\[ \\boxed{ A=U\\Sigma V^T } \]

Truncated:

\[ \\boxed{ A\\approx U_k\\Sigma_kV\_k^T } \]

---

### LoRA

\[ \\boxed{ W'=W+\\Delta W } \]

with:

\[ \\boxed{ \\Delta W\\approx BA } \]

and:

\[ \\boxed{ A\\in\\mathbb R^{r\\times d}, \\quad B\\in\\mathbb R^{d\\times r}, \\quad r\\ll d } \]

LoRA trainable parameters:

\[ \\boxed{2dr} \]

instead of:

\[ \\boxed{d^2} \]

---

### QLoRA

\[ \\boxed{ \\text{QLoRA} = \\text{Quantisation} + \\text{LoRA} } \]

---

# 32\. Ultra-compressed 30-second memory version

```
MODEL COMPRESSION
        ↓
Exploit redundancy
        ↓
────────────────────────────────

Too many BITS?
→ QUANTISATION
FP32 → INT8/INT4
fewer bits, possible quantisation error

PTQ:
train → quantise
cheap

QAT:
quantisation simulated during training
more expensive, usually better accuracy

────────────────────────────────

Too many PARAMETERS?
→ PRUNING
remove unnecessary weights

Magnitude:
small |w| → remove

Similarity:
similar neurons → remove redundancy

Structured:
remove channels/filters/blocks
→ hardware-friendly

Iterative:
prune → fine-tune → prune → fine-tune

────────────────────────────────

WHY DOES THIS WORK?
→ redundancy
→ manifold hypothesis
→ Lottery Ticket Hypothesis

────────────────────────────────

TOO MANY PARAMETERS TO FINE-TUNE?
→ PEFT

Freeze most layers
→ adapters
→ prefix tuning
→ LoRA

────────────────────────────────

LOW-RANK:
W ≈ UV

BUT:
weights do NOT necessarily have to be low-rank

LoRA:
ΔW ≈ BA

KEY:
SVD = compress W
LoRA = compress ΔW

LoRA:
freeze W
learn tiny BA
BA initially = 0
merge W + BA at inference

────────────────────────────────

QLoRA:
quantised base model + LoRA

────────────────────────────────

BIG IDEA:
The model is huge,
but the information needed to
compress it or adapt it
can be much smaller.
```

## Lecture 4 — single most important sentence

> **Neural networks are highly overparameterised, so we can exploit redundancy in numerical precision, parameters, architecture, and fine-tuning updates to reduce the cost of storing, computing, and adapting large models.**

### Highest-priority exam facts

If time is limited, prioritise these in this order:

1. **LoRA:** (\\Delta W\\approx BA)
2. **LoRA ≠ SVD:** update vs existing weights
3. **QLoRA:** quantisation + LoRA
4. **PTQ vs QAT**
5. **Quantisation vs pruning**
6. **Structured vs unstructured pruning**
7. **PEFT vs model compression**
8. **Why pruning can work:** redundancy / effective dimensionality
9. **Adapters and prefix tuning**
10. **Lottery Ticket Hypothesis**

This gives us the **Lecture 4 exam-compression layer**. Once you provide **Practical 4**, I can compress that separately and then we will have the complete **Lecture 4 + Practical 4** material ready for the eventual UAB Advanced Machine Learning mock exam.
