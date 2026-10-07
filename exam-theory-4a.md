Yes. This additional material is especially useful because it clarifies the **exam style and reasoning level** expected for Lecture 4.

I’ll treat it as part of the Lecture 4 exam-compression source material and incorporate the scenario-based, comparison, calculation, and synthesis questions into the final compression.

## Lecture 4 — Exam Compression: Model Compression & PEFT

### 1\. The central problem

Modern neural networks are **overparameterised and expensive**.

The lecture asks:

> **Do we really need all the parameters, all the numerical precision, and all the trainable weights?**

The main sources of inefficiency are:

- too many **bits** per parameter → **quantisation**
- too many **parameters** → **pruning**
- too many parameters to **update during fine-tuning** → **PEFT**
- redundant/high-dimensional transformations → **low-rank methods**
- low-dimensional task-specific updates → **LoRA**

### The big picture

\[ \\boxed{ \\text{Exploit redundancy in precision, parameters, architecture, and updates} } \]

---

# 2\. Compression vs PEFT — critical distinction

This is one of the first things to understand.

| Goal | Problem | Main methods |
| --- | --- | --- |
| **Model compression** | Make an existing model cheaper/smaller | Quantisation, pruning, factorisation |
| **Parameter-efficient fine-tuning (PEFT)** | Adapt a large pretrained model cheaply | Freezing, adapters, prefix tuning, LoRA |

### Model compression

Goal:

\[ \\boxed{\\text{same model} \\rightarrow \\text{cheaper model}} \]

### PEFT

Goal:

\[ \\boxed{\\text{same pretrained model} \\rightarrow \\text{cheap adaptation to new task}} \]

Do **not** confuse these.

---

# 3\. Quantisation

## Core idea

Represent weights using **fewer bits**.

Example:

\[ \\text{FP32}\\rightarrow\\text{INT8}\\rightarrow\\text{INT4} \]

Instead of storing a highly precise continuous floating-point value, map it to a smaller set of discrete values.

\[ \\boxed{\\text{many precise values}\\rightarrow\\text{fewer approximate values}} \]

---

## 4\. Why quantisation reduces memory

Bits per parameter:

| Representation | Bits |
| --- | --- |
| FP32 | 32 |
| FP16 | 16 |
| INT8 | 8 |
| INT4 | 4 |

For (10^9) parameters:

### FP32

\[ 10^9\\times32\\text{ bits}=4\\text{ GB} \]

### INT8

\[ 10^9\\times8\\text{ bits}=1\\text{ GB} \]

Therefore:

\[ \\boxed{\\text{FP32}\\rightarrow\\text{INT8}\\approx4\\times\\text{less weight storage}} \]

And:

\[ \\boxed{\\text{FP32}\\rightarrow\\text{INT4}\\approx8\\times\\text{less weight storage}} \]

The exact practical memory footprint can include metadata/scales and other model components, but the basic exam calculation is based on bits per parameter.

---

# 5\. Quantisation trade-off

Quantisation introduces approximation:

\[ w\\neq Q(w) \]

Therefore:

\[ \\boxed{ \\text{less precision} \\leftrightarrow \\text{less memory/computation} } \]

Potential problem:

\[ \\text{quantisation error}\\rightarrow\\text{accuracy degradation} \]

Aggressive quantisation can work surprisingly well, including around **4-bit representations** for some modern large models.

### Important exam reasoning

**Question:** Why doesn't INT8 automatically make inference faster?

**Answer:** Memory reduction does not guarantee computational speedup. The hardware/software stack must efficiently support INT8 arithmetic. If the device lacks low-precision acceleration, the model may save memory without achieving a large latency reduction.

---

# 6\. PTQ vs QAT

## Post-training quantisation — PTQ

Pipeline:

\[ \\boxed{ \\text{Train FP model} \\rightarrow \\text{quantise} } \]

Advantages:

- simple
- cheap
- little/no retraining
- useful when the trained model already exists

Disadvantage:

- model was not trained to compensate for quantisation errors
- accuracy may decrease

---

## Quantisation-aware training — QAT

During training, quantisation effects are simulated:

\[ \\text{weights} \\rightarrow \\text{quantise} \\rightarrow \\text{dequantise} \\rightarrow \\text{forward pass} \]

The model learns to compensate for the resulting errors.

### Key comparison

| PTQ | QAT |
| --- | --- |
| Cheap | More expensive |
| Simple | More complex |
| No/full retraining not required | Requires training |
| Can lose more accuracy | Usually better accuracy |
| Good when deployment is immediate | Good when accuracy is critical |

### Exam trigger

If the question says:

> "Accuracy dropped significantly after PTQ. What should you try?"

Answer:

\[ \\boxed{\\text{QAT}} \]

because the model can learn to compensate for quantisation effects.

---

# 7\. Pruning

Quantisation asks:

> **Do we need all the precision?**

Pruning asks:

> **Do we need all the parameters?**

Example:

\[ W= \\begin{bmatrix} 0.01&0.8\\ 0.002&-0.7 \\end{bmatrix} \]

Small weights may be removed:

\[ W'= \\begin{bmatrix} 0&0.8\\ 0&-0.7 \\end{bmatrix} \]

Therefore:

\[ \\boxed{\\text{Pruning}=\\text{remove unnecessary parameters}} \]

---

# 8\. Why can pruning work?

Neural networks are often **overparameterised**.

Not every parameter is equally important.

Parameters may be:

- redundant
- near zero
- correlated
- effectively unused

Therefore:

\[ \\boxed{ \\text{large parameter count}\\neq \\text{all parameters are essential} } \]

A model may sometimes lose a very large fraction of its weights while retaining much of its performance.

---

# 9\. Types of pruning

## Magnitude pruning

Remove weights with small magnitude:

\[ \\boxed{|w|\\text{ small}\\Rightarrow\\text{prune}} \]

Simple and easy to implement.

---

## Similarity pruning

Find neurons/features producing similar outputs:

\[ h_1(x)\\approx h_2(x) \]

If two neurons perform nearly the same function, one may be redundant.

\[ \\boxed{\\text{similar features}\\rightarrow\\text{remove redundancy}} \]

### Critical distinction

- **Magnitude pruning:** looks at **individual weight size**.
- **Similarity pruning:** looks at **redundancy between features/neurons**.

---

# 10\. Structured vs unstructured pruning

## Unstructured pruning

Remove individual weights.

Produces scattered zeros:

```
0  0.8  0
0  0    -0.7
0.2 0   0
```

Potential issue:

> Hardware may not efficiently exploit arbitrary sparsity.

Therefore:

\[ \\boxed{\\text{90% zeros}\\neq\\text{automatically 10× faster}} \]

---

## Structured pruning

Remove entire structures:

- neurons
- channels
- filters
- blocks

This produces smaller, regular dense computations.

### Why is it often more useful in practice?

GPUs and other hardware are generally designed to process regular dense matrix/tensor operations efficiently.

Thus:

\[ \\boxed{ \\text{structured pruning} \\rightarrow \\text{smaller regular computation} \\rightarrow \\text{more reliable practical speedup} } \]

---

# 11\. The critical pruning exam trap

### Question

> A model loses 90% of its weights through pruning. Does inference automatically become 10× faster?

### Answer

**No.**

The zeros must be exploited using an appropriate sparse representation and hardware/software support.

Otherwise, the system may still perform much of the original computation.

### Remember

\[ \\boxed{ \\text{parameter reduction} \\neq \\text{automatic runtime reduction} } \]

This distinction is highly exam-worthy.

---

# 12\. Iterative pruning

Aggressively pruning everything at once can severely damage performance.

Instead:

\[ \\text{train} \\rightarrow \\text{prune} \\rightarrow \\text{fine-tune} \\rightarrow \\text{prune} \\rightarrow \\text{fine-tune} \\rightarrow\\cdots \]

This allows the remaining model to adapt progressively.

\[ \\boxed{\\text{iterative pruning}=\\text{gradually remove parameters + recover performance}} \]

---

# 13\. Manifold hypothesis / intrinsic dimensionality

A key conceptual question:

> Why can a model with millions/billions of parameters be compressed so aggressively?

One explanation is that the effective useful solution space has much lower dimensionality than the raw parameter space.

For example:

\[ \\mathbb R^{1,000,000} \]

may contain useful solutions concentrated around a much smaller effective manifold.

Thus:

\[ \\boxed{ \\text{raw dimensionality} \\gg \\text{effective dimensionality} } \]

This provides conceptual motivation for compression and low-dimensional adaptation.

---

# 14\. Lottery Ticket Hypothesis

The Lottery Ticket Hypothesis proposes roughly:

> A large randomly initialised network contains smaller subnetworks that can achieve comparable performance when trained appropriately.

Conceptually:

\[ \\boxed{ \\text{large random network} \\rightarrow \\text{contains a useful "winning ticket"} } \]

### Why might a large network be useful?

It provides many possible subnetworks and initialisations.

A small network trained independently may not contain the same favourable subnetwork/initialisation.

### Exam interpretation

The hypothesis helps explain why:

\[ \\boxed{ \\text{train large}\\rightarrow\\text{find/prune useful subnetwork} } \]

can sometimes outperform simply training a small network from scratch.

---

# 15\. Combining quantisation and pruning

Compression techniques are not mutually exclusive.

For example:

\[ \\boxed{ \\text{Pruning}+\\text{Quantisation} } \]

gives:

- fewer parameters
- fewer bits per remaining parameter

Thus:

\[ \\boxed{ \\text{fewer weights} + \\text{cheaper representation} } \]

can compound the benefits.

---

# 16\. PEFT — the second half of the lecture

Now the problem changes.

Instead of:

> "How do I make this model smaller?"

we ask:

> **"How do I adapt a huge pretrained model without updating billions of parameters?"**

Suppose:

\[ 100\\text{ billion parameters} \]

Full fine-tuning updates all of them.

This is expensive in:

- GPU memory
- gradients
- optimizer states
- training computation
- storage of task-specific models

But the pretrained model already contains useful knowledge.

Therefore:

\[ \\boxed{ \\text{Only a small adaptation may be necessary} } \]

---

# 17\. Layer freezing

Simplest PEFT strategy:

\[ \\boxed{\\text{freeze most parameters}} \]

and train only selected layers.

Example:

```
Frozen → Frozen → Trainable → Trainable
```

### What does freezing save?

Primarily **training resources**:

- gradients
- optimizer states
- trainable parameters
- training memory
- potentially some training computation

### What does it NOT automatically save?

**Inference computation.**

Frozen layers still execute their forward pass.

### Critical exam trap

> "If I freeze 90% of the layers, inference becomes 90% cheaper."

**False.**

Freezing is primarily a **training-efficiency technique**.

---

# 18\. Adapter modules

Instead of modifying the pretrained weights, insert small trainable modules.

```
Frozen layer
      ↓
Small trainable adapter
      ↓
Frozen layer
```

Often use a bottleneck:

\[ d\\rightarrow r\\rightarrow d \]

where:

\[ r\\ll d \]

The adapter learns a small correction, often conceptually:

\[ y=x+\\text{adapter}(x) \]

### Advantages

- original model remains frozen
- few trainable parameters
- task-specific parameters are small
- different adapters can represent different tasks

### Disadvantage

Adapters remain additional computation at inference, so they can introduce some latency.

---

# 19\. Prefix tuning

Instead of modifying the main model weights, learn a task-specific prefix/context.

\[ \\boxed{ \\text{learned prefix}+\\text{input} \\rightarrow \\text{frozen model} } \]

The base model remains shared.

Different tasks can use different small learned prefixes.

---

# 20\. Separable convolutions — important intuition

A (3\\times3) convolution contains:

\[ 3\\times3=9 \]

parameters per relevant channel pair.

It can sometimes be decomposed into:

\[ 3\\times1 \]

followed by:

\[ 1\\times3 \]

giving:

\[ 3+3=6 \]

parameters.

Thus:

\[ 9\\rightarrow6 \]

This illustrates the general principle:

\[ \\boxed{ \\text{large transformation} \\rightarrow \\text{composition of smaller transformations} } \]

This motivates low-rank factorisation.

---

# 21\. Low-rank factorisation

Suppose:

\[ W\\in\\mathbb R^{d\\times d} \]

contains:

\[ d^2 \]

parameters.

Approximate it with:

\[ W\\approx UV \]

where:

\[ U\\in\\mathbb R^{d\\times r}, \\qquad V\\in\\mathbb R^{r\\times d} \]

and:

\[ r\\ll d. \]

Then the parameter count becomes:

\[ dr+rd=2dr \]

instead of:

\[ d^2. \]

### Compression ratio

\[ \\boxed{ \\frac{2dr}{d^2}=\\frac{2r}{d} } \]

---

# 22\. SVD and truncated SVD

Singular Value Decomposition:

\[ A=U\\Sigma V^T \]

If only a few singular values are important:

\[ A\\approx U_k\\Sigma_kV\_k^T \]

where only the largest (k) singular directions are retained.

This removes less-important directions.

### But important warning

The fact that neural networks have low **effective dimensionality** does **not** mean every trained weight matrix is sufficiently low-rank for aggressive SVD compression.

Directly replacing:

\[ W\\rightarrow W\_{\\text{low-rank}} \]

can cause significant performance degradation.

This leads directly to LoRA.

---

# 23\. The most important concept: LoRA

Suppose a pretrained model has:

\[ W \]

Full fine-tuning learns:

\[ W'=W+\\Delta W \]

where:

\[ \\Delta W\\in\\mathbb R^{d\\times d} \]

would normally require:

\[ d^2 \]

trainable parameters.

LoRA assumes:

\[ \\boxed{ \\Delta W\\approx BA } \]

where:

\[ A\\in\\mathbb R^{r\\times d} \]

and:

\[ B\\in\\mathbb R^{d\\times r} \]

with:

\[ r\\ll d. \]

Thus only:

\[ 2dr \]

parameters are trained.

---

# 24\. LoRA parameter-count calculation

Full fine-tuning:

\[ d^2 \]

LoRA:

\[ 2dr \]

Ratio:

\[ \\boxed{ \\frac{2dr}{d^2}=\\frac{2r}{d} } \]

### Example

If:

\[ d=10,000,\\qquad r=10 \]

Full update:

\[ 10,000^2=100,000,000 \]

LoRA:

\[ 2(10)(10,000)=200,000 \]

Therefore:

\[ \\frac{200,000}{100,000,000}=0.002 \]

So LoRA trains:

\[ \\boxed{0.2%} \]

of the parameters required for the full update.

### Exam skill

Be able to **derive this**, not merely recognise the formula.

---

# 25\. Why LoRA works

The critical assumption is:

> **The pretrained model does not need a completely new set of weights. The task-specific change can often lie in a low-dimensional subspace.**

Therefore:

\[ \\boxed{ \\text{huge pretrained model} + \\text{small low-rank correction} } \]

can be sufficient.

This is fundamentally different from assuming the original model weights themselves are low-rank.

---

# 26\. LoRA ≠ SVD compression

This is arguably the **most important distinction in Lecture 4**.

### SVD compression

You have:

\[ W \]

and want:

\[ W\\approx BA \]

You are compressing the **existing weights**.

### LoRA

You keep:

\[ W \]

and learn:

\[ \\Delta W\\approx BA \]

You are compressing the **fine-tuning update**.

Therefore:

\[ \\boxed{\\text{SVD}\\rightarrow\\text{compress weights}} \]

while:

\[ \\boxed{\\text{LoRA}\\rightarrow\\text{compress adaptation}} \]

### Excellent exam sentence

> The weights themselves do not have to be low-rank; LoRA assumes that the **task-specific update** can be represented in a low-dimensional subspace.

---

# 27\. LoRA initialisation

LoRA can be initialised so:

\[ BA=0 \]

Therefore initially:

\[ W'=W \]

The adapted model starts exactly at the pretrained model.

During training:

\[ BA\\rightarrow\\text{learned correction} \]

So:

\[ \\boxed{ \\text{pretrained model} \\rightarrow \\text{gradually learned adaptation} } \]

This is analogous to a residual correction.

---

# 28\. LoRA and inference

During training:

\[ W\_{\\text{effective}}=W+BA \]

After training, we can merge:

\[ \\boxed{ W'=W+BA } \]

and replace the original weight with the merged weight.

Therefore, unlike adapters:

\[ \\boxed{\\text{LoRA can have essentially no additional inference-layer overhead after merging}} \]

This is a major practical advantage.

---

# 29\. LoRA is modular

One base model:

\[ W \]

can have many task-specific updates:

\[ \\Delta W_1,\\Delta W_2,\\Delta W\_3,\\ldots \]

For example:

- medical task
- translation task
- style task
- classification task

Instead of storing several complete copies of the model:

\[ \\boxed{ \\text{one base model} + \\text{many tiny LoRA modules} } \]

This makes task-specific deployment/storage much cheaper.

---

# 30\. QLoRA

Combine:

- quantisation of the base model
- LoRA for adaptation

\[ \\boxed{ \\text{QLoRA} = \\text{Quantisation} + \\text{LoRA} } \]

Conceptually:

```
Quantised frozen base model
            +
      tiny LoRA update
            ↓
       task-specific model
```

This simultaneously reduces:

- base-model memory requirements
- number of trainable parameters

### Key idea

\[ \\boxed{ \\text{cheap storage} + \\text{cheap adaptation} } \]

---

# 31\. High-yield comparison table

| Method | What changes? | Main saving | Training required? | Inference effect |
| --- | --- | --- | --- | --- |
| **Quantisation** | Numerical precision | Memory / potentially computation | PTQ: no; QAT: yes | Can reduce memory/time if hardware supports it |
| **Pruning** | Number of active parameters | Parameters / potentially computation | Often fine-tuning | Can reduce inference cost if sparsity is exploited |
| **Layer freezing** | Trainable parameter set | Training memory/computation | Yes, but fewer parameters | Little/no inference saving by itself |
| **Adapters** | Adds small trainable modules | Trainable parameters | Yes | Some additional computation |
| **Prefix tuning** | Adds trainable prefix/context | Trainable parameters | Yes | Depends on implementation |
| **Low-rank factorisation** | Represents transformation compactly | Parameters/computation | Depends | Can reduce computation |
| **LoRA** | Low-rank update | Trainable parameters | Yes | Can be merged → essentially no extra inference cost |
| **QLoRA** | Quantised base + LoRA update | Memory + trainable parameters | Yes | Efficient after deployment |

---

# 32\. What each technique is really exploiting

This is a useful way to memorise the entire lecture.

| Technique | Exploits |
| --- | --- |
| **Quantisation** | Excess numerical precision |
| **Magnitude pruning** | Unimportant individual weights |
| **Similarity pruning** | Redundant features/neurons |
| **Structured pruning** | Redundant computational structures |
| **Layer freezing** | Parameters that don't need task-specific updates |
| **Adapters** | Small task-specific corrections |
| **Prefix tuning** | Small task-specific contextual changes |
| **Low-rank factorisation** | Redundant matrix directions |
| **LoRA** | Low-dimensional fine-tuning updates |
| **QLoRA** | Low-precision storage + low-dimensional adaptation |

---

# 33\. The most important exam traps

### Trap 1 — "Fewer parameters = automatically faster"

**False.**

Hardware/software must exploit the compression.

---

### Trap 2 — "INT8 always makes inference faster"

**False.**

Speed depends on hardware support for low-precision operations.

---

### Trap 3 — "Freezing layers reduces inference computation"

**False.**

It primarily reduces **training** cost.

---

### Trap 4 — "LoRA assumes pretrained weights are low-rank"

**False.**

LoRA assumes the **fine-tuning update** can be low-rank.

---

### Trap 5 — "LoRA and SVD do the same thing"

**False.**

\[ \\text{SVD}: W\\rightarrow W\_{\\text{low-rank}} \]

\[ \\text{LoRA}: \\Delta W\\rightarrow BA \]

---

### Trap 6 — "90% pruning means 10× inference speedup"

**Not necessarily.**

Sparse computation must actually be exploited.

---

### Trap 7 — "Quantisation and pruning are alternatives"

**False.**

They can be combined.

---

### Trap 8 — "Adapters and LoRA have identical inference behaviour"

**False.**

Adapters introduce additional modules; LoRA can be merged into the original weights.

---

# 34\. Exam-grade scenario reasoning

For a demanding exam, think in terms of **constraint → method**.

### Scenario A

> Model is too large to fit in device memory.

Think:

\[ \\boxed{\\text{Quantisation}} \]

Potentially combine with pruning.

---

### Scenario B

> Model is accurate before PTQ but accuracy drops significantly afterward.

Think:

\[ \\boxed{\\text{QAT}} \]

---

### Scenario C

> Model contains huge amounts of redundant parameters.

Think:

\[ \\boxed{\\text{Pruning}} \]

---

### Scenario D

> 90% of weights are zero but inference isn't much faster.

Think:

\[ \\boxed{\\text{sparse hardware/software support / structured pruning}} \]

---

### Scenario E

> Huge pretrained LLM must be adapted to a new task with limited GPU memory.

Think:

\[ \\boxed{\\text{PEFT}} \]

especially:

\[ \\boxed{\\text{LoRA / QLoRA}} \]

---

### Scenario F

> Need different task-specific adaptations while sharing one base model.

Think:

\[ \\boxed{\\text{LoRA modules}} \]

---

### Scenario G

> Need adaptation throughout the network but cannot update the full model.

Think:

\[ \\boxed{\\text{Adapters or LoRA}} \]

---

### Scenario H

> Need essentially no additional inference architecture after fine-tuning.

Think:

\[ \\boxed{\\text{LoRA + weight merging}} \]

---

# 35\. Must-know calculations

## Quantisation

\[ \\boxed{ \\text{storage}\\propto\\text{number of parameters}\\times\\text{bits/parameter} } \]

FP32 → INT8:

\[ \\boxed{4\\times\\text{smaller}} \]

FP32 → INT4:

\[ \\boxed{8\\times\\text{smaller}} \]

---

## Low-rank factorisation

Full matrix:

\[ d^2 \]

Low-rank:

\[ 2dr \]

Ratio:

\[ \\boxed{\\frac{2r}{d}} \]

---

## LoRA

Full update:

\[ d^2 \]

LoRA update:

\[ 2dr \]

Percentage of full update:

\[ \\boxed{ 100\\frac{2r}{d}% } \]

---

# 36\. High-probability exam questions

### Q1. What is quantisation and why does it reduce memory?

Explain discrete lower-precision representation and calculate FP32 → INT8 savings.

### Q2. Why can quantisation reduce accuracy?

Explain approximation/quantisation error.

### Q3. INT8 reduces memory but not runtime. Why?

Discuss hardware/software support.

### Q4. PTQ causes an accuracy drop. What should you try?

\[ \\boxed{\\text{QAT}} \]

Explain why.

### Q5. Does 90% pruning guarantee 10× speedup?

No. Explain sparse representation and hardware support.

### Q6. Compare magnitude and similarity pruning.

Weight magnitude vs feature redundancy.

### Q7. Why can structured pruning produce better practical speedups?

Regular smaller tensors are easier for hardware to process.

### Q8. Why does layer freezing save training cost but not necessarily inference cost?

Frozen parameters don't require gradients/optimizer states, but still perform forward computation.

### Q9. Derive the LoRA parameter reduction.

\[ d^2\\rightarrow2dr \]

### Q10. Why is LoRA not the same as SVD compression?

Weights vs fine-tuning update.

### Q11. Why can LoRA be merged without additional inference cost?

\[ W'=W+BA \]

### Q12. Why can one base model + multiple LoRA modules be more efficient than multiple full models?

Only tiny task-specific updates need to be stored.

### Q13. Design a strategy for adapting a huge LLM under limited GPU memory.

Strong answer:

\[ \\boxed{\\text{quantised frozen base}+\\text{LoRA}} \]

i.e. QLoRA-style strategy.

### Q14. Explain the apparent contradiction that billions of parameters can be mostly unnecessary.

Discuss overparameterisation, redundancy, intrinsic dimensionality, pruning and low-dimensional adaptation.

---

# 37\. Synthesis question — very important

A likely demanding exam question could give you a constraint such as:

> You have a 100B-parameter pretrained model. You need to deploy it for 20 different tasks. GPU memory is limited, and you want to avoid storing 20 complete copies.

A strong solution is:

1. **Quantise** the base model to reduce storage/memory.
2. **Freeze** the base model.
3. Use **LoRA** for each task.
4. Store separate small (A,B) matrices for each task.
5. Merge LoRA into the base weights when appropriate for deployment.

Conceptually:

\[ \\boxed{ \\text{one quantised base model} + \\text{20 tiny LoRA adaptations} } \]

This is the type of question where the examiner tests whether you understand **why** each method is being used rather than whether you can simply define it.

---

# 38\. 30-second exam memory sheet

If you have almost no time:

```
MODEL COMPRESSION
        ↓
Exploit redundancy

Quantisation
→ fewer bits
→ FP32 → INT8/INT4
→ memory ↓
→ approximation error
→ PTQ = cheap
→ QAT = better accuracy, more training

Pruning
→ remove parameters
→ magnitude = small weights
→ similarity = redundant features
→ structured = hardware-friendly
→ 90% sparsity ≠ automatically 10× faster

PEFT
→ adapt pretrained model cheaply

Freezing
→ fewer trainable parameters
→ training cost ↓
→ inference not automatically ↓

Adapters
→ small trainable bottleneck modules
→ some inference overhead

Prefix tuning
→ learn small task-specific prefix

Low-rank
→ W ≈ UV
→ d² → 2dr

SVD
→ compress EXISTING WEIGHTS

LoRA
→ freeze W
→ learn ΔW ≈ BA
→ compress FINE-TUNING UPDATE
→ d² → 2dr
→ can merge → no extra inference layer

QLoRA
→ quantised base + LoRA
→ cheap storage + cheap adaptation
```

## The single most important sentence

> **Model compression exploits redundancy in the model itself, while PEFT exploits the fact that adapting a large pretrained model often requires only a small, low-dimensional change.**

And the **highest-priority distinction to memorise** is:

\[ \\boxed{ \\text{SVD: compress }W \\qquad\\neq\\qquad \\text{LoRA: compress }\\Delta W } \]

This Lecture 4 compression is now structured so it can later be combined with the **Lecture 4 practical** and then ultimately with Lectures 1–3 + practicals to build the final UAB Advanced Machine Learning mock exam.
