# Advanced Machine Learning — Comprehensive MCQ Practice

Below is a **UAB-style MCQ bank for Chapters 1–4**, deliberately modeled on the real exam you provided.

### How to use it

- **“Mark all that apply”** means there may be more than one correct answer.
- The number of correct statements is **not related to the number of marks**, matching the real exam.
- Distractors are intentionally plausible.
- The explanations focus on **conceptual understanding and intuition**, rather than memorising definitions.
- Some questions ask you to distinguish closely related concepts, because that is a strong feature of the real exam.

---

# Chapter 1 — Transfer Learning and Domain Adaptation

## Q1 — Transfer learning

**Which of the following are generally true about transfer learning? Mark all that apply.**

- □ A model can reuse knowledge learned from a source task when solving a target task.
- □ Transfer learning is generally more useful when the source and target problems have some useful relationship.
- □ Transfer learning always requires the source and target datasets to have identical label spaces.
- □ A pretrained model can sometimes achieve useful target-task performance with substantially less target-task data.
- □ Transfer learning guarantees better target-task performance than training from scratch.

### Answer

**Correct: 1, 2, 4**

### Explanation

- **1 — True.** This is the central idea of transfer learning.
- **2 — True.** Related source and target domains/tasks usually make transferred representations more useful.
- **3 — False.** The source and target tasks can have different label spaces. For example, a pretrained ImageNet classifier can provide a feature extractor for a completely different classification problem.
- **4 — True.** Pretraining can provide useful representations, reducing the amount of target-labelled data required.
- **5 — False.** Transfer can sometimes cause **negative transfer**, where the source knowledge actually hurts target performance.

---

## Q2 — Fine-tuning

**Which of the following are true when fine-tuning a pretrained neural network? Mark all that apply.**

- □ The pretrained parameters can be used as an initialization for the target task.
- □ Fine-tuning necessarily updates every parameter in the network.
- □ The learning rate used during fine-tuning may be smaller than the pretraining learning rate.
- □ The final prediction layer may need to be modified if the target task has a different output space.
- □ Fine-tuning requires the source and target tasks to have exactly the same input distribution.

### Answer

**Correct: 1, 3, 4**

### Explanation

- **1 — True.** The pretrained model provides the starting point.
- **2 — False.** Fine-tuning can update all parameters, but one can also freeze some layers.
- **3 — True.** A smaller learning rate is commonly used to avoid destroying useful pretrained representations.
- **4 — True.** For example, a model pretrained for 1000 ImageNet classes might receive a new classification head for a 10-class target problem.
- **5 — False.** Some domain shift is acceptable; understanding and handling it is precisely part of transfer/domain adaptation.

---

## Q3 — Negative transfer

A model pretrained on a source domain performs **worse** on the target task after transfer than a model trained from scratch.

**Which explanations could plausibly account for this? Mark all that apply.**

- □ The source and target domains are substantially different.
- □ Features learned from the source task are poorly suited to the target task.
- □ The pretrained model has necessarily become underparameterised.
- □ Fine-tuning may preserve inappropriate source-task biases.
- □ Transfer learning mathematically guarantees negative transfer whenever datasets are different.

### Answer

**Correct: 1, 2, 4**

### Explanation

Negative transfer occurs when transferred knowledge is not useful—or actively harmful.

A major cause is **domain/task mismatch**. A model trained on one type of data may learn representations that are poorly suited to another.

The important conceptual point is:

> **Pretraining provides a useful initialization, not a guarantee of useful knowledge.**

---

## Q4 — Freezing layers

**Which statements about freezing layers during transfer learning are true? Mark all that apply.**

- □ Frozen parameters generally do not receive gradient-based updates.
- □ Frozen layers can still participate in the forward pass.
- □ Freezing layers can reduce the number of trainable parameters.
- □ Freezing a layer removes it from the model.
- □ Freezing can reduce the memory required for storing gradients for those parameters.

### Answer

**Correct: 1, 2, 3, 5**

### Explanation

This is a particularly important distinction.

A frozen layer is **still part of the model**.

During inference:

$$
x \rightarrow f_{\text{frozen}}(x)
$$

still occurs.

During training, however, its parameters are not updated.

Therefore:

- fewer trainable parameters;
- fewer gradients;
- potentially less optimizer state;

but **not necessarily less forward computation**.

---

## Q5 — Domain adaptation

**Which statements correctly describe domain adaptation? Mark all that apply.**

- □ It addresses situations where the source and target domains differ.
- □ Domain adaptation can be useful even when the underlying task is related between domains.
- □ Domain adaptation necessarily means changing the neural-network architecture.
- □ Domain shift can occur even when the task labels have the same meaning.
- □ Domain adaptation and transfer learning are completely unrelated concepts.

### Answer

**Correct: 1, 2, 4**

### Explanation

A simple example is:

- source: photographs of objects;
- target: sketches of the same objects.

The classification task may remain the same, but the input distribution changes.

This is **domain shift**.

Domain adaptation is therefore concerned with making learned representations/models work effectively across domains.

---

## Q6 — Deployment performance

**Which situations could cause deployment performance to be worse than validation performance? Mark all that apply.**

- □ The deployment data distribution differs from the training/validation distribution.
- □ The validation set is not representative of deployment.
- □ The model has overfit to characteristics specific to the development data.
- □ The model has learned features that do not generalise to the deployment environment.
- □ A sufficiently large validation set mathematically guarantees deployment performance.

### Answer

**Correct: 1, 2, 3, 4**

### Explanation

A validation score estimates performance under the assumptions represented by the validation data.

It does **not** guarantee future deployment performance.

A particularly important exam distinction is:

$$
\text{good validation performance}
\not\Rightarrow
\text{guaranteed good deployment performance}.
$$

---

# Chapter 2 — Curriculum Learning and Semi-Supervised Learning

## Q7 — Curriculum learning

**Which statements about curriculum learning are true? Mark all that apply.**

- □ Training examples can be ordered from easier to harder.
- □ The motivation is partly inspired by learning progressively more difficult concepts.
- □ Curriculum learning necessarily requires manually labelled difficulty scores.
- □ The curriculum can potentially make optimisation easier.
- □ Curriculum learning means permanently removing all difficult examples from training.

### Answer

**Correct: 1, 2, 4**

### Explanation

A curriculum may begin with relatively easy examples and progressively introduce harder ones.

Difficulty does **not necessarily need to be manually labelled**. It can be estimated using model loss, confidence, uncertainty, or other criteria.

The goal is generally not to discard difficult examples permanently, but to control **when/how they contribute** to learning.

---

## Q8 — Self-paced learning

Suppose a model computes a loss $L_i$ for each training example. At each epoch, it only trains on examples satisfying

$$
L_i < \lambda.
$$

The parameter $\lambda$ is gradually increased.

**What does this algorithm most closely represent?**

- □ Knowledge distillation
- □ Self-paced/curriculum learning
- □ Adversarial training
- □ Quantisation-aware training
- □ Federated learning

### Answer

**Correct: Self-paced/curriculum learning**

### Explanation

The model itself estimates example difficulty through loss.

Initially, only examples with relatively low loss are selected:

$$
L_i < \lambda.
$$

As $\lambda$ increases, harder examples become eligible.

This is exactly the kind of **algorithm interpretation** emphasized in the real exam.

---

## Q9 — Difficulty estimation

**Which quantities could plausibly be used as an example difficulty score? Mark all that apply.**

- □ Training loss
- □ Model uncertainty
- □ Prediction confidence
- □ Randomly generated parameter values
- □ Prediction error

### Answer

**Correct: 1, 2, 3, 5**

### Explanation

Difficulty is not uniquely defined.

Possible indicators include:

$$
\text{difficulty} \approx L(x,y),
$$

uncertainty, low confidence, or prediction error.

A random parameter value does not provide a meaningful measure of example difficulty.

---

## Q10 — Semi-supervised learning

Suppose you have:

- 1,000 labelled examples;
- 100,000 unlabelled examples.

**Which statements about semi-supervised learning are true? Mark all that apply.**

- □ It can exploit both labelled and unlabelled data.
- □ It is useful when obtaining labels is expensive.
- □ The unlabelled data must always be manually labelled before training.
- □ The usefulness of unlabelled data depends on assumptions about the relationship between the data distribution and the task.
- □ Semi-supervised learning necessarily means training without any labels.

### Answer

**Correct: 1, 2, 4**

### Explanation

Semi-supervised learning sits between:

$$
\text{supervised learning}
$$

and

$$
\text{unsupervised learning}.
$$

There are labelled examples, but unlabelled examples are also exploited.

A key conceptual point is that **unlabelled data is not automatically useful**. Its usefulness depends on whether the information it provides relates meaningfully to the target task.

---

## Q11 — Pseudo-labeling

A classifier is trained using a small labelled dataset. It is then used to predict labels for unlabelled examples. High-confidence predictions are added to the training set.

**Which statements are true? Mark all that apply.**

- □ This is an example of pseudo-labeling.
- □ The model's predictions are treated as approximate labels.
- □ Incorrect high-confidence predictions can potentially reinforce model errors.
- □ Pseudo-labeling guarantees that the newly generated labels are correct.
- □ Confidence thresholds can be used to control which pseudo-labels are accepted.

### Answer

**Correct: 1, 2, 3, 5**

### Explanation

Pseudo-labeling exploits the model's own predictions:

$$
x_u \rightarrow \hat y_u.
$$

A confidence threshold might then select only:

$$
p(\hat y_u|x_u) > \tau.
$$

The danger is **confirmation bias**: incorrect predictions can become training targets and reinforce themselves.

---

## Q12 — Curriculum versus pseudo-labeling

**Which statement best distinguishes curriculum learning from pseudo-labeling?**

- □ Curriculum learning controls the order/selection/difficulty of training examples, while pseudo-labeling creates approximate labels for unlabelled data.
- □ Curriculum learning always requires unlabelled data, while pseudo-labeling never does.
- □ They are mathematically identical.
- □ Pseudo-labeling is only applicable to regression.

### Answer

**Correct: 1**

### Explanation

They address different problems.

**Curriculum/self-paced learning:**

> Which examples should contribute to training, and when?

**Pseudo-labeling:**

> Can we assign useful approximate labels to unlabelled examples?

They can, however, be combined.

---

# Chapter 3 — Self-Supervised and Contrastive Learning

## Q13 — Self-supervised learning

**Which statements about self-supervised learning are true? Mark all that apply.**

- □ It constructs a learning signal from the data itself.
- □ It can reduce reliance on manually annotated labels.
- □ It is necessarily identical to unsupervised clustering.
- □ Pretext tasks can be used to learn useful representations.
- □ Contrastive learning is one form of self-supervised learning.

### Answer

**Correct: 1, 2, 4, 5**

### Explanation

Self-supervised learning creates supervision from the structure of the data.

For example:

$$
x \rightarrow \text{transformed/hidden information}
$$

and the model learns to predict or relate that information.

Contrastive learning is an important family of self-supervised methods.

---

## Q14 — Contrastive learning

Suppose an image $x$ is transformed twice:

$$
x_1 = t_1(x), \qquad x_2=t_2(x).
$$

The two transformed images are passed through an encoder.

**Which statements are generally true? Mark all that apply.**

- □ $x_1$ and $x_2$ can form a positive pair.
- □ The model is encouraged to produce similar representations for the positive pair.
- □ Other examples can be treated as negative examples.
- □ The goal is generally to make every pair of examples have identical embeddings.
- □ Data augmentation can define what information the representation should become invariant to.

### Answer

**Correct: 1, 2, 3, 5**

### Explanation

A central idea is:

$$
\operatorname{sim}(z_1,z_2) \uparrow
$$

for positive pairs.

For negatives:

$$
\operatorname{sim}(z_i,z_j) \downarrow.
$$

The augmentations are extremely important because they implicitly specify which changes should not substantially alter the representation.

---

## Q15 — Contrastive matrix

Consider four images:

$$
a,\quad a^*,\quad b,\quad b^*
$$

where $a^*$ is an augmentation of $a$, and $b^*$ is an augmentation of $b$.

The model uses a contrastive objective where positive pairs should have **low distance** and negative pairs should have **high distance**.

**Which distances should ideally be LOW? Mark all that apply.**

- □ $d(a,a^*)$
- □ $d(b,b^*)$
- □ $d(a,b)$
- □ $d(a,b^*)$
- □ $d(a^*,b^*)$

### Answer

**Correct:****$d(a,a^*)$****and****$d(b,b^*)$**

### Explanation

The positive pairs are:

$$
(a,a^*)
$$

and

$$
(b,b^*).
$$

Therefore:

$$
d(a,a^*) \rightarrow \text{LOW}
$$

$$
d(b,b^*) \rightarrow \text{LOW}.
$$

The cross-image pairs are negatives under the simplified setup:

$$
d(a,b),d(a,b^*),d(a^*,b),d(a^*,b^*)
\rightarrow \text{HIGH}.
$$

This is very similar to the matrix-style question in the real exam.

---

## Q16 — Why augmentation matters

**Why are augmentations important in contrastive learning? Mark all that apply.**

- □ They define transformations under which the representation should ideally remain stable.
- □ They can create positive pairs without requiring additional manual labels.
- □ Poorly chosen augmentations can remove information that is actually important for the task.
- □ They guarantee that the learned representation will be useful for every downstream task.
- □ They can encourage invariance to irrelevant visual changes.

### Answer

**Correct: 1, 2, 3, 5**

### Explanation

The choice of augmentation effectively communicates an assumption:

> "These two different views should represent the same underlying semantic content."

If that assumption is wrong, the model may learn an undesirable invariance.

For example, if colour is crucial for the downstream task, extremely aggressive colour transformations could be harmful.

---

## Q17 — Positive and negative pairs

**Which statements about positive and negative pairs in contrastive learning are true? Mark all that apply.**

- □ Positive pairs are generally encouraged to have similar representations.
- □ Negative pairs are generally encouraged to have distinguishable representations.
- □ Two augmented views of the same underlying example can form a positive pair.
- □ Every pair in a batch must necessarily be a positive pair.
- □ The exact definition of positive and negative pairs depends on the contrastive learning method.

### Answer

**Correct: 1, 2, 3, 5**

### Explanation

The exact loss can vary substantially between contrastive methods.

The general intuition is:

$$
\text{positive similarity} \uparrow
$$

and

$$
\text{negative similarity} \downarrow.
$$

---

## Q18 — Representation learning

**A useful representation learned through self-supervised learning should ideally:**

- □ encode information useful for downstream tasks;
- □ be invariant to transformations regarded as irrelevant;
- □ necessarily preserve every pixel-level detail;
- □ allow downstream models to extract useful task-specific information.

### Answer

**Correct: 1, 2, 4**

### Explanation

A representation is not necessarily supposed to preserve everything.

Instead, we want it to preserve **useful information** while becoming invariant to irrelevant variation.

This creates an important conceptual tension:

$$
\text{invariance}
\quad\text{vs.}\quad
\text{information preservation}.
$$

---

# Chapter 4 — Model Compression, Quantisation, Pruning and PEFT

## Q19 — Quantisation

**Which statements about quantisation are true? Mark all that apply.**

- □ Quantisation can represent values using fewer bits.
- □ FP32 → INT8 can substantially reduce weight storage.
- □ Quantisation necessarily removes entire neurons.
- □ Quantisation introduces numerical approximation.
- □ Lower precision can potentially reduce model accuracy.

### Answer

**Correct: 1, 2, 4, 5**

### Explanation

Quantisation changes the **numerical representation**.

For example:

$$
32\text{-bit FP32}
\rightarrow
8\text{-bit INT8}.
$$

It does not inherently remove neurons or parameters.

---

## Q20 — Memory calculation

A model contains $1$ billion parameters.

Ignoring overhead, which statements are true?

- □ FP32 requires approximately $4$ GB.
- □ FP16 requires approximately $2$ GB.
- □ INT8 requires approximately $1$ GB.
- □ INT4 requires approximately $0.5$ GB.
- □ INT8 requires more storage than FP16.

### Answer

**Correct: 1, 2, 3, 4**

### Explanation

Using:

$$
\text{memory}=
\frac{N\times\text{bits per parameter}}{8}.
$$

For $10^9$ parameters:

$$
FP32 = 10^9\times4 \approx 4\text{ GB}
$$

$$
FP16 = 10^9\times2 \approx 2\text{ GB}
$$

$$
INT8 = 10^9\times1 \approx 1\text{ GB}
$$

$$
INT4 = 10^9\times0.5 \approx 0.5\text{ GB}.
$$

---

## Q21 — Quantisation accuracy

**Why can aggressive quantisation reduce accuracy? Mark all that apply.**

- □ Quantisation introduces numerical error.
- □ Some weights/activations may be particularly sensitive to precision loss.
- □ All neural networks are mathematically invariant to numerical precision.
- □ Very low precision can make it difficult to represent important numerical differences.
- □ Quantisation always removes model layers.

### Answer

**Correct: 1, 2, 4**

### Explanation

Quantisation maps a continuous/high-precision value into a restricted set of representable values.

Conceptually:

$$
w \rightarrow Q(w).
$$

Therefore:

$$
Q(w)\neq w
$$

in general.

If the resulting error is sufficiently large in sensitive parts of the network, performance can degrade.

---

## Q22 — Pruning

**Which statements about pruning are true? Mark all that apply.**

- □ Pruning attempts to remove parameters or structures considered unnecessary.
- □ Magnitude pruning can remove weights with small absolute values.
- □ Pruning and quantisation are exactly the same operation.
- □ Structured pruning can remove channels, filters or other regular structures.
- □ Pruning can potentially reduce computation as well as model size.

### Answer

**Correct: 1, 2, 4, 5**

### Explanation

Pruning and quantisation exploit different forms of redundancy.

**Pruning:**

$$
W \rightarrow W_{\text{sparse}}
$$

by removing parameters/structures.

**Quantisation:**

$$
W \rightarrow Q(W)
$$

by representing values with lower precision.

---

## Q23 — Structured versus unstructured pruning

**Which statements correctly distinguish structured and unstructured pruning? Mark all that apply.**

- □ Unstructured pruning can remove individual weights.
- □ Structured pruning can remove entire channels.
- □ Structured pruning is generally easier for standard hardware to exploit.
- □ Unstructured sparsity necessarily produces proportional hardware speedup.
- □ A model with 90% unstructured sparsity is not automatically 10 times faster.

### Answer

**Correct: 1, 2, 3, 5**

### Explanation

This distinction is highly exam-worthy.

A dense matrix with many zeros may still be processed by hardware using dense operations.

By contrast, removing an entire channel can reduce the dimensions of subsequent operations.

Thus:

$$
\text{parameter reduction}
\neq
\text{guaranteed latency reduction}.
$$

---

## Q24 — LoRA

**Which statements about LoRA are true? Mark all that apply.**

- □ LoRA freezes the pretrained weights.
- □ LoRA learns a low-rank update.
- □ LoRA can substantially reduce the number of trainable parameters.
- □ LoRA requires the pretrained weight matrix itself to be low rank.
- □ LoRA is a parameter-efficient fine-tuning method.

### Answer

**Correct: 1, 2, 3, 5**

### Explanation

The central idea is:

$$
W' = W + \Delta W
$$

with LoRA approximating:

$$
\Delta W = BA.
$$

For

$$
A\in\mathbb R^{r\times d},
\qquad
B\in\mathbb R^{d\times r},
$$

the trainable parameter count is:

$$
2dr.
$$

The crucial distinction is:

> LoRA assumes the **adaptation/update** can be represented effectively in a low-dimensional space.

It does **not** require:

$$
\operatorname{rank}(W)\ll d.
$$

---

## Q25 — LoRA parameter count

A square weight matrix has dimension

$$
d\times d.
$$

A LoRA adapter uses rank $r$.

**Which statements are true? Mark all that apply.**

- □ Dense fine-tuning requires $d^2$ trainable parameters for this matrix.
- □ LoRA requires $2dr$ trainable parameters.
- □ If $r\ll d$, LoRA can require dramatically fewer trainable parameters.
- □ LoRA requires $r^2$ parameters.
- □ If $r=d$, the LoRA parameter count is $2d^2$.

### Answer

**Correct: 1, 2, 3, 5**

### Explanation

Dense:

$$
N_{\text{dense}}=d^2.
$$

LoRA:

$$
N_{\text{LoRA}}
=
dr+dr
=
2dr.
$$

The ratio is:

$$
\frac{N_{\text{LoRA}}}{N_{\text{dense}}}
=
\frac{2dr}{d^2}
=
\frac{2r}{d}.
$$

Therefore, when

$$
r\ll d,
$$

the trainable parameter fraction is small.

---

## Q26 — LoRA and optimizer state

**Why can LoRA reduce training memory? Mark all that apply.**

- □ Fewer parameters require gradients.
- □ Fewer trainable parameters require optimizer states.
- □ The frozen pretrained weights disappear from GPU memory.
- □ The forward pass still uses the pretrained model.
- □ The total memory reduction can therefore be smaller than the reduction in trainable parameter count.

### Answer

**Correct: 1, 2, 4, 5**

### Explanation

This is another distinction likely to matter in the exam.

Freezing reduces:

- trainable gradients;
- optimizer state.

But the base model still exists.

Thus:

$$
\text{trainable parameter reduction}
\neq
\text{complete memory reduction}.
$$

---

## Q27 — LoRA versus SVD

**Which statements correctly distinguish SVD-based compression from LoRA? Mark all that apply.**

- □ SVD can approximate an existing matrix using lower-rank factors.
- □ LoRA learns a low-rank update to a pretrained matrix.
- □ SVD and LoRA necessarily solve exactly the same problem.
- □ LoRA does not require the original pretrained matrix itself to be low rank.
- □ SVD can be used as a form of matrix compression.

### Answer

**Correct: 1, 2, 4, 5**

### Explanation

For an existing matrix:

$$
W\approx U_r\Sigma_rV_r^T
$$

is a low-rank approximation.

LoRA instead leaves $W$ intact and learns:

$$
W'=W+BA.
$$

Therefore:

> **SVD compresses the existing representation; LoRA learns a low-dimensional adaptation.**

---

## Q28 — LoRA initialization

Suppose LoRA uses

$$
W'=W+BA.
$$

At initialization:

$$
B=0.
$$

**Which statements are true?**

- □ Initially, $BA=0$.
- □ Initially, the adapted model can behave identically to the pretrained model, assuming no other changes.
- □ The LoRA parameters can subsequently learn a non-zero update.
- □ The model must initially produce random predictions.
- □ Increasing the rank is guaranteed to solve a poor LoRA implementation.

### Answer

**Correct: 1, 2, 3**

### Explanation

If

$$
B=0,
$$

then:

$$
BA=0.
$$

Therefore:

$$
W'=W.
$$

This is useful because the adapter can begin training from the pretrained model's behaviour rather than introducing a random perturbation.

---

# Integrated UAB-Style Questions

The following questions deliberately combine multiple chapters, because the real exam questions often test whether you can **connect concepts**.

---

## Q29 — Transfer + PEFT

A company has a pretrained model and wants to adapt it to several related tasks using limited GPU memory.

**Which statements are reasonable? Mark all that apply.**

- □ Freezing the pretrained model can reduce the number of trainable parameters.
- □ LoRA can store a small task-specific update while sharing the base model.
- □ Full fine-tuning is always preferable because it has more trainable parameters.
- □ Different tasks can potentially have different LoRA adapters attached to the same base model.
- □ The shared pretrained model must be duplicated completely for every task.

### Answer

**Correct: 1, 2, 4**

### Explanation

A very natural architecture is:

$$
\boxed{\text{shared base model}}
+
\boxed{\text{task-specific adapter}}.
$$

For $n$ tasks, this can avoid storing $n$ complete copies of the model.

---

## Q30 — Compression strategy

You have a large model that:

- fits poorly in device memory;
- has acceptable accuracy;
- runs too slowly;
- contains substantial redundancy.

**Which techniques could potentially help? Mark all that apply.**

- □ Quantisation
- □ Structured pruning
- □ LoRA alone
- □ Removing redundant parameters
- □ Increasing numerical precision from INT8 to FP32

### Answer

**Correct: 1, 2, 4**

### Explanation

Quantisation can reduce:

$$
\text{memory bandwidth and storage}.
$$

Structured pruning can reduce:

$$
\text{actual computational dimensions}.
$$

LoRA primarily addresses **adaptation efficiency**, not direct compression of an already-trained model.

Increasing precision generally works in the opposite direction.

---

## Q31 — Choosing a method

A student says:

> "I will use LoRA because I need to reduce inference latency."

**Which evaluation is most accurate?**

- □ Correct: LoRA is primarily designed to make inference faster.
- □ Correct: LoRA removes most of the pretrained computation.
- □ Incorrect: LoRA is primarily a parameter-efficient adaptation method; it does not automatically eliminate the base model's forward computation.
- □ Incorrect: LoRA cannot be used for fine-tuning.
- □ Correct: LoRA always reduces the number of layers.

### Answer

**Correct: 3**

### Explanation

This is a classic conceptual trap.

LoRA primarily reduces:

$$
\text{trainable parameters}
$$

and therefore training/storage costs associated with adaptation.

It does **not** mean:

$$
\text{90\% fewer trainable parameters}
\Rightarrow
\text{90\% fewer inference FLOPs}.
$$

---

## Q32 — Integrated scenario

You have only 500 labelled examples but 500,000 unlabelled examples for a target task.

A pretrained model is available.

**Which approaches could reasonably be considered? Mark all that apply.**

- □ Transfer learning
- □ Semi-supervised learning
- □ Self-supervised pretraining
- □ Pseudo-labeling
- □ Ignoring the unlabelled data because it cannot contain useful information

### Answer

**Correct: 1, 2, 3, 4**

### Explanation

Several techniques can potentially exploit the situation:

$$
\text{pretrained knowledge}
+
\text{small labelled set}
+
\text{large unlabelled set}.
$$

One possible pipeline is:

$$
\text{self-supervised representation learning}
\rightarrow
\text{transfer learning}
\rightarrow
\text{semi-supervised/pseudo-label training}.
$$

The exact strategy depends on the data and task.

---

## Q33 — Contrastive learning and transfer

A vision model is pretrained using contrastive learning. It is then fine-tuned on a classification task.

**Which statements are reasonable? Mark all that apply.**

- □ Contrastive pretraining can learn representations without manual class labels.
- □ The learned representation may provide a useful initialization for the downstream classification task.
- □ The contrastive pretraining objective must be identical to the downstream classification objective.
- □ Fine-tuning can adapt the representation to the downstream task.
- □ Contrastive learning guarantees perfect downstream classification.

### Answer

**Correct: 1, 2, 4**

### Explanation

The objectives can differ.

During contrastive learning:

$$
\text{learn useful representation}.
$$

During classification:

$$
\text{learn class-discriminative decision function}.
$$

This is a classic example of **representation transfer**.

---

## Q34 — The "more parameters" trap

A student claims:

> "Method A trains 100 million parameters and Method B trains only 5 million. Therefore Method A must always achieve better accuracy."

**Which statements are correct? Mark all that apply.**

- □ More trainable parameters can provide greater adaptation capacity.
- □ More parameters do not guarantee better validation accuracy.
- □ A lower-dimensional adaptation can be sufficient for the target task.
- □ The optimal choice depends on the task, data and computational constraints.
- □ Parameter count alone determines generalisation performance.

### Answer

**Correct: 1, 2, 3, 4**

### Explanation

The key idea is:

$$
\text{capacity} \neq \text{guaranteed performance}.
$$

LoRA may perform extremely well despite training far fewer parameters because the required task-specific change may have relatively low effective dimensionality.

---

## Q35 — Comprehensive scenario

You are given a pretrained model and four possible strategies:

| Strategy | Trainable parameters | Base model | Quantisation |
| --- | --- | --- | --- |
| A | All | Updated | FP32 |
| B | Small adapter | Frozen | FP32 |
| C | Small adapter | Frozen | INT8 |
| D | All | Updated | INT8 |

You have **very limited GPU memory**, want to adapt the model to many tasks, and want to maintain good accuracy.

**Which statements are most defensible? Mark all that apply.**

- □ B or C may be attractive because only a small task-specific component is trained.
- □ C potentially combines PEFT with quantisation.
- □ A is automatically the best because it has the largest number of trainable parameters.
- □ A shared base model plus task-specific adapters can reduce task-specific storage.
- □ The final choice should ideally be validated experimentally rather than inferred from parameter counts alone.

### Answer

**Correct: 1, 2, 4, 5**

### Explanation

This combines almost the entire Chapter 4 story.

A possible architecture is:

$$
\boxed{\text{INT8 shared base}}
+
\boxed{\text{task-specific LoRA}}.
$$

This can reduce:

- trainable parameters;
- optimizer state;
- task-specific storage;
- potentially base-model memory.

But the final system must still be experimentally evaluated for:

- accuracy;
- memory;
- training time;
- inference latency.

---

# High-Discrimination Questions

These are the questions I would pay particular attention to before the exam because they test **subtle conceptual distinctions**.

---

## Q36 — Which statement is WRONG?

**Which of the following statements is incorrect?**

- □ Freezing parameters reduces the number of parameters being optimised.
- □ Frozen parameters can still contribute to the forward pass.
- □ LoRA learns a low-rank update to pretrained weights.
- □ Quantisation reduces the numerical precision used to represent values.
- □ LoRA requires the pretrained weight matrix itself to have low rank.

### Answer

**Incorrect statement: 5**

### Why?

The low-rank assumption is about:

$$
\Delta W
$$

rather than necessarily about:

$$
W.
$$

That distinction is fundamental.

---

## Q37 — Which statement is WRONG?

**Which statement about contrastive learning is incorrect?**

- □ Positive pairs should generally have similar representations.
- □ Negative examples should generally be distinguishable from positive examples.
- □ Data augmentation can generate positive pairs.
- □ The model should map every image in the dataset to exactly the same embedding.
- □ The learned representation can subsequently be transferred to another task.

### Answer

**Incorrect statement: 4**

### Why?

If every example had exactly the same embedding, the representation would contain essentially no useful discriminative information.

Contrastive learning tries to achieve something closer to:

$$
\text{same semantic instance}
\rightarrow
\text{similar representation}
$$

while retaining enough structure to distinguish different examples.

---

## Q38 — Which statement is WRONG?

**Which statement about semi-supervised learning is incorrect?**

- □ It can exploit unlabelled data.
- □ It can be useful when labels are expensive.
- □ Pseudo-labeling is one possible technique.
- □ The quality of unlabelled data and the assumptions made by the method matter.
- □ The presence of unlabelled data guarantees improved performance.

### Answer

**Incorrect statement: 5**

### Why?

More data is not automatically useful.

Bad assumptions, distribution mismatch, or incorrect pseudo-labels can actually hurt performance.

---

## Q39 — Which statement is WRONG?

**Which statement about model compression is incorrect?**

- □ Quantisation changes numerical precision.
- □ Pruning removes parameters or structures.
- □ Structured pruning can be more hardware-friendly than arbitrary sparsity.
- □ Compression always preserves exactly the original accuracy.
- □ Different compression techniques can potentially be combined.

### Answer

**Incorrect statement: 4**

### Why?

Compression generally involves a trade-off:

$$
\text{efficiency}
\leftrightarrow
\text{accuracy}.
$$

The goal is often to find a point where efficiency improves while accuracy degradation remains acceptable.

---

## Q40 — The most important distinction

Consider:

$$
W'=W+BA.
$$

**Which statements are true? Mark all that apply.**

- □ $W$ can remain frozen.
- □ $BA$ represents the learned task-specific update.
- □ $A$ and $B$ can have far fewer parameters than $W$.
- □ $W$ must itself be low rank.
- □ Increasing $r$ increases the capacity of the adapter.
- □ Increasing $r$ guarantees higher validation accuracy.

### Answer

**Correct: 1, 2, 3, 5**

### Explanation

For:

$$
A\in\mathbb R^{r\times d},
\qquad
B\in\mathbb R^{d\times r},
$$

we have:

$$
N_{\text{LoRA}}=2dr.
$$

Increasing $r$ increases the number of trainable parameters and therefore the possible adaptation capacity.

But:

$$
r\uparrow
\not\Rightarrow
\text{accuracy}\uparrow
$$

as a mathematical guarantee.

---

# Final Exam-Style Rapid Review

If you can confidently answer the following **without looking at your notes**, you are covering the most important conceptual distinctions from Chapters 1–4.

### Transfer learning

- **Q:** Does transfer learning guarantee improvement?

**A:** No. Negative transfer is possible.

- **Q:** Must source and target tasks have identical labels?

**A:** No.

- **Q:** Can frozen layers still perform computation?

**A:** Yes.

- **Q:** Why use a smaller learning rate during fine-tuning?

**A:** To reduce destructive changes to useful pretrained representations.

### Curriculum / semi-supervised learning

- **Q:** What does self-paced learning use to determine example difficulty?

**A:** Potentially the model's own loss/confidence/error.

- **Q:** What happens as the self-paced threshold $\lambda$ increases?

**A:** More difficult examples can become eligible.

- **Q:** What is pseudo-labeling?

**A:** Using model predictions as approximate labels for unlabelled examples.

- **Q:** What is the major danger of pseudo-labeling?

**A:** Confirmation/error reinforcement.

### Self-supervised / contrastive learning

- **Q:** What is a positive pair?

**A:** Two views/examples intended to represent the same underlying semantic content.

- **Q:** Positive pair distance?

**A:** Ideally low.

- **Q:** Negative pair distance?

**A:** Ideally high.

- **Q:** Why are augmentations important?

**A:** They define transformations to which the representation should ideally be invariant.

### Compression / PEFT

- **Q:** Quantisation vs pruning?

**A:** Quantisation reduces numerical precision; pruning removes parameters/structures.

- **Q:** Structured vs unstructured pruning?

**A:** Structured removes regular structures such as channels; unstructured can remove individual weights.

- **Q:** Why doesn't 90% sparsity guarantee 10× speedup?

**A:** Hardware/software may not efficiently exploit arbitrary sparsity.

- **Q:** What does LoRA train?

**A:** A low-rank update.

$$
\boxed{W'=W+BA}
$$

- **Q:** LoRA parameter count for $d\times d$ matrix?

$$
\boxed{2dr}
$$

- **Q:** Does LoRA require $W$ to be low rank?

**A:** **No.** The low-rank assumption concerns the adaptation $\Delta W$.

- **Q:** Why does LoRA reduce training memory?

**A:** Fewer trainable parameters mean fewer gradients and optimizer states.

- **Q:** Does LoRA automatically make inference 10× faster?

**A:** **No.**

- **Q:** What is the central SVD vs LoRA distinction?

$$
\boxed{\text{SVD: compress/approximate }W}
$$

$$
\boxed{\text{LoRA: learn a low-rank }\Delta W}
$$

## The 10 traps I would memorise

1. **Frozen ≠ removed.**
2. **Fewer trainable parameters ≠ proportionally fewer FLOPs.**
3. **LoRA low-rank update ≠ low-rank pretrained matrix.**
4. **Quantisation ≠ pruning.**
5. **Pruning ≠ guaranteed speedup.**
6. **Transfer learning ≠ guaranteed improvement.**
7. **Unlabelled data ≠ automatically useful data.**
8. **Pseudo-labels ≠ guaranteed-correct labels.**
9. **Contrastive positive pairs ≠ all pairs should be similar.**
10. **More trainable parameters ≠ guaranteed higher accuracy.**

These are particularly aligned with the **reasoning-oriented style of the actual UAB exam** you provided: the examiner can turn almost every one of these distinctions into a multi-select question where several options look superficially correct.
