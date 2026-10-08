Absolutely. Below are **comprehensive model solutions for all three UAB-style mock exams**. I’ve written them as an **exam answer key**, with explanations of why the correct answers are correct and why the tempting alternatives are wrong where that is useful.

The solutions follow the same principle as the real exam: **understanding the mechanism matters more than reproducing memorised definitions**.

---

:::writing{variant="document" id="38264" title="Advanced Machine Learning — Comprehensive Solutions to Mock Exam 1"}

# Advanced Machine Learning

## Comprehensive Solutions — Mock Exam 1

**Total:** 50 points

---

# Question 1 — Transfer Learning and Domain Adaptation

### 5 points

### Correct answers

- ☑ Transfer learning can be useful when the target task has less labelled data than the source task.
- ☐ Fine-tuning always requires changing the architecture of the pretrained model.
- ☑ Transfer learning is generally more useful when the source and target problems share useful representations.
- ☑ Domain adaptation can be useful when the source and target tasks are related but their data distributions differ.
- ☐ A pretrained model can never perform worse on the target task than a randomly initialised model.

### Explanation

**Statement 1 — True.**

Transfer learning is particularly useful when the target dataset is small. A model pretrained on a large source dataset may already have learned useful features, so the target task does not need to learn everything from random initialisation.

For example, an image model pretrained on millions of images may already understand edges, textures, shapes, and object-level structures.

---

**Statement 2 — False.**

Fine-tuning does not necessarily require changing the architecture.

Typically, the pretrained architecture is retained and its parameters are further trained on the target task. The output layer may need to change if the target task has a different number of classes, but this is not a requirement of fine-tuning itself.

---

**Statement 3 — True.**

Transfer works best when knowledge learned on the source task is relevant to the target task.

For example, transferring from general image recognition to medical-image classification may be useful because lower-level visual features can transfer even though the target classes are different.

---

**Statement 4 — True.**

Domain adaptation addresses situations where there is a difference between the source and target data distributions.

Conceptually:

$$
P_{\text{source}}(x) \neq P_{\text{target}}(x)
$$

while the source and target tasks remain sufficiently related that knowledge can be transferred.

---

**Statement 5 — False.**

Transfer learning can produce **negative transfer**.

If the source task/domain is sufficiently different from the target, the pretrained representation may be poorly suited to the target task and can sometimes make performance worse than learning from scratch.

### Key exam point

A strong answer should distinguish:

$$
\text{transfer learning}
$$

from

$$
\text{domain adaptation}.
$$

Transfer learning is the broader idea of reusing knowledge, while domain adaptation specifically focuses on transferring across differing data domains/distributions.

---

# Question 2 — Fine-Tuning Behaviour

### 5 points

Given:

| Model | Training accuracy | Validation accuracy |
| --- | --- | --- |
| From scratch | 96% | 71% |
| Pretrained + frozen backbone | 84% | 79% |
| Pretrained + fine-tuning | 91% | 83% |

## (a) Why is the student's conclusion incomplete?

The student's statement was:

> "Fine-tuning is better because it always increases training accuracy."

This is incorrect.

The important quantity for judging generalisation is not simply training accuracy.

The relevant comparison is:

$$
\text{training performance}
\quad\text{vs.}\quad
\text{validation/test performance}.
$$

Here, fine-tuning gives:

$$
91\% \text{ training},\qquad 83\% \text{ validation}.
$$

The frozen pretrained model gives:

$$
84\% \text{ training},\qquad 79\% \text{ validation}.
$$

So fine-tuning improves validation performance, which is stronger evidence that adaptation is useful.

The model trained from scratch achieves the **highest training accuracy** but the **worst validation accuracy**.

This suggests that it has learned the training set particularly well but generalises poorly.

---

## (b) Why can the frozen pretrained model outperform the model trained from scratch?

The pretrained model has already learned useful representations from a much larger dataset.

Even though the backbone is frozen, those representations may capture:

- edges;
- textures;
- shapes;
- visual patterns;
- higher-level semantic features.

The target task therefore only needs to learn how to use those existing representations.

By contrast, the model trained from scratch must learn useful representations using only the target training data.

Because the target dataset may be relatively small, this can lead to poorer generalisation.

### Key exam point

The important conceptual relationship is:

$$
\text{pretraining}
\rightarrow
\text{useful representation}
\rightarrow
\text{less target data required}.
$$

---

# Question 3 — Self-Supervised Learning

### 4 points

## (a)

This is **self-supervised learning**, specifically a contrastive/self-supervised representation-learning setup.

The model receives no human-provided class labels.

Instead, the training signal is constructed from the data itself.

---

## (b)

The augmentations create different views of the **same underlying example**.

For example:

$$
x \rightarrow x_1
$$

and

$$
x \rightarrow x_2.
$$

The model is encouraged to produce similar representations:

$$
z_1 \approx z_2.
$$

This teaches the model to learn features that are invariant to transformations that should not change the semantic identity of the example.

For images, these transformations might include:

- cropping;
- colour changes;
- rotations;
- noise;
- flipping.

The augmentation therefore helps define what information the representation should consider important.

### Key exam point

The augmentation is not merely data expansion.

It provides a **learning signal about invariance**.

---

# Question 4 — Contrastive Learning

### 5 points

The desired distances are:

| Pair | Desired distance |
| --- | --- |
| $a,a^*$ | **LOW** |
| $b,b^*$ | **LOW** |
| $a,b$ | **HIGH** |
| $a,b^*$ | **HIGH** |
| $a^*,b$ | **HIGH** |
| $a^*,b^*$ | **HIGH** |

The reason is that:

$$
a \leftrightarrow a^*
$$

are two views of the same underlying image, while

$$
b \leftrightarrow b^*
$$

are two views of another image.

Thus:

$$
d(z_a,z_{a^*}) \rightarrow \text{LOW}
$$

and

$$
d(z_b,z_{b^*}) \rightarrow \text{LOW}.
$$

The remaining pairs represent different underlying examples and are therefore treated as negative pairs.

---

## Why not make all embeddings close?

If the objective simply encouraged:

$$
z_i \approx z_j
$$

for every pair, the model could solve the problem by mapping every image to approximately the same representation:

$$
z_1=z_2=\cdots=z_n.
$$

This is a **collapsed representation**.

It minimises the similarity objective but contains almost no information about the input.

Contrastive learning therefore needs some mechanism encouraging different examples to remain distinguishable.

### Key exam point

A useful contrastive objective needs both:

- **invariance** between positive pairs;
- **discrimination** between negative/different examples.

---

# Question 5 — Model Compression and Quantisation

### 6 points

The student's claim is:

> "INT8 quantisation makes a neural network four times smaller because FP32 uses 32 bits and INT8 uses 8 bits. Therefore, it will always run four times faster."

## (a)

The storage calculation is approximately correct for the **weights alone**, assuming no additional overhead.

FP32 uses:

$$
32\text{ bits/parameter}
$$

while INT8 uses:

$$
8\text{ bits/parameter}.
$$

Therefore:

$$
\frac{32}{8}=4.
$$

So the theoretical weight-storage requirement is approximately reduced by a factor of 4.

---

## (b)

Memory reduction does not imply a four-times inference-speed improvement.

Actual latency depends on many factors, including:

- hardware support for INT8;
- memory bandwidth;
- arithmetic throughput;
- kernel implementations;
- communication overhead;
- activation computation;
- layers that remain at higher precision;
- data movement.

If the hardware has highly optimised INT8 operations, speedup can be substantial.

If it does not, the reduction in weight size may produce little speed improvement.

---

## (c)

Quantisation introduces numerical approximation.

For example, many FP32 values must be represented using a much smaller set of INT8 values.

This introduces **quantisation error**.

If the model is sensitive to these numerical changes, predictions can change and accuracy can decrease.

### Key exam point

Separate:

$$
\text{model size}
$$

from

$$
\text{inference latency}.
$$

They are related but not equivalent.

---

# Question 6 — LoRA

### 7 points

Given:

$$
W\in\mathbb{R}^{d\times d}
$$

and

$$
W'=W+BA
$$

where

$$
A\in\mathbb{R}^{r\times d},
\qquad
B\in\mathbb{R}^{d\times r}.
$$

## (a)

Ordinary fine-tuning directly learns:

$$
\Delta W\in\mathbb{R}^{d\times d}.
$$

Therefore:

$$
N_{\text{dense}}=d^2.
$$

---

## (b)

LoRA learns $A$ and $B$.

The number of parameters in $A$ is:

$$
rd.
$$

The number of parameters in $B$ is:

$$
dr.
$$

Therefore:

$$
N_{\text{LoRA}}
=
rd+dr
=
2dr.
$$

---

## (c)

The ratio is:

$$
\frac{N_{\text{LoRA}}}{N_{\text{dense}}}
=
\frac{2dr}{d^2}
=
\frac{2r}{d}.
$$

When

$$
r\ll d,
$$

we have:

$$
\frac{2r}{d}\ll1.
$$

Therefore, LoRA trains far fewer parameters than dense fine-tuning.

---

## (d)

No.

LoRA does **not** require the pretrained matrix $W$ to be low rank.

The assumption is instead that the **task-specific update**

$$
\Delta W
$$

can be effectively represented or approximated by a low-rank matrix:

$$
\Delta W\approx BA.
$$

The pretrained matrix itself can remain full rank.

### Key exam distinction

This is one of the most important distinctions in the topic:

$$
\boxed{\text{LoRA: low-rank update}}
$$

is not the same as

$$
\boxed{\text{low-rank compression of }W}.
$$

---

# Question 7 — Knowledge Distillation

### 6 points

### Correct statements

- ☑ The student can learn from both ground-truth labels and teacher predictions.
- ☐ The teacher must have exactly the same architecture as the student.
- ☑ Soft teacher predictions can provide information about relationships between classes.
- ☐ The student must contain at least as many parameters as the teacher.
- ☑ A student can inherit undesirable biases or weaknesses from its teacher.
- ☑ Temperature can be used to soften the teacher's output distribution.

---

## Why statement 1 is true

Knowledge distillation often combines two losses:

$$
L
=
\alpha L_{\text{hard}}
+
(1-\alpha)L_{\text{distill}}.
$$

The first uses ground-truth labels.

The second encourages the student to reproduce information from the teacher.

---

## Why statement 2 is false

The teacher and student can have different architectures.

In fact, a major purpose of distillation is often to train a **smaller student** from a larger teacher.

---

## Why statement 3 is true

Suppose an image is labelled "cat".

A hard label gives:

$$
[0,1,0,\ldots].
$$

A teacher might instead produce something like:

$$
[0.01,0.85,0.10,0.04,\ldots].
$$

The soft probabilities contain information about which alternative classes the teacher considers similar.

This information is sometimes called **dark knowledge**.

---

## Why statement 4 is false

The student is often deliberately smaller.

The teacher could have:

$$
100\text{M parameters}
$$

while the student has:

$$
10\text{M parameters}.
$$

---

## Why statement 5 is true

The student learns from the teacher's behaviour.

Therefore, teacher biases, systematic mistakes, or vulnerabilities can potentially be transferred.

---

## Why statement 6 is true

Temperature $T$ can soften logits:

$$
p_i
=
\frac{\exp(z_i/T)}
{\sum_j\exp(z_j/T)}.
$$

A larger $T$ produces a softer probability distribution.

---

# Question 8 — Algorithm Interpretation

### 12 points

The algorithm:

1. calculates example losses;
2. selects examples with

$$
L_i<\lambda;
$$

3. updates using only those examples;
4. gradually increases $\lambda$.

## (a)

This is **self-paced learning**, closely related to curriculum learning.

More specifically, the model itself provides the difficulty signal through its current loss.

---

## (b)

Low-loss examples are treated as easier examples.

Training initially on examples the model can already handle can make optimisation easier and provide relatively clean/useful gradient signals.

The model learns simpler patterns first.

---

## (c)

As training progresses, $\lambda$ increases.

Therefore:

$$
L_i<\lambda
$$

becomes easier to satisfy.

More difficult examples are gradually admitted into training.

Thus the curriculum progresses from easier to harder examples.

---

## (d)

If $\lambda$ is extremely high from the beginning, almost every example will be selected.

The curriculum effectively disappears.

The method becomes much closer to ordinary training on the complete dataset.

A high threshold can also introduce difficult/noisy examples too early, potentially making optimisation less stable.

### Key exam interpretation

The algorithm is essentially implementing:

$$
\text{easy examples first}
\rightarrow
\text{harder examples later}.
$$

The important conceptual distinction is that **self-paced learning uses the model's own behaviour to determine difficulty**.

---

# Mock Exam 1 — High-Level Answer Summary

| Question | Core answer |
| --- | --- |
| Q1 | Transfer useful with related source/target; domain adaptation handles distribution shift; negative transfer possible |
| Q2 | Validation performance matters more than training accuracy for generalisation |
| Q3 | Self-supervised/contrastive representation learning |
| Q4 | Same-example pairs LOW; different-example pairs HIGH |
| Q5 | INT8 gives roughly $4\times$ weight-storage reduction vs FP32, but not necessarily $4\times$ speed |
| Q6 | Dense: $d^2$; LoRA: $2dr$ |
| Q7 | KD uses teacher soft targets; student can be smaller/different; temperature softens outputs |
| Q8 | Self-paced learning: select low-loss examples and progressively increase difficulty |

:::

---

:::writing{variant="document" id="51973" title="Advanced Machine Learning — Comprehensive Solutions to Mock Exam 2"}

# Advanced Machine Learning

## Comprehensive Solutions — Mock Exam 2

**Total:** 50 points

---

# Question 1 — Transfer Learning

### 4 points

## (a)

The student's reasoning is incorrect.

Even when the target classes differ from the source classes, the pretrained model may contain useful **general representations**.

For an image model, early and intermediate layers may learn:

- edges;
- textures;
- shapes;
- spatial patterns;
- generic visual features.

These features can transfer to a different target classification problem.

Therefore:

$$
\text{different classes}
\neq
\text{no transferable knowledge}.
$$

---

## (b)

Transfer can be harmful through **negative transfer**.

This can occur when:

- the source and target domains are very different;
- the source task encourages irrelevant features;
- the pretrained representation contains strong task-specific biases;
- the target data distribution is substantially different.

In such cases, training from scratch may outperform transfer learning.

---

# Question 2 — Domain Shift

### 5 points

### Correct answers

- ☑ The deployment distribution may differ from the training distribution.
- ☐ The model necessarily has too few parameters.
- ☑ The validation set may not have captured the relevant distribution shift.
- ☑ Domain adaptation could potentially improve performance.
- ☐ The high training accuracy proves the model has learned features that generalise to deployment.

---

## Explanation

The deployment environment is geographically different.

Therefore it is plausible that:

$$
P_{\text{deployment}}(x)
\neq
P_{\text{training}}(x).
$$

The validation set may still resemble the training distribution, producing:

$$
\text{validation accuracy}=94\%.
$$

But deployment may be substantially different:

$$
\text{deployment accuracy}=63\%.
$$

This is a form of **distribution shift/domain shift**.

Increasing model capacity is not guaranteed to solve this problem. The problem may be that the model has learned features that do not transfer to the deployment distribution.

### Strong exam answer

> High validation accuracy only tells us that the model performs well on the validation distribution. If deployment comes from a different distribution, validation performance may not accurately estimate deployment performance.

---

# Question 3 — Semi-Supervised Learning

### 5 points

## (a)

Ignoring the unlabelled data wastes potentially useful information.

The 200,000 unlabelled examples can provide information about the structure of the target data distribution.

This can be particularly valuable when labels are expensive but raw examples are abundant.

---

## (b)

One possible approach is **pseudo-labelling**.

A model can first be trained on the labelled data.

It can then predict labels for unlabelled examples:

$$
\hat y=f_\theta(x).
$$

Examples for which the model is sufficiently confident can be incorporated into subsequent training.

For example:

$$
\max_y p_\theta(y|x)>\tau
$$

could be used as a confidence criterion.

The model can then train using both:

- genuine labelled examples;
- high-confidence pseudo-labelled examples.

Other valid answers could include consistency regularisation or teacher-student/mean-teacher methods.

### Important caveat

Incorrect pseudo-labels can reinforce model errors.

Therefore, confidence thresholds or other mechanisms are often used to reduce confirmation bias.

---

# Question 4 — Self-Distillation

### 6 points

## (a)

The student can outperform the teacher even when the architectures are identical because **distillation changes the training signal**, not necessarily the model capacity.

The teacher provides soft predictions:

$$
p_T(y|x).
$$

The student can learn from both:

$$
y_{\text{true}}
$$

and

$$
p_T(y|x).
$$

The teacher's soft outputs contain information about relationships between classes and can act as a form of regularisation.

The student may also benefit from:

- a different optimisation trajectory;
- improved regularisation;
- smoother targets;
- reduced overfitting.

Therefore identical architecture does not imply identical learned function.

---

## (b)

No.

There is no guarantee that self-distillation improves performance.

The student can:

- reproduce the teacher;
- improve slightly;
- remain approximately equal;
- perform worse.

The result is empirical.

### Key exam point

**Same architecture does not mean same optimisation outcome.**

---

# Question 5 — LoRA Implementation

### 6 points

The student uses:

$$
W'=W+BA
$$

with both $A$ and $B$ randomly initialised.

## (a)

The likely issue is that:

$$
BA\neq0
$$

at initialisation.

Therefore the effective weight matrix becomes:

$$
W'=W+\Delta W
$$

where $\Delta W$ is already non-zero.

The model therefore starts from a significantly different function from the pretrained model.

---

## (b)

Ideally, the initial LoRA update should satisfy:

$$
BA=0.
$$

A common strategy is to initialise one LoRA factor to zero while the other is randomly initialised.

For example:

$$
B=0
$$

initially.

Then:

$$
BA=0.
$$

---

## (c)

This allows the initial adapted model to behave approximately like the original pretrained model:

$$
W'=W.
$$

Training can then gradually introduce a task-specific update.

This is desirable because the pretrained model already provides a strong starting point.

### Key exam point

The important idea is **identity-preserving initialisation**:

$$
\boxed{\Delta W_0\approx0}.
$$

---

# Question 6 — Model Merging and Task Arithmetic

### 6 points

Given:

$$
\theta_0
$$

as the pretrained model,

$$
\theta_A
$$

as task A,

and

$$
\theta_B
$$

as task B.

---

## (a)

The task vectors are:

$$
\Delta_A=\theta_A-\theta_0
$$

and

$$
\Delta_B=\theta_B-\theta_0.
$$

They represent the parameter changes associated with adapting the base model to each task.

Conceptually:

$$
\theta_0
\xrightarrow{\Delta_A}
\theta_A.
$$

---

## (b)

The coefficients $\alpha$ and $\beta$ control how strongly each task vector contributes.

The merged model is:

$$
\theta_{\text{merged}}
=
\theta_0+\alpha\Delta_A+\beta\Delta_B.
$$

For example:

- increasing $\alpha$ emphasises task A;
- increasing $\beta$ emphasises task B.

---

## (c)

Task vectors may interfere with each other.

For example:

$$
\Delta_A
$$

and

$$
\Delta_B
$$

may contain parameter changes that are compatible, but they may also point in conflicting directions.

Adding them may therefore produce:

- interference;
- loss of task performance;
- unexpected behaviour;
- degradation relative to the original models.

This is why model merging is not guaranteed to work simply because the arithmetic is straightforward.

---

# Question 7 — Meta-Learning

### 6 points

Given:

$$
\theta_M
\leftarrow
\theta_M+\beta(\theta_j-\theta_M).
$$

## (a)

$\theta_M$ represents the **shared/meta-model parameters**.

They are intended to become a useful starting point for adaptation to new tasks.

---

## (b)

$\theta_j$ represents the parameters obtained after adapting the model to task $j$.

The model starts from the shared parameters and performs task-specific learning.

---

## (c)

If:

$$
\beta=1,
$$

then:

$$
\theta_M
\leftarrow
\theta_M+(\theta_j-\theta_M)
=
\theta_j.
$$

Therefore the meta-model becomes exactly the task-adapted model.

This is a very large update toward that task.

If $\beta$ is very small, then the meta-model moves only slightly toward the task-specific parameters.

Thus $\beta$ controls the interpolation:

$$
\theta_M'
=
(1-\beta)\theta_M+\beta\theta_j.
$$

---

# Question 8 — Ensemble Reasoning

### 12 points

Each network has approximately 80% accuracy, but averaging gives 86%.

## (a)

Individual models make different prediction errors.

Suppose model 1 is wrong on one subset of examples and model 2 is wrong on another subset.

When predictions are averaged, some individual errors can cancel.

The ensemble can therefore produce a more stable prediction.

For regression, this can be seen directly through variance reduction.

If predictions have variance $\sigma^2$ and are approximately independent, averaging $n$ predictions gives variance approximately:

$$
\frac{\sigma^2}{n}.
$$

The exact behaviour for classification is more complicated, but the same basic intuition applies: averaging can reduce variance and exploit complementary errors.

---

## (b)

Diversity is important.

If all five networks make exactly the same predictions and errors, averaging does not provide much benefit.

The ensemble is most useful when models are:

- individually competent;
- but make sufficiently different errors.

This is why different initialisations, data subsets, architectures, augmentation strategies, or training procedures can improve ensemble diversity.

---

## (c)

Yes.

The basic averaging idea also applies to regression.

Suppose we have predictions:

$$
\hat y_1,\hat y_2,\ldots,\hat y_n.
$$

We can construct:

$$
\hat y_{\text{ensemble}}
=
\frac1n\sum_{i=1}^{n}\hat y_i.
$$

This can reduce prediction variance.

Decision trees can also be combined, which is the basic intuition behind random forests and related ensemble methods.

---

## (d)

No.

Choosing the single best model does not exploit the complementary information contained in the other models.

For example:

- Model A may be best on one subset.
- Model B may be best on another subset.

The ensemble can combine them.

Validation-set selection can also overfit if many candidate models are compared.

### Key exam point

The fundamental ensemble idea is:

$$
\boxed{\text{multiple reasonably good and diverse models}}
$$

can outperform

$$
\boxed{\text{one individually strongest model}}.
$$

---

# Mock Exam 2 — High-Level Answer Summary

| Question | Core answer |
| --- | --- |
| Q1 | Different target classes do not prevent useful representation transfer; negative transfer possible |
| Q2 | Deployment shift can explain validation/deployment gap |
| Q3 | Unlabelled data can support pseudo-labelling, consistency training, etc. |
| Q4 | Self-distillation changes the training signal; same architecture does not imply same performance |
| Q5 | LoRA should start with approximately zero update |
| Q6 | Task vectors represent deviations from a shared base; merging can suffer interference |
| Q7 | Reptile-style interpolation toward task-adapted parameters |
| Q8 | Ensembles reduce variance and exploit complementary errors |

:::

---

:::writing{variant="document" id="64731" title="Advanced Machine Learning — Comprehensive Solutions to Mock Exam 3"}

# Advanced Machine Learning

## Comprehensive Solutions — Mock Exam 3

**Total:** 50 points

This paper is deliberately the most difficult. Several questions test whether the student can distinguish **parameter efficiency, memory efficiency, computational efficiency, and generalisation** rather than treating them as equivalent.

---

# Question 1 — Deployment Generalisation

### 4 points

### Correct answers

- ☑ The deployment distribution may differ from the training distribution.
- ☐ The model necessarily has too few parameters.
- ☑ The validation set may not have captured the relevant distribution shift.
- ☑ Domain adaptation could potentially improve performance.
- ☐ Increasing model capacity is guaranteed to solve the problem.

---

## Explanation

The deployment data comes from a different geographic region.

Therefore:

$$
P_{\text{deployment}}(x)
\neq
P_{\text{training}}(x)
$$

is plausible.

The validation set may have been sampled from the original training distribution and therefore may not reproduce the deployment conditions.

Thus:

$$
\text{validation accuracy}=96\%
$$

does not necessarily imply:

$$
\text{deployment accuracy}\approx96\%.
$$

---

## Why insufficient capacity is not necessarily the explanation

The model achieves:

$$
98\%
$$

training accuracy.

This suggests that it has sufficient capacity to fit the training data.

The problem is more likely related to generalisation under distribution shift than simple underfitting.

---

## Why increasing capacity is not guaranteed to help

More capacity can sometimes make the model fit the training distribution even more closely without making it robust to the changed deployment distribution.

Domain adaptation, robust training, additional representative data, or distribution-aware evaluation may be more appropriate.

### Strong exam statement

> Validation performance estimates generalisation to the validation distribution, not necessarily to an unseen deployment distribution that may differ systematically.

---

# Question 2 — Compression

### 5 points

The sequence is:

$$
\text{FP32}
\rightarrow
\text{INT8}
\rightarrow
\text{Pruned}
\rightarrow
\text{INT4}.
$$

## (a)

Quantisation exploits **numerical redundancy/precision**.

Instead of representing weights using 32-bit floating point numbers, fewer bits are used:

$$
32\rightarrow8\rightarrow4.
$$

Pruning exploits **parameter redundancy** by removing parameters that are considered unnecessary or less important.

Thus:

$$
\boxed{\text{Quantisation: fewer bits}}
$$

$$
\boxed{\text{Pruning: fewer retained parameters}}
$$

---

## (b)

An INT4 model can theoretically require less weight storage, but latency depends on hardware.

An INT8 implementation may have highly optimised hardware support, while INT4 sparse computation may have:

- poor hardware utilisation;
- expensive sparse indexing;
- conversion overhead;
- unsupported kernels;
- memory-access inefficiencies.

Therefore a smaller theoretical model can still have worse practical latency.

---

## (c)

Structured pruning removes regular structures such as:

- channels;
- filters;
- neurons;
- blocks.

The resulting smaller dense computation can often be executed efficiently using existing hardware.

Unstructured pruning may produce a matrix containing many zeros but with an irregular pattern.

The theoretical number of non-zero parameters can be greatly reduced without giving proportional hardware speedup.

---

# Question 3 — PEFT and Memory

### 6 points

The claim is:

> "Only 1% of the parameters are trainable, therefore GPU memory usage during training should also fall by 99%."

This is **incomplete**.

## (a)

The full pretrained model generally still needs to participate in the forward pass.

Therefore its weights must still be available.

Furthermore, GPU memory can contain:

- model weights;
- activations;
- gradients;
- optimizer states;
- temporary buffers.

Reducing trainable parameters primarily reduces the memory associated with the **trainable state**, not necessarily the entire memory footprint.

---

## (b)

Frozen parameters generally do not require:

- gradients;
- optimizer state for updating those parameters.

For example, in Adam, trainable parameters normally have additional first- and second-moment states.

If a parameter is frozen, these optimizer states do not need to be maintained for it.

Thus PEFT can produce a substantial training-memory reduction.

---

## (c)

The pretrained weights are still needed because LoRA modifies the forward computation:

$$
W'
=
W+BA.
$$

The output depends on both:

$$
W
$$

and

$$
BA.
$$

The base model therefore remains part of the computation.

### Critical distinction

$$
\boxed{\text{few trainable parameters}}
$$

does not mean

$$
\boxed{\text{few total parameters}}.
$$

---

# Question 4 — LoRA Rank

### 5 points

Given:

$$
\Delta W=BA.
$$

## (a)

Increasing $r$ increases the capacity of the low-rank update.

The LoRA parameter count is:

$$
N_{\text{LoRA}}=2dr.
$$

Therefore increasing $r$ increases the number of trainable parameters linearly.

A higher rank allows a richer update:

$$
r\uparrow
\quad\Rightarrow\quad
\text{greater adaptation capacity}.
$$

---

## (b)

Higher capacity does not guarantee better validation performance.

A larger rank may:

- overfit limited target data;
- increase optimisation cost;
- provide little additional useful capacity;
- capture noise rather than useful task information.

Therefore rank must be treated as a hyperparameter.

---

## (c)

A smaller rank means:

$$
N_{\text{LoRA}}=2dr
$$

is smaller.

This reduces:

- adapter storage;
- trainable parameters;
- gradient memory;
- optimizer state.

It may also make it easier to maintain many task-specific adapters.

---

# Question 5 — Contrastive Learning Failure

### 6 points

The student uses:

> "Make all embeddings in a batch as similar as possible."

The result is:

$$
z_1\approx z_2\approx\cdots\approx z_n.
$$

## (a)

This is **representation collapse**.

The model has found a trivial solution that satisfies the similarity objective.

---

## (b)

The representation contains almost no information about the input.

If every image produces approximately:

$$
z_i=z
$$

then the embedding cannot distinguish between different images or semantic categories.

---

## (c)

Contrastive learning normally distinguishes positive and negative relationships.

For positive pairs:

$$
d(z_i,z_i^+)\rightarrow\text{LOW}.
$$

For negative/different examples:

$$
d(z_i,z_j)\rightarrow\text{HIGH}.
$$

The negative component prevents the trivial solution where every representation is identical.

### Important nuance

Different modern self-supervised methods avoid collapse in different ways. Therefore, a good answer does not need to claim that **every** self-supervised method literally uses negative pairs.

For the specific objective in this question, however, the missing discrimination/negative component is the key issue.

---

# Question 6 — Curriculum Learning

### 5 points

## (a)

This implements **curriculum learning**, specifically a difficulty-based curriculum.

The training examples are ordered according to an estimated difficulty.

---

## (b)

Starting with easier examples can simplify optimisation.

The model can first learn relatively simple patterns before encountering more difficult examples.

This can provide a progression:

$$
\text{simple patterns}
\rightarrow
\text{more complex patterns}.
$$

The intuition is analogous to learning simple concepts before harder ones.

---

## (c)

A possible failure mode is that the difficulty estimate is wrong.

For example, "easy" examples may be:

- unrepresentative;
- biased;
- redundant;
- misleading.

Another possibility is that the model never receives sufficient exposure to difficult examples or becomes biased toward the easiest part of the data.

---

# Question 7 — Meta-Learning

### 7 points

## (a)

Algorithm A is more closely related to **Reptile**.

Reptile repeatedly:

1. starts from shared parameters;
2. adapts to a task;
3. obtains task-specific parameters;
4. moves the shared parameters toward those task-adapted parameters.

This resembles:

$$
\theta
\leftarrow
\theta+\beta(\theta_j-\theta).
$$

---

## (b)

Algorithm B is more closely related to **MAML**.

MAML explicitly considers how the parameters change after adaptation and computes gradients through the adaptation process.

---

## (c)

The key conceptual difference is how the meta-update is obtained.

### Reptile

Reptile approximately moves the initialisation toward parameters that work well after task-specific adaptation.

Conceptually:

$$
\theta_{\text{meta}}
\rightarrow
\theta_{\text{task-adapted}}.
$$

It avoids explicitly differentiating through the entire inner optimisation process in the same way as MAML.

### MAML

MAML asks:

> "Which initial parameters will allow a model to adapt rapidly to a new task?"

It therefore optimises the performance **after adaptation** and computes meta-gradients through the adaptation process.

Thus:

$$
\boxed{\text{Reptile: move toward adapted parameters}}
$$

versus

$$
\boxed{\text{MAML: optimise the initialisation through adaptation gradients}}.
$$

### Exam trap

Do not say that MAML and Reptile have completely different objectives.

They share the broad objective of finding parameters that are a good starting point for rapid adaptation.

Their optimisation procedures differ.

---

# Question 8 — Integrated Reasoning Problem

### 12 points

We have:

- one large pretrained model;
- 20 target tasks;
- limited GPU memory;
- limited storage;
- little labelled data;
- inference-latency constraints.

The proposed alternatives are:

### Strategy A

Full fine-tuning 20 times, then quantise.

### Strategy B

One shared frozen model + one LoRA adapter per task + possible quantisation.

---

## (a) Which strategy should be investigated first?

**Strategy B.**

The main reasons are:

1. the target datasets are relatively small;
2. GPU memory is limited;
3. 20 full copies are expensive to train and store;
4. PEFT is specifically designed to reduce trainable parameter count;
5. LoRA allows task-specific adaptation while retaining a shared base model.

---

## (b) How does Strategy B reduce training cost and storage?

### Training

Instead of updating all parameters:

$$
W\rightarrow W+\Delta W,
$$

LoRA trains only:

$$
A,B.
$$

For a square matrix:

$$
N_{\text{dense}}=d^2
$$

whereas:

$$
N_{\text{LoRA}}=2dr.
$$

When:

$$
r\ll d,
$$

the trainable parameter count is much smaller.

This reduces:

- gradients;
- optimizer states;
- trainable parameter storage.

### Storage

Instead of storing:

$$
20\times\text{complete models},
$$

we can store:

$$
1\times\text{shared base model}
+
20\times\text{small adapters}.
$$

This can be dramatically smaller.

---

## (c) What information is lost by freezing the base model?

The base model cannot directly change its representation in response to the target task.

All adaptation must occur through the LoRA parameters.

Therefore, if the target task requires a substantial transformation that cannot be represented well by the chosen low-rank updates, performance may suffer.

This is one reason rank is an important hyperparameter.

---

## (d) How can we investigate whether the LoRA rank is sufficient?

Run a controlled rank sweep.

For example:

$$
r\in\{2,4,8,16,32\}.
$$

Keep other experimental factors as constant as possible.

Measure:

- validation accuracy;
- training cost;
- memory;
- adapter size;
- inference latency.

The goal is not simply to choose the rank with the highest accuracy.

Instead, choose an appropriate **accuracy-efficiency trade-off**.

For example, if:

| Rank | Accuracy |
| --- | --- |
| 8 | 91.8% |
| 16 | 92.0% |
| 32 | 92.1% |

then rank 8 or 16 may be preferable if the additional accuracy from rank 32 is not worth the additional cost.

---

## (e) What if INT4 causes unacceptable accuracy degradation?

Two reasonable alternatives are:

### Alternative 1 — INT8

Use INT8 instead of INT4.

Advantages:

- generally less quantisation error;
- often strong hardware support;
- typically better accuracy than more aggressive INT4 quantisation.

Trade-off:

$$
\text{more memory}
$$

than INT4.

---

### Alternative 2 — FP16/BF16

Use a higher-precision representation.

Advantages:

- less numerical approximation;
- potentially better accuracy.

Trade-off:

- larger model memory;
- potentially greater bandwidth requirements.

---

Other acceptable answers include:

- quantisation-aware training;
- selective higher precision for sensitive layers;
- structured pruning;
- mixed-precision deployment.

---

# Mock Exam 3 — High-Level Answer Summary

| Question | Core answer |
| --- | --- |
| Q1 | Deployment distribution shift explains validation/deployment gap |
| Q2 | Quantisation reduces precision; pruning removes parameters; hardware determines actual speed |
| Q3 | PEFT reduces trainable state, not necessarily total model memory |
| Q4 | Rank controls LoRA capacity and cost |
| Q5 | Collapse occurs when all embeddings become similar |
| Q6 | Curriculum learning progresses from easier to harder examples |
| Q7 | Reptile moves toward task-adapted parameters; MAML differentiates through adaptation |
| Q8 | Shared frozen base + task-specific LoRA is a strong starting strategy |

---

# Cross-Exam Concepts You Should Be Able to Explain

The following distinctions are particularly important because they recur across the three exams.

## 1\. Training efficiency vs inference efficiency

Reducing trainable parameters does **not** automatically reduce inference computation.

For example, LoRA may train only:

$$
2dr
$$

parameters instead of:

$$
d^2,
$$

but the base matrix $W$ is still needed during the forward pass.

Therefore:

$$
\boxed{\text{parameter-efficient training}\neq\text{automatically cheaper inference}}
$$

---

## 2\. Parameter count vs memory

A model's memory footprint includes more than trainable parameters.

During training, memory can be consumed by:

- weights;
- gradients;
- optimizer state;
- activations;
- temporary buffers.

Therefore:

$$
\boxed{\text{fewer trainable parameters}\neq\text{proportionally less total GPU memory}}
$$

---

## 3\. Model compression vs PEFT

Quantisation and pruning primarily target **model/deployment efficiency**.

LoRA primarily targets **efficient adaptation**.

A useful conceptual separation is:

| Method | Main idea |
| --- | --- |
| Quantisation | Fewer bits per value |
| Pruning | Remove redundant parameters/structures |
| LoRA | Train a low-rank task-specific update |
| Freezing | Stop selected parameters from being updated |
| Distillation | Transfer behaviour from teacher to student |

These techniques can also be combined.

---

## 4\. LoRA vs low-rank compression

This distinction is extremely exam-worthy.

LoRA uses:

$$
W'=W+BA.
$$

The claim is **not**:

$$
W\approx BA.
$$

Instead:

$$
\Delta W\approx BA.
$$

Therefore:

$$
\boxed{\text{LoRA assumes a useful low-rank update, not necessarily a low-rank base model}}
$$

---

## 5\. Validation vs deployment

A validation score estimates performance on the validation distribution.

It does not guarantee deployment performance.

If:

$$
P_{\text{deployment}}(x)
\neq
P_{\text{validation}}(x),
$$

then validation performance may be a poor estimate of deployment performance.

This is why domain shift is important.

---

## 6\. Ensemble diversity

The central intuition behind ensembles is not simply:

> "More models are always better."

It is:

$$
\boxed{\text{competent models}+\text{different errors}}
$$

can produce a stronger combined predictor.

If all models make identical errors, averaging provides little benefit.

---

## 7\. Knowledge distillation

The teacher provides a richer target than a hard class label.

Hard label:

$$
y\in\{0,1,\ldots,K-1\}.
$$

Teacher:

$$
p_T(y|x).
$$

The student can therefore learn both:

- the correct class;
- the teacher's relative beliefs about alternative classes.

Temperature can soften these probabilities.

---

## 8\. Curriculum and self-paced learning

The central intuition is:

$$
\boxed{\text{easy first}\rightarrow\text{harder later}}
$$

but the exact mechanism matters.

In **self-paced learning**, the model can use its own current loss as a difficulty signal.

For example:

$$
L_i<\lambda.
$$

As:

$$
\lambda\uparrow,
$$

more examples become eligible.

---

## 9\. Reptile vs MAML

Both are meta-learning methods designed around rapid adaptation.

A useful exam-level distinction is:

### Reptile

$$
\theta
\leftarrow
\theta+\beta(\theta_j-\theta)
$$

Move the shared parameters toward parameters obtained after task-specific training.

### MAML

Optimise the initialisation by differentiating through the adaptation process.

The key question is:

> Which initial parameters allow rapid improvement after a small number of gradient steps?

---

## 10\. The most important general exam strategy

When confronted with an unfamiliar algorithm, do **not** immediately try to remember its name.

Instead ask:

### Step 1 — What is changing?

Are we changing:

- weights?
- data?
- labels?
- representations?
- model architecture?
- training examples?

### Step 2 — What is frozen?

Ask:

$$
\text{What does the algorithm deliberately avoid changing?}
$$

### Step 3 — What signal drives the change?

Is it:

- ground-truth labels?
- teacher predictions?
- example loss?
- similarity?
- gradients?
- task performance?
- distribution statistics?

### Step 4 — What problem is this solving?

For example:

$$
\text{limited labels}
\rightarrow
\text{self/semi-supervised learning}
$$

$$
\text{limited training memory}
\rightarrow
\text{PEFT}
$$

$$
\text{large deployment model}
\rightarrow
\text{quantisation/pruning/distillation}
$$

$$
\text{rapid adaptation to new tasks}
\rightarrow
\text{meta-learning}
$$

$$
\text{source/target distribution mismatch}
\rightarrow
\text{domain adaptation}
$$

### Step 5 — What is the trade-off?

Almost every advanced ML technique has one.

For example:

$$
\text{LoRA}
\rightarrow
\text{less training cost}
\quad\text{but potentially less adaptation capacity}
$$

$$
\text{quantisation}
\rightarrow
\text{less memory}
\quad\text{but possible accuracy loss}
$$

$$
\text{pruning}
\rightarrow
\text{fewer parameters}
\quad\text{but hardware speedup depends on structure}
$$

$$
\text{distillation}
\rightarrow
\text{smaller student}
\quad\text{but student may inherit teacher weaknesses}
$$

That style of reasoning is very close to the real exam questions you supplied and is more valuable for preparation than memorising isolated definitions.

:::
