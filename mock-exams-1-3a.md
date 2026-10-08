Yes. I’ve restructured the mocks around the **actual UAB exam pattern**, rather than the earlier generic 100-mark format.

The important characteristics I’m preserving are:

- **90 minutes**
- **50 points**
- Mixture of **multiple-choice and free-written answers**
- MCQs may have **multiple correct answers**
- Questions test **high-level intuition and reasoning**
- You may need to **interpret an algorithm, pseudocode, equation, matrix, or code/error message**
- Questions ask _why_ something works, not merely for definitions
- Some questions combine several concepts
- Very little emphasis on lengthy numerical calculation
- Answers should be concise but technically justified
- The three mocks increase in difficulty, but all remain plausible UAB-style exams

The current UAB 2026/27 course guide also describes the course as focusing on resource-constrained learning, transfer/extension of existing models, model analysis, vulnerabilities, robustness, and privacy, with an emphasis on explaining principles, comparing methods, and selecting appropriate methods under constraints. ([UAB Apps](<https://apps.uab.cat/guies/public/html/2026/assignatura/106575/en?utm_source=chatgpt.com>))

Below are the three redesigned mocks.

---

:::writing{variant="document" id="61427" title="Advanced Machine Learning — UAB-Style Mock Exam 1"}

# Advanced Machine Learning

## Mock Exam 1 — UAB Style

**Time:** 90 minutes **Total:** 50 points

### Instructions

- Answer all questions.
- For multiple-choice questions, **mark all answers that apply**.
- The number of points is **not related to the number of correct options**.
- If an answer could be interpreted in more than one way, briefly explain your interpretation.
- For free-written questions, concise but technically precise answers are expected.
- You do not need to reproduce algorithms exactly. You may use natural language, pseudocode, equations, or diagrams.
- Focus on explaining **what an algorithm is doing, what it is optimising, and why it works**.

---

# Question 1 — Transfer Learning and Domain Adaptation

### 5 points

**Which of the following statements about transfer learning and domain adaptation are true? Mark all that apply.**

□ Transfer learning can be useful when the target task has less labelled data than the source task.

□ Fine-tuning always requires changing the architecture of the pretrained model.

□ Transfer learning is generally more useful when the source and target problems share useful representations.

□ Domain adaptation can be useful when the source and target tasks are related but their data distributions differ.

□ A pretrained model can never perform worse on the target task than a randomly initialised model.

**Explain your answer to the final statement.**

---

# Question 2 — Fine-Tuning Behaviour

### 5 points

A student uses a pretrained image classification model and obtains the following results:

| Model | Training accuracy | Validation accuracy |
| --- | --- | --- |
| From scratch | 96% | 71% |
| Pretrained + frozen backbone | 84% | 79% |
| Pretrained + fine-tuning | 91% | 83% |

The student concludes:

> "Fine-tuning is better because it always increases training accuracy."

### (a) 2 points

Explain why this conclusion is incorrect or incomplete.

### (b) 3 points

Give a plausible explanation for why the pretrained model with a frozen backbone can already outperform the model trained from scratch on validation data.

---

# Question 3 — Self-Supervised Learning

### 4 points

A student wants to learn useful image representations without class labels.

They implement the following procedure:

1. Take an image $x$.
2. Create two randomly augmented versions $x_1$ and $x_2$.
3. Pass both through an encoder.
4. Encourage their embeddings to be close.
5. For images from different examples, encourage their embeddings to be far apart.

### (a) 2 points

What type of learning paradigm is this describing?

### (b) 2 points

Explain the role of the data augmentations.

---

# Question 4 — Contrastive Learning

### 5 points

A contrastive-learning model processes four images:

$$
a,\quad a^*,\quad b,\quad b^*
$$

where $a^*$ is an augmentation of $a$, and $b^*$ is an augmentation of $b$.

The model produces embeddings:

$$
z_a,\quad z_{a^*},\quad z_b,\quad z_{b^*}.
$$

For each pair, the training objective can either encourage a **HIGH** or **LOW** distance.

Fill in the intended objective:

| Pair | Desired distance |
| --- | --- |
| $a,a^*$ | \_\_\_\_\_\_ |
| $b,b^*$ | \_\_\_\_\_\_ |
| $a,b$ | \_\_\_\_\_\_ |
| $a,b^*$ | \_\_\_\_\_\_ |
| $a^*,b$ | \_\_\_\_\_\_ |
| $a^*,b^*$ | \_\_\_\_\_\_ |

Then explain why simply making **all** embeddings close together would not produce useful representations.

---

# Question 5 — Model Compression and Quantisation

### 6 points

Consider the following claim:

> "INT8 quantisation makes a neural network four times smaller because FP32 uses 32 bits and INT8 uses 8 bits. Therefore, it will always run four times faster."

### (a) 2 points

Which part of the claim concerning model size is approximately correct?

### (b) 2 points

Why does the memory reduction not necessarily translate into a four-times speedup?

### (c) 2 points

Give one reason why quantisation can reduce model accuracy.

---

# Question 6 — LoRA

### 7 points

A pretrained model contains a weight matrix

$$
W\in\mathbb{R}^{d\times d}.
$$

During adaptation, instead of directly learning an update $\Delta W$, LoRA uses

$$
W' = W + BA
$$

where

$$
A\in\mathbb{R}^{r\times d},
\qquad
B\in\mathbb{R}^{d\times r}.
$$

### (a) 2 points

How many trainable parameters would ordinary fine-tuning require for this matrix?

### (b) 2 points

How many trainable parameters does LoRA introduce?

### (c) 2 points

Why is LoRA particularly parameter-efficient when

$$
r\ll d?
$$

### (d) 1 point

Does the use of LoRA imply that the original pretrained matrix $W$ is low rank? Explain briefly.

---

# Question 7 — Knowledge Distillation

### 6 points

**Which of the following are true about knowledge distillation? Mark all that apply.**

□ The student can learn from both ground-truth labels and teacher predictions.

□ The teacher must have exactly the same architecture as the student.

□ Soft teacher predictions can provide information about relationships between classes.

□ The student must contain at least as many parameters as the teacher.

□ A student can inherit undesirable biases or weaknesses from its teacher.

□ Temperature can be used to soften the teacher's output distribution.

### (a) 3 points

Choose the correct statements.

### (b) 3 points

Explain why the teacher's soft predictions can contain information that is not captured by the hard class label alone.

---

# Question 8 — Algorithm Interpretation

### 12 points

A lecturer presents the following training algorithm.

For a parameter $\lambda$, at each epoch:

1. Compute the loss for every training example.
2. Select examples whose loss is below $\lambda$.
3. Backpropagate only through the selected examples.
4. Update the model.
5. Increase $\lambda$.
6. Repeat.

### (a) 3 points

What learning strategy is this most closely related to?

### (b) 3 points

What is the intuition behind initially selecting only low-loss examples?

### (c) 3 points

Why is $\lambda$ increased during training?

### (d) 3 points

What could happen if $\lambda$ were set extremely high from the beginning?

---

# End of Mock Exam 1

**Total: 50 points**

**Suggested time:** Questions 1–3: 25 minutes Questions 4–6: 30 minutes Questions 7–8: 30 minutes Review: 5 minutes

:::

---

:::writing{variant="document" id="73852" title="Advanced Machine Learning — UAB-Style Mock Exam 2"}

# Advanced Machine Learning

## Mock Exam 2 — UAB Style

**Time:** 90 minutes **Total:** 50 points

### Instructions

- Answer all questions.
- For multiple-choice questions, mark **all** correct answers.
- Justify free-written answers using intuition rather than memorised definitions.
- You may use pseudocode, equations, diagrams, or natural language.
- Some questions deliberately contain plausible but incorrect statements.

---

# Question 1 — Transfer Learning

### 4 points

You have two datasets:

- **Source:** 2 million labelled photographs of everyday objects.
- **Target:** 2,000 labelled photographs of specialised industrial components.

A student proposes training the target model completely from scratch because:

> "The target classes are different, so the source model contains no useful information."

### (a) 2 points

Explain why this reasoning is potentially incorrect.

### (b) 2 points

Give one situation in which transferring the source model could actually be harmful.

---

# Question 2 — Domain Shift

### 5 points

A model is trained using photographs taken in daylight.

During deployment, most images are taken at night.

The model achieves:

- Training accuracy: 97%
- Validation accuracy: 94%
- Deployment accuracy: 63%

**Which of the following could explain this behaviour? Mark all that apply.**

□ The deployment distribution differs from the training distribution.

□ The model necessarily has insufficient capacity.

□ The validation split may not represent the deployment distribution.

□ Domain adaptation could potentially help.

□ The high training accuracy proves the model has learned features that generalise to deployment.

### Explain one reason why a high validation accuracy does not guarantee high deployment accuracy in this scenario.

---

# Question 3 — Semi-Supervised Learning

### 5 points

A dataset contains:

- 5,000 labelled examples;
- 200,000 unlabelled examples.

A student proposes:

> "We should simply ignore the unlabelled data because supervised learning requires labels."

### (a) 2 points

Explain why this may be a poor strategy.

### (b) 3 points

Describe one reasonable semi-supervised approach for making use of the unlabelled examples.

Your description does not need to reproduce exact mathematical details.

---

# Question 4 — Self-Distillation

### 6 points

A student trains two networks with the **same architecture**.

The first network is trained normally.

The second network is then trained using both the original labels and predictions from the first network.

Surprisingly, the second network achieves higher test accuracy.

### (a) 3 points

Explain how this can happen even though the student has the same architecture as the teacher.

### (b) 3 points

Would you expect the student to necessarily outperform the teacher? Explain.

---

# Question 5 — LoRA Implementation

### 6 points

A student implements LoRA as follows:

```
W' = W + BA

initialise W from the pretrained model
initialise A randomly
initialise B randomly

freeze W

train A and B
```

Before any training, the adapted model performs substantially worse than the original pretrained model.

The student says:

> "This must mean LoRA is fundamentally unsuitable."

### (a) 2 points

Identify a likely implementation issue.

### (b) 2 points

What should the LoRA update ideally be at the beginning of training?

### (c) 2 points

Why is this useful for preserving the pretrained model's initial behaviour?

---

# Question 6 — Model Merging and Task Arithmetic

### 6 points

Suppose a pretrained model has parameters

$$
\theta_0.
$$

Two task-specific models are obtained:

$$
\theta_A
$$

and

$$
\theta_B.
$$

A student defines:

$$
\Delta_A=\theta_A-\theta_0
$$

and

$$
\Delta_B=\theta_B-\theta_0.
$$

They then construct:

$$
\theta_{\text{merged}}
=
\theta_0+\alpha\Delta_A+\beta\Delta_B.
$$

### (a) 2 points

What is the intuition behind $\Delta_A$ and $\Delta_B$?

### (b) 2 points

What does changing $\alpha$ and $\beta$ do?

### (c) 2 points

Why might simply adding two task vectors fail to produce a good model?

---

# Question 7 — Meta-Learning

### 6 points

Consider the following Reptile-style update:

$$
\theta_M
\leftarrow
\theta_M+\beta(\theta_j-\theta_M).
$$

### (a) 2 points

Explain what $\theta_M$ represents.

### (b) 2 points

Explain what $\theta_j$ represents.

### (c) 2 points

What happens when

$$
\beta=1?
$$

What happens when $\beta$ is very small?

---

# Question 8 — Ensemble Reasoning

### 12 points

Five neural networks have been trained independently on the same task.

Each achieves approximately 80% test accuracy.

However, averaging their predictions produces 86% test accuracy.

A student claims:

> "This is impossible because none of the five models has more than 80% accuracy."

### (a) 4 points

Explain theoretically why averaging predictions can improve performance.

### (b) 3 points

Why is diversity between the models important?

### (c) 2 points

Would the same basic idea work for linear regression models? Explain.

### (d) 3 points

Would simply selecting the single model with the highest validation accuracy necessarily give the same benefit? Explain.

---

# End of Mock Exam 2

**Total: 50 points**

**Suggested time:** Questions 1–3: 30 minutes Questions 4–6: 30 minutes Questions 7–8: 25 minutes Review: 5 minutes

:::

---

:::writing{variant="document" id="85194" title="Advanced Machine Learning — UAB-Style Mock Exam 3"}

# Advanced Machine Learning

## Mock Exam 3 — UAB Style

**Time:** 90 minutes **Total:** 50 points

### Instructions

This paper is deliberately closer to a difficult real exam.

Questions may require you to identify the underlying algorithm from its behaviour, distinguish between superficially similar methods, or reason about why an apparently sensible approach fails.

- Mark all correct answers for multiple-choice questions.
- Explain ambiguous answers.
- Do not spend excessive time deriving equations that are not necessary.
- The key question is usually: **what is the method trying to achieve, and why?**

---

# Question 1 — Deployment Generalisation

### 4 points

A model produces the following results:

|  | Accuracy |
| --- | --- |
| Training | 98% |
| Validation | 96% |
| Deployment | 61% |

The deployment environment contains data from a different geographic region.

**Which explanations are plausible? Mark all that apply.**

□ The deployment distribution may differ from the training distribution.

□ The model necessarily has too few parameters.

□ The validation set may not have captured the relevant distribution shift.

□ Domain adaptation could potentially improve performance.

□ Increasing model capacity is guaranteed to solve the problem.

Then briefly explain why **validation performance alone cannot diagnose all forms of deployment failure**.

---

# Question 2 — Compression

### 5 points

A model is compressed using the following sequence:

$$
\text{FP32}
\rightarrow
\text{INT8}
\rightarrow
\text{Pruned}
\rightarrow
\text{INT4}.
$$

### (a) 2 points

Which two different kinds of redundancy are primarily being exploited by quantisation and pruning?

### (b) 2 points

Why might an INT4 model with fewer parameters still have worse practical latency than an INT8 model?

### (c) 1 point

Why can structured pruning be more attractive for deployment than arbitrary unstructured pruning?

---

# Question 3 — PEFT and Memory

### 6 points

A 10-billion-parameter model is adapted using LoRA.

The student says:

> "Only 1% of the parameters are trainable, therefore GPU memory usage during training should also fall by 99%."

### (a) 2 points

Why is this statement incomplete?

### (b) 2 points

Which components associated with the frozen parameters no longer need to be stored as trainable state?

### (c) 2 points

Why must the original model weights still normally be available during LoRA training?

---

# Question 4 — LoRA Rank

### 5 points

Consider:

$$
\Delta W=BA
$$

with

$$
A\in\mathbb{R}^{r\times d},
\qquad
B\in\mathbb{R}^{d\times r}.
$$

### (a) 2 points

Explain the effect of increasing $r$.

### (b) 2 points

Why does a larger rank not automatically guarantee better validation performance?

### (c) 1 point

Give one reason for preferring a smaller rank in deployment.

---

# Question 5 — Contrastive Learning Failure

### 6 points

A student implements contrastive learning but accidentally applies the **same augmentation pipeline to every image** and uses the following objective:

> "Make all embeddings in a batch as similar as possible."

After training, all images have almost identical embeddings.

### (a) 2 points

What problem has occurred?

### (b) 2 points

Why is this representation not useful?

### (c) 2 points

Explain how a contrastive objective normally avoids this type of collapse.

---

# Question 6 — Curriculum Learning

### 5 points

A training algorithm works as follows:

```
for each epoch:
    compute a difficulty score for every training example
    sort examples by difficulty
    train only on the easiest examples
    increase the difficulty threshold
```

### (a) 2 points

What learning strategy is this implementing?

### (b) 2 points

Why might starting with easier examples help optimisation?

### (c) 1 point

Give one possible failure mode of this strategy.

---

# Question 7 — Meta-Learning

### 7 points

You are given two algorithms.

### Algorithm A

For each task:

1. Start from the same initial parameters.
2. Perform several gradient steps on that task.
3. Compute how the adapted parameters should change the shared initialisation.
4. Update the shared initialisation.

### Algorithm B

For each task:

1. Start from the shared parameters.
2. Perform several gradient steps.
3. Compute gradients through the adaptation process itself.
4. Update the shared parameters using these higher-order gradients.

### (a) 2 points

Which algorithm is more closely related to Reptile?

### (b) 2 points

Which algorithm is more closely related to MAML?

### (c) 3 points

Explain the conceptual difference between the two approaches.

---

# Question 8 — Integrated Reasoning Problem

### 12 points

You are given a large pretrained model.

You need to adapt it to **20 different target tasks**.

The constraints are:

- the GPU available for training has limited memory;
- you have limited storage;
- the base model is expensive to retrain;
- the target tasks have relatively little labelled data;
- inference latency matters;
- each task should be independently deployable.

A student proposes:

> "Fine-tune the entire model separately 20 times, then quantise each resulting model."

Another student proposes:

> "Keep one shared pretrained model, freeze it, train a small task-specific LoRA adapter for each task, and consider quantisation of the shared model."

### (a) 3 points

Which strategy would you investigate first, and why?

### (b) 3 points

Explain how the second strategy reduces training cost and storage requirements.

### (c) 2 points

What information is lost by freezing the base model?

### (d) 2 points

How could you investigate whether the chosen LoRA rank is sufficient?

### (e) 2 points

Suppose INT4 quantisation causes a significant accuracy loss. Give two sensible alternatives and explain what trade-off each introduces.

---

# End of Mock Exam 3

**Total: 50 points**

**Suggested time:** Questions 1–3: 25 minutes Questions 4–6: 30 minutes Questions 7–8: 30 minutes Review: 5 minutes

:::

### Why these are much closer to the real UAB exam

I deliberately changed the design from the earlier mocks in several important ways.

The real examples you supplied repeatedly use a pattern of **"recognise the situation → identify the underlying method → explain why"**, rather than asking for long textbook definitions. For example, the Reptile question gives the update equation and asks what the terms mean and what happens when $\beta=1$. The curriculum question gives an algorithm and asks the student to identify and explain it. Your contrastive-learning question similarly gives an implementation and asks the student to reason about the desired distances.

I have therefore built the new mocks around the same pattern:

- **MCQ with multiple correct answers**, rather than conventional single-answer MCQs.
- **"Explain your answer"** follow-ups.
- Algorithm identification from **pseudocode**.
- Interpretation of **equations** rather than lengthy derivations.
- **Debugging/reasoning from an incorrect implementation**.
- Small tables and experimental results requiring interpretation.
- Questions where a statement sounds plausible but is subtly wrong.
- Cross-topic reasoning, especially **LoRA + quantisation + deployment**, **transfer learning + domain shift**, and **self-distillation + knowledge distillation**.
- Minimal arithmetic, because the real exam examples you provided are overwhelmingly conceptual and algorithmic.
- A progression from straightforward recognition in Mock 1 to deliberately misleading statements and integrated reasoning in Mock 3.

I have also kept the current course coverage in mind. The official 2026/27 UAB guide lists **transfer learning/domain adaptation, parameter-efficient learning/model compression, semi- and self-supervised learning, ensemble learning/knowledge distillation, continual/online learning, multi-task/meta-learning, explainability, adversarial robustness, and federated learning/privacy** among the course contents. ([UAB Apps](<https://apps.uab.cat/guies/public/html/2026/assignatura/106575/en?utm_source=chatgpt.com>))

**One important observation:** your supplied previous exam examples are specifically labelled as covering the **first half of the course**, so these three mocks deliberately concentrate more heavily on that first-half material rather than pretending that every later topic must appear equally. That makes them better practice for the structure you showed us.

Content diagnostic: content\_references
