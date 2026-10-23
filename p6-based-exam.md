# Advanced Machine Learning — Practice Exam 6

## Few-Shot Learning and Meta-Learning

**Exam format:** Multiple-choice and written answers\
**Total marks:** 50\
**Suggested duration:** 90 minutes\
**Topics:** Few-shot learning, episodic evaluation, supervised baselines, Reptile, Prototypical Networks (ProtoNet), optimisation, debugging, and experimental design.

---

## Instructions

1. Attempt each question before reading its model answer.
2. Multiple-choice questions may have more than one correct answer.
3. For written questions, explain the mechanism and reasoning rather than merely naming the algorithm.
4. No executable code is required.
5. The model answers and marking schemes are provided for self-assessment.

**Mark distribution:** Multiple-choice questions: 16 marks. Written questions: 34 marks.

---

# Part A — Exam Questions and Model Answers

## Question 1 — Few-Shot Learning (2 marks)

**Question**

Which statements about few-shot learning are correct? Select all that apply.

A. It aims to recognise classes using only a few labelled examples per class.

B. It necessarily requires fine-tuning every backbone parameter on the support set.

C. It can be evaluated on classes that were excluded from meta-training.

D. High training accuracy on the original classes guarantees strong few-shot performance.

\<details\> \<summary\>\<strong\>Model answer and explanation\</strong\>\</summary\>

**Correct answers: A and C.**

- **A — True.** Few-shot learning aims to recognise categories using only a small number of labelled examples.
- **B — False.** Fine-tuning the entire backbone is not required. A method may train a new classifier head, adapt parameters, or construct class prototypes.
- **C — True.** A common evaluation protocol uses held-out classes to measure generalisation to unfamiliar categories.
- **D — False.** High accuracy on familiar classes does not guarantee that the model can generalise to new classes from a few examples.

**Key takeaway:** Few-shot learning is about generalising from limited labelled examples, not simply achieving high accuracy on the original training classes.

**Marking:** 2 marks for selecting exactly A and C.

\</details\>

---

## Question 2 — Episodes and Data Splits (2 marks)

**Question**

Which statements about few-shot tasks are correct? Select all that apply.

A. The support set provides examples used to adapt or construct a task-specific classifier.

B. The query set provides examples used to evaluate predictions for the task.

C. Meta-validation and meta-testing must use the same tasks as meta-training.

D. An episode is one sampled task, usually containing support and query sets.

\<details\> \<summary\>\<strong\>Model answer and explanation\</strong\>\</summary\>

**Correct answers: A, B and D.**

- **A — True.** The support set defines class prototypes in ProtoNet or provides training examples for task adaptation.
- **B — True.** Query examples measure how well the model classifies additional examples from the task. During meta-training, query labels also provide the loss used to update the model.
- **C — False.** Meta-validation and meta-testing should use separate held-out tasks. When evaluating generalisation to unseen classes, these tasks should contain classes excluded from meta-training.
- **D — True.** An episode is one sampled few-shot task containing its selected classes and associated support and query examples.

**Key takeaway:** Support examples define or adapt the classifier; query examples evaluate it.

**Marking:** 2 marks for selecting exactly A, B and D.

\</details\>

---

## Question 3 — Task Construction (3 marks)

**Question**

A five-way, five-shot task contains 12 query examples per class.

1. Calculate the total number of support examples.
2. Calculate the total number of query examples.
3. Define an episode.

\<details\> \<summary\>\<strong\>Model answer and marking scheme\</strong\>\</summary\>

**1\. Support examples**

There are five classes and five support examples per class:

$$
N_{\text{support}}=5\times5=25
$$

**2\. Query examples**

There are five classes and 12 query examples per class:

$$
N_{\text{query}}=5\times12=60
$$

**3\. Episode**

An episode is one sampled few-shot classification task consisting of selected classes and their support and query examples.

**Marking:**

- Correct support count: 1 mark.
- Correct query count: 1 mark.
- Correct episode definition: 1 mark.

\</details\>

---

## Question 4 — Meta-Learning Evaluation (3 marks)

**Question**

Explain the difference between meta-training, meta-validation, and meta-testing. Why should the meta-test set not be repeatedly used to choose model checkpoints?

\<details\> \<summary\>\<strong\>Model answer and marking scheme\</strong\>\</summary\>

- **Meta-training:** The model learns from sampled tasks. The training objective updates its parameters or representations so that it can solve similar tasks.
- **Meta-validation:** Separate held-out tasks are used to monitor progress, compare hyperparameters, and select model checkpoints.
- **Meta-testing:** Reserved tasks are used to estimate final generalisation performance.

Repeatedly selecting checkpoints based on meta-test performance leaks information from the test set into model selection. This can lead to overfitting to the meta-test tasks and an optimistically biased estimate of generalisation.

Ideally, select the model using meta-validation tasks and evaluate it on the reserved meta-test tasks after model selection is complete.

**Marking:**

- Meta-training correctly explained: 1 mark.
- Meta-validation and meta-testing distinguished: 1 mark.
- Test-set leakage and optimistic bias explained: 1 mark.

\</details\>

---

## Question 5 — Supervised Baseline (4 marks)

**Question**

The baseline is trained on meta-training classes. During few-shot evaluation, its backbone is frozen and a fresh classification head is trained on each task's support set.

Which statements are correct? Select all that apply.

A. The backbone extracts image representations.

B. The fresh head assigns outputs to the classes in the current task.

C. The original head is guaranteed to classify unseen classes correctly without adaptation.

D. The experiment tests whether supervised pre-training learned transferable features.

\<details\> \<summary\>\<strong\>Model answer and explanation\</strong\>\</summary\>

**Correct answers: A, B and D.**

- **A — True.** The backbone transforms each image into a feature representation.
- **B — True.** A new classification head is trained to map the extracted features to the classes present in the current few-shot task.
- **C — False.** The original head was trained for the original class vocabulary. It is not automatically suitable for unseen classes.
- **D — True.** Freezing the backbone and training a fresh head tests whether the features learned through supervised pre-training transfer to unfamiliar classes.

The baseline is a useful reference point for evaluating whether meta-learning provides an advantage over ordinary supervised representation learning.

**Marking:** 4 marks for selecting exactly A, B and D.

\</details\>

---

## Question 6 — Evaluation Metrics (3 marks)

**Question**

Explain top-1 accuracy, top-5 accuracy, and cross-entropy loss. Which metric is most directly relevant when comparing the three methods on held-out few-shot tasks, and why?

\<details\> \<summary\>\<strong\>Model answer and marking scheme\</strong\>\</summary\>

**Top-1 accuracy**

The fraction of examples for which the highest-scoring predicted class is correct.

**Top-5 accuracy**

The fraction of examples for which the true class appears among the five highest-scoring predicted classes.

**Cross-entropy loss**

A classification loss that penalises the model when it assigns low probability to the correct class. It measures the discrepancy between predicted class probabilities and the ground-truth labels.

**Most relevant comparison metric**

Average query top-1 accuracy on held-out meta-test tasks is the most directly relevant metric because it measures how well each method classifies examples in the few-shot evaluation setting.

Baseline training accuracy measures performance on familiar training classes and does not directly establish generalisation to unseen tasks.

**Marking:**

- Correct top-1 and top-5 definitions: 1 mark.
- Correct cross-entropy explanation: 1 mark.
- Identifies and justifies held-out query accuracy: 1 mark.

\</details\>

---

## Question 7 — Reptile Update (4 marks)

**Question**

Reptile updates its meta-parameters using

$$
\theta \leftarrow \theta+\epsilon(\theta'-\theta)
$$

Explain $\theta$, $\theta'$, and $\epsilon$.

If $\theta=2$, $\theta'=6$, and $\epsilon=0.25$, calculate the updated parameter.

\<details\> \<summary\>\<strong\>Model answer and marking scheme\</strong\>\</summary\>

**Meaning of the terms**

- $\theta$: The original meta-model parameter before task-specific adaptation.
- $\theta'$: The corresponding parameter after inner-loop fine-tuning on a sampled task.
- $\epsilon$: The outer-loop meta-step size, which controls how far the meta-parameter moves toward the task-adapted value.

**Calculation**

Substitute the values into the update rule:

$$
\begin{aligned}
\theta_{\text{new}}
&=2+0.25(6-2)\\
&=2+0.25(4)\\
&=2+1\\
&=3
\end{aligned}
$$

The updated parameter is **3**.

The meta-model moves one-quarter of the distance from 2 toward 6.

**Marking:**

- Correct interpretation of $\theta$ and $\theta'$: 2 marks.
- Correct interpretation of $\epsilon$: 1 mark.
- Correct calculation: 1 mark.

\</details\>

---

## Question 8 — Reptile Algorithm (4 marks)

**Question**

Which statements about Reptile are correct? Select all that apply.

A. It samples tasks and performs inner-loop fine-tuning on each sampled task.

B. Its outer update moves the meta-initialisation toward the task-adapted parameters.

C. Setting $\epsilon=1$ makes the updated parameter equal to the task-adapted parameter for that update.

D. The outer meta-step size must equal the inner-loop learning rate.

E. It classifies queries by comparing embeddings to mean class prototypes, without parameter adaptation.

\<details\> \<summary\>\<strong\>Model answer and explanation\</strong\>\</summary\>

**Correct answers: A, B and C.**

- **A — True.** Reptile adapts the current model on a sampled task using gradient-based optimisation.
- **B — True.** The outer update moves the initial parameters toward the parameters obtained after task-specific adaptation.
- **C — True.** If $\epsilon=1$,

$$
\theta_{\text{new}}=\theta+(\theta'-\theta)=\theta'
$$

The updated parameter equals the task-adapted parameter for that update.

- **D — False.** The inner-loop learning rate controls fine-tuning on the task. The outer meta-step size controls the movement of the meta-initialisation. They are separate hyperparameters.
- **E — False.** This describes prototype-based classification, as used by ProtoNet, rather than Reptile.

**Important distinction:** Reptile learns an initialisation that can adapt quickly; ProtoNet learns an embedding space that supports prototype-based classification.

**Marking:** 4 marks for selecting exactly A, B and C.

\</details\>

---

## Question 9 — Prototypical Networks (4 marks)

**Question**

Three support embeddings from one class are

$$
z_1=(1,2),\qquad
z_2=(2,3),\qquad
z_3=(3,1)
$$

Calculate the prototype. Explain how ProtoNet uses it to classify a query.

\<details\> \<summary\>\<strong\>Model answer and marking scheme\</strong\>\</summary\>

**Step 1: Calculate the prototype**

ProtoNet calculates the mean of the support embeddings belonging to the class:

$$
c=\frac{z_1+z_2+z_3}{3}
$$

Calculate each coordinate separately:

$$
c=
\left(
\frac{1+2+3}{3},
\frac{2+3+1}{3}
\right)
$$

Therefore,

$$
\boxed{c=(2,2)}
$$

**Step 2: Classify a query**

1. Pass the query image through the backbone to obtain its embedding.
2. Calculate its distance to each class prototype.
3. Convert distances to negative-distance logits.
4. Predict the class with the largest logit, corresponding to the nearest prototype.

For Euclidean distance,

$$
d(z,c_k)=\sqrt{\sum_j(z_j-c_{k,j})^2}
$$

The prototype summarises the average embedding of the support examples in its class.

**Marking:**

- Correct mean calculation: 2 marks.
- Correct query-to-prototype distance explanation: 1 mark.
- Correct negative-distance classification explanation: 1 mark.

\</details\>

---

## Question 10 — ProtoNet Implementation and Tensor Shapes (4 marks)

**Question**

A five-way task contains 25 support examples, 75 query examples, and a 64-dimensional embedding.

Which statements are correct? Select all that apply.

A. The prototype tensor has shape `(5, 64)`.

B. The query-logit tensor has shape `(75, 5)`.

C. Positive Euclidean distances should be used directly as logits because a larger distance indicates a better match.

D. Each prototype is the mean of the embeddings of support examples belonging to its class.

E. The backbone cannot be trained because ProtoNet has no learned linear classification head.

\<details\> \<summary\>\<strong\>Model answer and explanation\</strong\>\</summary\>

**Correct answers: A, B and D.**

**A — True.** There is one prototype per class, and each prototype has 64 features. The shape is `(5, 64)`.

**B — True.** There are 75 query examples and five classes, so each query has five logits. The shape is `(75, 5)`.

**C — False.** Smaller distances indicate better matches. Therefore, ProtoNet negates the distances so that the nearest prototype has the largest logit.

**D — True.** The prototype is the mean of the support embeddings belonging to a particular class.

**E — False.** The backbone is trainable. The query loss propagates gradients through the query embeddings and the support-derived prototypes into the backbone.

**Expected shapes**

| Tensor | Shape |
| --- | --- |
| Support embeddings | `(25, 64)` |
| Prototypes | `(5, 64)` |
| Query embeddings | `(75, 64)` |
| Query logits | `(75, 5)` |
| Query labels | `(75,)` |

**Marking:** 4 marks for selecting exactly A, B and D.

\</details\>

---

## Question 11 — Temperature Scaling (3 marks)

**Question**

A query has Euclidean distances

$$
d=[1,3,2]
$$

to three prototypes. Calculate its logits when temperature $T=2$. Explain the effect of changing a positive temperature.

\<details\> \<summary\>\<strong\>Model answer and marking scheme\</strong\>\</summary\>

ProtoNet uses negative distances divided by temperature:

$$
z_k=-\frac{d_k}{T}
$$

For $T=2$,

$$
z=
\left[
-\frac{1}{2},
-\frac{3}{2},
-\frac{2}{2}
\right]
$$

Therefore,

$$
\boxed{z=[-0.5,-1.5,-1.0]}
$$

**Effect of temperature**

- Increasing $T$ reduces the magnitude of the logits and generally produces a softer probability distribution after softmax.
- Decreasing $T$ increases the relative sharpness of the probability distribution.
- For any positive temperature, the ordering of the logits remains unchanged, so the nearest prototype remains the predicted class.

Temperature changes the confidence distribution, not which prototype is nearest, assuming the distances and embeddings remain fixed.

**Marking:**

- Correct logits: 2 marks.
- Correct explanation of temperature and unchanged ranking for $T>0$: 1 mark.

\</details\>

---

## Question 12 — ProtoNet Meta-Training (4 marks)

**Question**

Describe one ProtoNet meta-training episode. Explain why the loss is calculated on query examples rather than using the support examples alone.

\<details\> \<summary\>\<strong\>Model answer and marking scheme\</strong\>\</summary\>

**One meta-training episode**

1. Sample a task from the meta-training classes.
2. Pass the support images through the backbone.
3. Calculate the mean embedding for each support class to construct the prototypes.
4. Pass the query images through the same backbone.
5. Calculate distances from query embeddings to the prototypes and convert them into logits.
6. Calculate cross-entropy loss using the query labels.
7. Backpropagate the loss and update the backbone parameters.

The objective is

$$
L_{\text{query}}
=
\operatorname{CE}(z_{\text{query}},y_{\text{query}})
$$

**Why use the query loss?**

The support examples define the class prototypes. The query examples measure whether those prototypes can classify additional examples correctly.

Training on query loss encourages the backbone to learn an embedding space that generalises beyond the particular examples used to construct the prototypes.

The gradients flow through the query embeddings and the support-derived prototypes. Consequently, the backbone learns representations that improve query classification.

**Marking:**

- Correct episodic forward pass: 2 marks.
- Correct query loss and optimiser update: 1 mark.
- Explains generalisation and gradient flow: 1 mark.

\</details\>

---

## Question 13 — Comparing the Three Methods (4 marks)

**Question**

Compare the baseline, Reptile, and ProtoNet in terms of what they learn and how they adapt to a new task in this practical.

\<details\> \<summary\>\<strong\>Model answer and marking scheme\</strong\>\</summary\>

| Property | Supervised baseline | Reptile | ProtoNet |
| --- | --- | --- | --- |
| Training approach | Ordinary supervised learning | Episodic meta-learning | Episodic meta-learning |
| Main learned component | Backbone and original classification head | Adaptable parameter initialisation | Embedding backbone |
| Task-specific classifier | Fresh classification head | Fresh head during adaptation | Mean support-set prototypes |
| Test-time adaptation | Train the new head while freezing the backbone | Fine-tune parameters on the support set | Compute prototypes from support embeddings |
| Gradient updates at test time | Yes, for the new head | Yes, during fine-tuning | No |

**Explanation**

The baseline learns visual representations from labelled meta-training classes. During few-shot evaluation, its backbone is frozen and a new classifier head is trained on the support examples.

Reptile learns an initialisation that can be adapted quickly. It fine-tunes the model on the support set and moves the meta-initialisation toward the task-adapted parameters during meta-training.

ProtoNet learns an embedding space in which examples can be classified by comparing query embeddings to prototypes calculated from the support embeddings. It does not perform gradient-based fine-tuning during meta-testing.

**Marking:**

- Baseline mechanism: 1 mark.
- Reptile mechanism: 1 mark.
- ProtoNet mechanism: 1 mark.
- Clearly distinguishes all three adaptation procedures: 1 mark.

\</details\>

---

## Question 14 — Debugging and Experimental Design (6 marks)

**Question**

A five-way ProtoNet produces logits of shape `(75, 5)`, but cross-entropy raises an error because query labels are `12, 27, 41, 58, 93`.

1. Explain the likely cause and how to fix it.
2. Explain one risk of selecting the best checkpoint repeatedly using meta-test accuracy.
3. Describe a better model-selection procedure.

\<details\> \<summary\>\<strong\>Model answer and marking scheme\</strong\>\</summary\>

### 1\. Label encoding problem

The five logit columns represent five task classes indexed from 0 to 4. However, the query labels contain global dataset IDs rather than valid task-local class indices.

For a five-class classification problem, cross-entropy expects integer class targets in the range

$$
0,1,2,3,4
$$

The global labels must be remapped consistently to local labels.

For example:

| Global label | Task-local label |
| --- | --- |
| 12 | 0 |
| 27 | 1 |
| 41 | 2 |
| 58 | 3 |
| 93 | 4 |

The same mapping must be used for support and query labels. The order of the prototype rows must match the local class indices represented by the query-logit columns.

### 2\. Risk of repeated meta-test checkpoint selection

Repeatedly selecting checkpoints based on meta-test accuracy leaks test information into model selection. The model and its hyperparameters may become indirectly tailored to the meta-test tasks, leading to an optimistically biased estimate of generalisation.

### 3\. Better procedure

Use separate meta-validation tasks to monitor performance, select checkpoints, and tune hyperparameters. Once model selection is complete, evaluate the chosen model on the reserved meta-test tasks for the final performance estimate.

**Marking:**

- Identifies global-label versus task-local-index mismatch: 2 marks.
- Explains consistent remapping and prototype alignment: 1 mark.
- Explains repeated meta-test selection and leakage: 1 mark.
- Gives a suitable validation-based selection procedure: 2 marks.

\</details\>

---

# Part B — Consolidated Revision Sheet

## 1\. Core terminology

| Term | Definition |
| --- | --- |
| Few-shot learning | Learning to recognise new classes from a small number of labelled examples. |
| Meta-learning | Learning across tasks so that the model can solve new tasks effectively. |
| Episode | One sampled task containing support and query sets. |
| Support set | Labelled examples used to construct prototypes or adapt a classifier. |
| Query set | Examples used to evaluate the task-specific classifier and calculate episodic training loss. |
| Meta-training | Optimising the model using sampled training tasks. |
| Meta-validation | Selecting hyperparameters and checkpoints using held-out validation tasks. |
| Meta-testing | Measuring final generalisation on reserved test tasks. |
| Backbone | The feature extractor that maps an image to an embedding. |
| Embedding | A learned numerical representation of an image. |
| Prototype | The mean embedding of the support examples belonging to a class. |

## 2\. Reptile: Essential equations

The Reptile update is

$$
\boxed{
\theta_{\text{new}}
=
\theta+\epsilon(\theta'-\theta)
}
$$

where:

- $\theta$ is the initial meta-parameter.
- $\theta'$ is the parameter after task-specific fine-tuning.
- $\epsilon$ is the outer-loop meta-step size.

Interpretation:

- $\epsilon=0$: no meta-update.
- $0<\epsilon<1$: move partway toward the task-adapted parameters.
- $\epsilon=1$: move completely to the task-adapted parameters for that update.

The inner-loop learning rate and outer-loop meta-step size are different hyperparameters.

**Remember:** Reptile learns an initialisation that supports rapid adaptation.

## 3\. ProtoNet: Essential equations

### Prototype construction

For class $k$, with support set $S_k$,

$$
\boxed{
c_k=\frac{1}{|S_k|}
\sum_{(x_i,y_i)\in S_k}f_\theta(x_i)
}
$$

### Euclidean distance

$$
d(z,c_k)
=
\sqrt{\sum_j(z_j-c_{k,j})^2}
$$

### Classification logits

$$
\boxed{
z_k=-\frac{d(f_\theta(x),c_k)}{T}
}
$$

where $T>0$ is the temperature.

### Query loss

$$
\boxed{
L_{\text{query}}
=
\operatorname{CE}(z_{\text{query}},y_{\text{query}})
}
$$

The query loss trains the backbone by backpropagating through query embeddings and the support-derived prototypes.

**Remember:** ProtoNet learns an embedding space suitable for prototype-based classification. It does not need a fixed learned linear classification head.

## 4\. Comparison of the three approaches

| Question | Baseline | Reptile | ProtoNet |
| --- | --- | --- | --- |
| What is learned? | Features and original classifier | Adaptable initialisation | Embedding space |
| How is a new task handled? | Train a new classifier head | Fine-tune parameters | Calculate support prototypes |
| Test-time gradient updates? | Yes, for the new head in this practical | Yes | No |
| Main principle | Transfer learned features | Learn how to adapt | Classify by embedding similarity |

## 5\. Common implementation errors

- **Wrong label indices:** Cross-entropy targets must match the task-local class indices represented by the logit columns.
- **Incorrect prototype order:** Prototype rows and target labels must use the same class mapping.
- **Wrong tensor shapes:** For $Q$ queries and $N$ classes, logits should have shape `(Q, N)`.
- **Missing optimiser steps:** Training requires clearing gradients, computing the loss, calling backpropagation, and updating parameters.
- **Incorrect model mode:** Use `train()` for optimisation and `eval()` for evaluation.
- **Test-set leakage:** Use validation tasks for checkpoint selection and reserve test tasks for final evaluation.
- **Misinterpreting metrics:** Training accuracy on familiar classes is not equivalent to few-shot query accuracy on held-out classes.

---

# Part C — Final Exam Checklist

Before the exam, make sure you can do each of the following without notes:

- [ ]Explain why few-shot learning differs from ordinary supervised learning.
- [ ]Distinguish support sets, query sets, episodes, and dataset splits.
- [ ]Calculate support and query counts for an $N$-way, $K$-shot task.
- [ ]Explain the baseline backbone and task-specific classification head.
- [ ]Calculate a Reptile parameter update and interpret its meta-step size.
- [ ]Calculate a prototype from numerical embeddings.
- [ ]Explain why ProtoNet uses negative distances as logits.
- [ ]Explain temperature scaling and query-set cross-entropy.
- [ ]Describe how query loss trains the ProtoNet backbone.
- [ ]Compare the adaptation mechanisms of the baseline, Reptile, and ProtoNet.
- [ ]Diagnose task-local label errors and tensor-shape mismatches.
- [ ]Explain why meta-validation should guide checkpoint selection.

## Exam-writing strategy

For a 2-mark question, provide the correct concept and a concise justification.

For a 3-mark question, explain the mechanism and its purpose.

For a 4–6-mark question, structure your answer around the method's objective, essential steps, mathematical reasoning where appropriate, and the distinction or limitation relevant to the question.

**The central lesson of Practical 6:** The goal is not simply to learn a fixed set of image categories. It is to learn representations or initial parameters that make it possible to recognise new categories from only a few labelled examples.
