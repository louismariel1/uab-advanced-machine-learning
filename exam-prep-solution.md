# Advanced Machine Learning — Stage 2: Solutions and Explanations

*Study mode · Past exam · Autumn 2026 course material*

The goal of this stage is to understand why each answer is correct, how to explain it under exam conditions, and which common misconceptions to avoid. The original exam is worth **35 points**. We will follow the same question order and preserve the original point allocations.

For multiple-choice questions, evaluate each statement independently. For written questions, focus on a clear answer, the underlying mechanism, and the relevant limitations.

---

## Question 1 — Deployment Performance (2 points)

**Correct answer: D only.**

- **A. Insufficient model capacity — Incorrect.** A model that is too simple may underfit, performing poorly on both validation and deployment data. Insufficient capacity alone does not explain why deployment performance is worse than validation performance.
- **B. Insufficient labelled data — Incorrect.** Too little labelled data can lead to poor learning or generalisation, but does not necessarily create a deployment-versus-validation performance gap.
- **C. Overfitting the training split — Incorrect as stated.** Overfitting usually produces a gap between training and validation performance. If deployment data follows the same distribution as validation data, overfitting alone does not necessarily make deployment performance worse than validation performance.
- **D. Deployment distribution differs from training — Correct.** This is a form of **distribution shift**. If deployment data differs from the data used for training and validation, the model may encounter patterns it has not learned to handle.

### Key concept

$$
P_{\mathrm{deployment}}(X,Y)
\neq
P_{\mathrm{validation}}(X,Y)
$$

A change in the input distribution, \(P(X)\), the relationship between inputs and labels, \(P(Y \mid X)\), or both can degrade deployment performance.

**Exam takeaway:** Distinguish underfitting and overfitting from distribution shift. They describe different causes of poor model performance.

---

## Question 2 — Fine-Tuning (2 points)

**Best answer: A only, under the strict wording of the question.**

- **A. Similar source and target domains — True.** Features learned from a related source task are more likely to be useful for the target task, reducing the amount of adaptation required.
- **B. Early layers need more fine-tuning — False.** Early layers often learn general features, while later layers tend to capture more task-specific representations. However, this is a general tendency, not an absolute rule.
- **C. Architecture must remain the same — False as an absolute statement.** The pretrained backbone is often retained, but the output head may need to change when the target task has a different number or type of outputs.
- **D. Learning rate must remain the same — False.** Fine-tuning often uses a smaller learning rate than pretraining to preserve useful pretrained representations and avoid destabilising them.

### Key concept: Transfer learning

A typical workflow is:

1. Pretrain a model on a source task with abundant data.
2. Reuse its learned representations.
3. Replace or adapt the output head if needed.
4. Fine-tune some or all parameters on the target task, often with a reduced learning rate.

**Exam nuance:** If the lecturer uses “architecture” to mean the fixed backbone rather than the complete network including its output head, C could be interpreted differently. The original wording is somewhat ambiguous; the safest explanation distinguishes the retained backbone from a potentially changed task-specific head.

---

## Question 3 — Knowledge Distillation (2 points)

**Correct answers: A and D.**

- **A. Student learns from ground-truth labels and teacher supervision — True.** A common distillation objective combines the ordinary supervised loss with a loss that matches the teacher's softened output distribution.
- **B. Temperature scaling is applied only to the teacher — False.** Standard logit-based distillation typically applies the same temperature to both teacher and student distributions when computing the distillation loss.
- **C. Student must have the same number of layers — False.** The student is often deliberately smaller or computationally cheaper than the teacher.
- **D. Student can inherit teacher biases and vulnerabilities — True.** The student learns from the teacher's outputs and may reproduce its systematic errors, biases, and some adversarial weaknesses.
- **E. Distillation is more effective when the teacher has high confidence on every prediction — False.** A teacher's softened distribution can communicate information about alternative classes. Excessive confidence can suppress this useful information, although teacher quality and calibration matter.

A common objective is:

$$
\mathcal{L} = \alpha \mathcal{L}_{\mathrm{CE}}+(1-\alpha)T^2\mathcal{L}_{\mathrm{KD}}
$$

where:

- \(\mathcal{L}_{\mathrm{CE}}\): supervised cross-entropy against ground-truth labels.
- \(\mathcal{L}_{\mathrm{KD}}\): divergence between the teacher's and student's softened probability distributions, commonly KL divergence.
- \(T\): temperature, typically greater than 1 for softened targets.
- \(\alpha\): balances supervised learning and distillation.

The \(T^2\) factor is a conventional scaling used in the standard formulation; not every implementation uses exactly this objective.

**Exam takeaway:** The teacher transfers more than just its top predicted class. Its relative probabilities across classes can convey useful information, often called *dark knowledge*.

---

## Question 4 — Different Output Dimensions in Knowledge Distillation (3 points)

### Model answer

Yes. Knowledge distillation does not require the teacher and student to have the same number of output neurons in every possible implementation. However, the distillation objective must account for the difference.

For example, if the teacher and student predict different sets of classes, we can:

1. **Map outputs to a shared label space.** Compare probabilities only for corresponding classes.
2. **Use a projection or alignment mechanism.** Transform outputs or intermediate representations into compatible spaces.
3. **Distil intermediate representations or other knowledge.** Match selected hidden features, relations, or task-independent signals rather than directly matching the final output distributions.

A standard KL-divergence loss between teacher and student output distributions requires compatible dimensions and corresponding class meanings. If their output neurons have different meanings and no mapping exists, directly comparing their logits or probability vectors is not meaningful.

**Exam takeaway:** Different output dimensions are possible, but the teacher's knowledge must be represented in a form the student can learn from.

---

## Question 5 — Contrastive Learning (3 points)

The objective is to bring together embeddings of augmented views of the same original image and separate embeddings from different original images.

- **Positive pairs:** \((a,a^*)\) and \((b,b^*)\).
- **Negative pairs:** Every cross-image combination between the \(a\) group and the \(b\) group.

The completed distance matrix is:

| | \(a\) | \(a^*\) | \(b\) | \(b^*\) |
|:---|:---:|:---:|:---:|:---:|
| \(a\) | — | LOW | HIGH | HIGH |
| \(a^*\) | LOW | — | HIGH | HIGH |
| \(b\) | HIGH | HIGH | — | LOW |
| \(b^*\) | HIGH | HIGH | LOW | — |

The diagonal entries are omitted because each embedding's distance from itself is zero.

### Why?

The model should learn representations that remain similar despite appropriate augmentations. At the same time, it should distinguish representations originating from different images.

A simplified contrastive objective can be expressed as:

$$
\mathcal{L} = -\log\frac{\exp\left(s(z_a,z_{a^*})/\tau\right)}{\sum_{k \ne a}\exp\left(s(z_a,z_k)/\tau\right)}
$$

where \(s\) is a similarity measure, \(z\) denotes an embedding, and \(\tau\) is a temperature parameter. This illustrative loss increases the relative similarity of the positive pair compared with other candidates.

**Important distinction:** The exam describes distances, so the required answers are LOW for positive pairs and HIGH for negative pairs. Many contrastive losses are written using similarity instead, in which case the direction is reversed.

**Exam takeaway:** Positive pairs together; negative pairs apart.

---

## Question 6 — Reptile and MAML (7 points)

### (a) Explain \(\theta_M\) and \(\theta_j\) (2 points)

In the Reptile update rule,

$$
\theta_M
\leftarrow
\theta_M+\beta(\theta_j-\theta_M)
$$

- \(\theta_M\) represents the current **meta-parameters**: the shared model initialisation that Reptile is learning.
- \(\theta_j\) represents the parameters obtained after adapting the model to a sampled task \(j\), typically through several ordinary gradient-descent steps on that task's training data.

The meta-update moves the shared initialisation toward the task-adapted parameters.

### (b) Effect of setting \(\beta=1.0\) (2 points)

Substituting \(\beta=1\):

$$
\begin{aligned}
\theta_M'
&= \theta_M+(\theta_j-\theta_M)\\
&= \theta_j
\end{aligned}
$$

The updated meta-parameters become exactly the adapted parameters for task \(j\).

This eliminates interpolation with the previous initialisation for that update. With multiple tasks, the meta-parameters can therefore be strongly affected by the most recently processed task. A value between 0 and 1 instead produces a partial move toward the task-adapted parameters.

### (c) Main difference between Reptile and MAML (3 points)

Both algorithms aim to learn an initialisation that can adapt quickly to a new task using only a few training examples.

Reptile repeatedly adapts a model to sampled tasks and moves the initialisation toward the resulting task-specific parameters. Its outer update does not require explicitly differentiating through the inner optimisation process.

MAML explicitly differentiates the query or meta-loss through the task-specific adaptation steps. This allows the outer update to account for how changing the initial parameters affects the model's performance after adaptation. Standard MAML uses higher-order derivatives, although first-order approximations are also possible.

| Aspect | Reptile | MAML |
|:---|:---|:---|
| Main update | Move toward adapted parameters | Optimise the post-adaptation meta-loss |
| Differentiation through inner updates | Not explicitly required | Explicitly used in standard MAML |
| Higher-order derivatives | Not required by its basic update | Used by standard MAML |
| Goal | Fast adaptation | Fast adaptation |

**Exam takeaway:** Reptile moves toward task-adapted weights; MAML optimises the initialisation based on how well adaptation will work.

---

## Question 7 — Ensemble Learning (7 points)

### (a) Why does averaging five networks improve accuracy? (4 points)

### Model answer

The improvement is explained by ensemble learning. Although the five networks have the same architecture and training data, different initialisations and stochastic training processes can cause them to learn different parameters and make different prediction errors.

Averaging their predictions can reduce variance and cancel out some individual errors, provided their errors are not perfectly correlated.

For classification, an ensemble can average predicted class probabilities and then select the class with the highest average probability. For regression, it can average numerical predictions.

The individual networks achieve approximately 80% test accuracy, whereas the ensemble achieves 85% because their predictions collectively provide a more reliable estimate than an individual network's prediction.

Selecting the single best network does not exploit complementary information from the other four models. Furthermore, the network with the best observed validation or test score is not necessarily the best model on future unseen examples.

### Theoretical intuition

Suppose each model's prediction contains an error component. Averaging independent or weakly correlated errors can reduce their contribution to the final prediction. For example, if the prediction errors are independent and have the same variance \(\sigma^2\), the variance of their average is:

$$
\operatorname{Var}(\bar{\epsilon})
=
\frac{\sigma^2}{n}
$$

for \(n\) models.

With correlated errors, the reduction is smaller. If all models make exactly the same errors, averaging provides no diversity-based benefit.

The accuracy improvement from 80% to 85% is an observed result in the question, not a universal guarantee of ensembling.

### (b) Does averaging work for linear regression and decision trees? (3 points)

Yes. Prediction averaging is not restricted to neural networks.

- **Linear regression:** Average the numerical predictions of multiple models. When the models use the same input features, averaging their predictions is equivalent to averaging their coefficient vectors and intercepts.
- **Decision trees:** Average the numerical predictions of regression trees or the class-probability predictions of classification trees. This is the principle behind random forests and other tree ensembles.

The benefit depends on model diversity and error correlation, not exclusively on the model family. Averaging hard class labels is also possible through majority voting, although it discards the confidence information contained in probability predictions.

**Exam takeaway:** Ensembling can reduce variance and improve generalisation when the models make sufficiently different errors. It does not guarantee better performance in every situation.

---

## Question 8 — Self-Paced Curriculum Learning (4 points)

### Model answer

Self-paced curriculum learning trains a model on easier examples first and gradually introduces more difficult examples as training progresses.

A basic implementation is:

1. Initialise the model and choose an initial difficulty threshold \(\lambda\).
2. For each training example, calculate a difficulty score. A common choice is the example's current loss.
3. Select examples whose loss is below the current threshold.
4. Update the model using only the selected examples.
5. Increase \(\lambda\) gradually so that examples with larger losses become eligible for training.
6. Repeat until training finishes.

### Pseudocode

```text
Initialise model parameters θ
Choose initial threshold λ

For each training epoch:
    Calculate loss ℓᵢ for each training example i
    Select examples where ℓᵢ < λ
    Update θ using only the selected examples
    Increase λ gradually

Return trained model θ
```

For example, an example with a small loss may be considered easy because the model already predicts it reasonably well. An example with a large loss may be considered difficult, noisy, ambiguous, or currently poorly learned.

**Important distinction:** A high loss does not necessarily mean an example is intrinsically difficult. It could be mislabeled or out-of-distribution. A robust implementation may therefore need safeguards against noisy examples.

The threshold \(\lambda\) controls the curriculum: a small threshold initially selects fewer, easier examples, while a larger threshold admits more challenging examples.

**Exam takeaway:** Calculate difficulty, select easy examples, update selectively, and gradually relax the selection threshold.

---

## Question 9 — Self-Distillation (5 points)

### Model answer

Self-distillation can improve student performance even when the teacher and student have the same architecture and use the same training dataset because the student is trained with a different learning signal.

The teacher provides soft probability distributions rather than only hard class labels. These distributions contain information about the relationships between classes, including which alternative classes the teacher considers plausible.

The student can benefit in several ways:

1. **Regularisation:** Soft targets can provide a smoother learning signal than hard labels, discouraging excessively confident predictions and potentially improving generalisation.
2. **Dark knowledge:** The teacher's relative probabilities across classes communicate similarities and distinctions that are not explicit in one-hot labels.
3. **Different optimisation trajectory:** Even with identical architectures and data, the student can converge to different parameters because it is trained using the teacher's predictions in addition to, or instead of, the original labels.
4. **Improved generalisation:** The combination of these effects can help the student learn a representation or decision boundary that performs better on unseen data.

For example, suppose an image belongs to the class *cat*. A hard label assigns probability 1 to cat and 0 to every other class. A teacher might instead assign probabilities of 0.75 to cat, 0.20 to dog, and 0.05 to rabbit. These softer targets communicate that the image shares some characteristics with dogs, giving the student additional information.

**Important qualification:** Self-distillation does not guarantee that the student will outperform the teacher. The improvement depends on the teacher's quality, the distillation objective, optimisation, and generalisation. Nor does distillation create new information from nothing; it can make better use of the training signal through regularisation and a different learning process.

**Exam takeaway:** Same architecture does not mean identical learned parameters or generalisation. Distillation changes the training signal, which can improve the student's performance.

---

## Final Revision Sheet

Use this table to consolidate the key ideas before moving to the interactive exam.

| Question | Core concept | What to remember |
|:---:|:---|:---|
| 1 | Distribution shift | Deployment data can differ from validation data. |
| 2 | Transfer learning | Related domains help; fine-tuning often uses a lower learning rate. |
| 3 | Knowledge distillation | Combine ground-truth supervision and teacher knowledge. |
| 4 | Distillation with different outputs | Align output spaces or use alternative distillation signals. |
| 5 | Contrastive learning | Positive pairs close; negative pairs far apart. |
| 6 | Meta-learning | Reptile moves toward adapted weights; MAML differentiates through adaptation. |
| 7 | Ensemble learning | Diverse errors can cancel through prediction averaging. |
| 8 | Self-paced learning | Select easier examples first and gradually admit harder ones. |
| 9 | Self-distillation | Soft targets can regularise learning and improve generalisation. |

---

## Stage 2 Complete

You now have the model answers and explanations for all nine questions, covering the full 35-point past exam.

The next step is **Stage 3 — Interactive Exam Practice Form**: take the exam again without consulting these solutions, ideally under the original 90-minute time limit. After you submit your answers, I can grade them question by question and identify which concepts need further revision.
