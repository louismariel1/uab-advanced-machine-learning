# Advanced Machine Learning — Knowledge Distillation Practice Exam: Complete Solutions

This solution follows the 35-point practice exam, with answers, explanations, calculations, and marking guidance. The emphasis is on understanding the underlying principles, as expected in your original exam.

## Section A — Multiple choice (10 points)

### Question 1 — Purpose of knowledge distillation (2 points)

Correct answers: A, C, E.

- A — True. The independent student is trained using ordinary supervised learning. It provides a baseline against which we can measure whether distillation improves performance.
- B — False. The teacher and student can have different architectures and different numbers of parameters. The important requirement for conventional output-based distillation is that their output distributions represent compatible classes.
- C — True. The distilled student learns from the ground-truth labels through cross-entropy and from the teacher's predictions through the distillation loss.
- D — False. Distillation is not guaranteed to improve performance. Its effectiveness depends on the teacher's quality, the student architecture, the data, the temperature, and the loss weighting.
- E — True. The teacher is pretrained and kept fixed while the student learns from its predictions. This provides a stable source of supervision.

Key takeaway: The teacher supplies additional training information; the student is the model being optimised.

### Question 2 — Soft labels and temperature (2 points)

Correct answers: A, B, D.

- A — True. Increasing $T$ reduces the differences between logits relative to the temperature, making the softmax distribution more uniform.
- B — True. At $T=1$, temperature-scaled softmax is ordinary softmax.
- C — False. Dividing all logits by a positive temperature preserves their ordering. Therefore, the highest-logit class remains the same, except for ties that were already present.
- D — True. A softer distribution reveals how the teacher distributes probability among alternative classes, giving the student information about class similarities.
- E — False. Softmax probabilities sum to one for any positive temperature, apart from negligible floating-point rounding.

Key takeaway: Temperature changes the distribution's sharpness, not the relative ranking of logits.

### Question 3 — Distillation loss (2 points)

Correct answers: A, B, C, E.

- A — True. PyTorch's standard `F.kl_div()` interface expects log probabilities as its first argument.
- B — True. The teacher's temperature-scaled probabilities form the target distribution.
- C — True. The $T^2$ factor compensates for the reduction in gradient magnitude associated with temperature scaling.
- D — False. KL divergence is non-negative, but it can equal zero when the two distributions are identical.
- E — True. With `reduction='batchmean'`, the summed divergence is divided by the batch size, not by the number of classes.

The practical's loss is

$$
L_d=T^2D_{\mathrm{KL}}(p_t\parallel p_s),
$$

where $p_t$ and $p_s$ are the softened teacher and student distributions, respectively.

Important notation detail: The chapter writes $D_{\mathrm{KL}}(p_s\parallel p_t)$ in its formula, but the supplied PyTorch implementation passes the student log probabilities first and teacher probabilities second. Under PyTorch's standard interface, this computes $D_{\mathrm{KL}}(p_t\parallel p_s)$, not $D_{\mathrm{KL}}(p_s\parallel p_t)$. For the practical, follow the specified implementation and understand the argument order.

### Question 4 — Training and evaluation (2 points)

Correct answers: A, C, D.

- A — True. `torch.no_grad()` prevents the teacher's forward pass from building a computation graph, saving memory and computation.
- B — False. The teacher's parameters should not be included in the student optimizer.
- C — True. Cross-entropy trains the student against the correct labels.
- D — True. `teacher.eval()` disables training-specific behaviours such as dropout during teacher inference.
- E — False. The loss combines the hard-label and soft-label objectives according to $\alpha$. Using only distillation loss corresponds to $\alpha=0$, not every possible setting.

Key takeaway: The teacher generates targets; the student receives gradients and updates its weights.

### Question 5 — Interpreting experimental results (2 points)

Given:

- Independent student: $A_s=65\%$
- Teacher: $A_t=80\%$
- Distilled student: $A_d=72\%$

Correct answers: A, B, D.

- A — True. The improvement is $72-65=7$ percentage points.
- B — True. The original gap is $80-65=15$ percentage points. The student recovers $7/15\approx46.7\%$ of that gap.
- C — False. The distilled student's accuracy is 90% of the teacher's accuracy numerically, but this is not the same as closing 90% of the performance gap.
- D — True. The distilled student improves over the baseline but remains 8 percentage points below the teacher.
- E — False. One experiment cannot establish that distillation always improves performance.

The gap-closed calculation is

$$
\begin{aligned}
\text{Gap closed}
&=\frac{A_d-A_s}{A_t-A_s}\\
&=\frac{72-65}{80-65}\\
&=\frac{7}{15}\approx 46.7\%.
\end{aligned}
$$

## Section B — Short written answers (12 points)

### Question 6 — Teacher and student architectures (3 points)

(a) Why is the teacher expected to achieve higher accuracy? (1 point)

The teacher has more and wider convolutional blocks, giving it greater representational capacity to learn complex patterns. This makes higher accuracy plausible, although it is not guaranteed: optimisation, regularisation, and the amount of training data also affect performance.

(b) Why can their architectures differ? (1 point)

Output-based knowledge distillation transfers knowledge through predictions rather than requiring the teacher's internal layers to match the student's. The two models can therefore have different depths and parameter counts, provided their outputs correspond to compatible classes and can be compared using the distillation loss.

(c) Why train the independent student first? (1 point)

The independent student establishes the performance baseline without teacher guidance. Comparing it with the distilled student allows us to estimate whether distillation provided a benefit under otherwise comparable conditions.

What earns full marks: Capacity difference, compatible outputs rather than identical architectures, and the purpose of the baseline.

### Question 7 — Temperature and probability distributions (4 points)

The logits are

$$
z=[4,2,0].
$$

Temperature-scaled softmax is

$$
p_i=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}.
$$

(a) Which distribution is more concentrated? (1 point)

The distribution with $T=1$ is more concentrated on the highest-scoring class than the distribution with $T=4$. The higher temperature reduces the relative differences between the scaled logits.

For illustration, the approximate distributions are:

| Class logit | $T=1$ | $T=4$ |
| --- | --- | --- |
| 4 | 0.867 | 0.506 |
| 2 | 0.117 | 0.307 |
| 0 | 0.016 | 0.186 |

These values are calculated from the given logits, not measured model outputs.

(b) Why can higher temperature provide useful information? (1 point)

A high temperature makes non-maximal class probabilities more visible. For example, the student can learn that the teacher considers the second class more plausible than the third, instead of receiving almost all its information from the winning class.

(c) What happens as $T$ becomes very large? (1 point)

For a finite set of logits, as $T\to\infty$, the softmax distribution approaches the uniform distribution:

$$
p_i\to\frac{1}{K},
$$

where $K$ is the number of classes. The logits' relative differences become negligible compared with the temperature.

(d) Does temperature scaling force the student to match the teacher at $T=1$? (1 point)

No. Temperature scaling is used to construct the softened distributions for the distillation objective. The student is optimised to match the teacher's softened distribution, while the hard-label cross-entropy supplies a separate objective. The student is not required to reproduce the teacher's exact ordinary-softmax probabilities.

### Question 8 — Combining the two losses (5 points)

The combined objective is

$$
L=\alpha L_h+(1-\alpha)L_d.
$$

(a) What does each loss teach the student? (2 points)

- Hard-label loss $L_h$: Cross-entropy compares the student's predictions with the correct class labels. It directly encourages the student to classify the training examples correctly.
- Distillation loss $L_d$: KL divergence compares the student's softened probability distribution with the teacher's softened distribution. It transfers additional information about the teacher's relative preferences among classes.

Together, these losses combine direct supervision with knowledge extracted from the teacher.

(b) What happens when $\alpha=1$? (1 point)

$$
L=L_h.
$$

The distillation term has zero weight, so the student is trained using ordinary supervised cross-entropy. Assuming the other training settings are unchanged, this is conceptually equivalent to independent supervised training, although random initialisation and execution differences may still affect the result.

(c) What happens when $\alpha=0$? (1 point)

$$
L=L_d.
$$

The hard-label term disappears, and the student learns exclusively from the teacher's softened predictions. It no longer receives a direct cross-entropy objective against the ground-truth labels.

(d) Why does $\alpha=0.5$ not guarantee equal contributions to parameter updates? (1 point)

The coefficient gives each loss a weight of 0.5 in the scalar objective, but the losses can have different numerical magnitudes and gradient magnitudes. The actual parameter update depends on their weighted gradients, their directions, and the optimiser. Therefore, equal coefficients do not necessarily mean equal influence on learning.

## Section C — Algorithm interpretation and application (8 points)

### Question 9 — Identify the algorithm (4 points)

(a) Name the learning technique. (1 point)

The procedure implements teacher–student knowledge distillation. The student is trained using a combination of ground-truth labels and the teacher's predictions.

(b) Why are teacher predictions calculated without tracking gradients? (1 point)

The teacher is already trained and acts as a fixed source of supervision. Using `torch.no_grad()` prevents PyTorch from constructing a computation graph for the teacher's forward pass, reducing memory usage and unnecessary computation. Gradients are still calculated for the student, allowing its parameters to be updated.

(c) What if the teacher is accidentally included in the optimizer? (1 point)

If gradients are also calculated for the teacher and its parameters are updated, the teacher may change during training instead of providing stable targets. This undermines the intended procedure, in which the student learns from a fixed pretrained teacher.

A useful distinction: merely including teacher parameters in an optimizer does not necessarily change them if they have no gradients. In the given procedure, `torch.no_grad()` prevents the teacher's forward pass from producing those gradients, so accidental optimizer inclusion alone does not automatically update the teacher.

(d) What if the teacher has poor accuracy? (1 point)

An inaccurate teacher may provide misleading probability distributions, encouraging the student to imitate incorrect predictions. This can reduce the benefit of distillation or even make the distilled student worse than the independent student. The hard-label loss can counteract some of this effect, but it does not guarantee that the student will improve.

### Question 10 — Diagnose a failed experiment (4 points)

The observed results are:

| Model | Validation accuracy |
| --- | --- |
| Independent student | 68% |
| Teacher | 84% |
| Distilled student | 64% |

(a) Calculate the change in accuracy. (1 point)

$$
\Delta A=A_d-A_s=64\%-68\%=-4
$$

The distilled student performed 4 percentage points worse than the independent student.

(b) Give two plausible reasons. (2 points)

Any two well-explained reasons can earn full marks. For example:

1. Inappropriate temperature: A temperature of 20 may soften the teacher's distribution excessively, making it less informative about class relationships. The resulting distillation signal may not help the student.
2. Inappropriate loss weighting: With $\alpha=0.5$, the distillation loss may exert too much or too little influence relative to cross-entropy. The student's updates may therefore be less effective than ordinary supervised training.

Other valid explanations include a poorly calibrated teacher, noisy or incorrect teacher predictions, optimisation difficulties, an insufficient training budget, or random variation in training.

(c) Suggest one controlled experiment. (1 point)

Keep the dataset split, initialisation protocol, architecture, optimiser, and training duration fixed, and change only one hyperparameter.

For example, compare $T=1, 5, 10, 20$ while keeping $\alpha=0.5$ constant. Measure each run's best validation accuracy and compare it with the 68% independent-student baseline.

If changing temperature improves the result, this suggests that the original temperature was unsuitable. For a reliable conclusion, repeat experiments with multiple random seeds.

Exam tip: A good diagnostic answer proposes a test that isolates a possible cause rather than changing several hyperparameters simultaneously.

## Section D — Theoretical reasoning (5 points)

### Question 11 — Why can a student learn more than the hard labels provide? (5 points)

(a) Explain hard labels versus soft labels. (1 point)

A hard label identifies the correct class. In this example, it identifies horse as the true class, represented by a one-hot vector with probability 1 for horse and 0 for every other class.

A soft label is a probability distribution produced by the teacher. It assigns 0.75 to horse, 0.10 to deer, 0.07 to dog, and smaller probabilities to the remaining classes.

(b) Why does the teacher provide additional information? (2 points)

The hard label tells the student which class is correct but does not distinguish between alternative incorrect classes. From the hard label alone, the student cannot tell whether the teacher considers deer more similar to horses than trucks are.

The teacher's soft labels encode its learned relationships between classes. In this example, the relatively high probability assigned to deer and dog suggests that these classes share some visual characteristics with horses. The student can use this information to learn a richer representation of the classification problem.

These relative probabilities are often described as dark knowledge.

(c) How do the two objectives complement one another? (1 point)

Cross-entropy encourages the student to predict the ground-truth class correctly, while the distillation loss encourages it to reproduce the teacher's softened probability distribution. Combining them balances direct supervision with the teacher's learned class relationships.

(d) How can the student outperform the teacher despite having the same architecture? (1 point)

The student is not necessarily limited to reproducing the teacher's mistakes. Its training objective combines teacher predictions with ground-truth labels, and optimisation may lead it to a solution that generalises better to the validation set. The different training signal can act as a form of regularisation, while random initialisation and stochastic optimisation can also lead to different solutions.

Therefore, a student can outperform its teacher even when both share the same architecture and training data, although this outcome is not guaranteed.

## Consolidated marking guide

Section A — Multiple choice

## 10 points

Section B — Short written answers

## 12 points

Section C — Algorithm interpretation

## 8 points

Section D — Theoretical reasoning

## 5 points

## Total

# 35 points

## Final revision: essential facts to memorise

| Concept | Essential fact |
| --- | --- |
| Hard labels | Ground-truth class targets |
| Soft labels | Teacher's predicted probability distribution |
| Temperature | Higher $T$ generally softens the distribution |
| KL divergence | Measures the difference between probability distributions |
| Temperature correction | Distillation loss includes $T^2$ |
| Combined loss | $L=\alpha L_h+(1-\alpha)L_d$ |
| $\alpha=1$ | Hard-label loss only |
| $\alpha=0$ | Distillation loss only |
| Teacher | Fixed, evaluated without gradients |
| Student | Trained through backpropagation |
| Distillation benefit | Potentially better student generalisation, not guaranteed |
| Gap closed | $(A_d-A_s)/(A_t-A_s)$ |

### Three particularly important exam traps

1. Do not confuse probability distributions with logits. Temperature is applied to logits before softmax.
2. Do not reverse the KL-divergence argument order. The supplied PyTorch implementation computes $D_{\mathrm{KL}}(p_t\parallel p_s)$ under the standard interface, despite the reversed notation in the chapter's displayed formula.
3. Do not claim that distillation always improves performance. It is an empirical outcome that must be established by comparing validation performance against the independent baseline.

If you can explain the reasoning behind these answers without relying on the solutions, you will have a strong foundation for knowledge-distillation questions in the style of the original Advanced Machine Learning exam.
