# Advanced Machine Learning — MCQ Practice Bank

Topic: Knowledge Distillation and Teacher–Student Learning Based on: Notebook chunks 1 and 2 Format: Multiple-choice questions with answers and explanations Difficulty: Intermediate to advanced

This practice bank follows the structure of your Few-Shot Learning and Meta-Learning example. It covers the main concepts, mathematical calculations, PyTorch implementation, tensor shapes, debugging, training procedures, and experimental design required for the knowledge distillation assignment.

The questions are designed to test both conceptual understanding and your ability to apply the ideas in exam-style scenarios.

## Part A — Fundamentals of Knowledge Distillation

### Question 1. What is the primary goal of knowledge distillation?

Which statement is correct?

- A. To increase the number of parameters in the student model.
- B. To train a smaller student model to learn from a larger or more capable teacher model.
- C. To replace all training labels with randomly generated labels.
- D. To ensure that the student model has exactly the same architecture as the teacher.

Correct answer: B

Explanation:

Knowledge distillation transfers information from a teacher model to a student model. The teacher provides additional training signals, often through its predicted probability distribution over classes.

The goal is to help the student achieve better performance than it would through independent training alone, while retaining the advantages of a smaller or more efficient model.

- A is incorrect because distillation does not require increasing the model size.
- C is incorrect because the teacher's predictions are learned outputs, not random labels.
- D is incorrect because teacher and student models can have different architectures.

Exam tip: Remember the direction of knowledge transfer: the teacher guides the student during training.

### Question 2. Teacher versus student models

A large neural network achieves high classification accuracy but requires substantial computational resources. A smaller network is easier to deploy but has lower accuracy when trained independently.

Which statements are correct? Select all that apply.

- A. The large network can serve as the teacher.
- B. The smaller network can serve as the student.
- C. The teacher must be updated whenever the student is updated.
- D. The student can learn from the teacher's probability distributions.
- E. The student must use exactly the same number of layers as the teacher.

Correct answers: A, B and D

Explanation:

The teacher provides guidance, and the student learns from that guidance.

During standard offline distillation:

- The teacher is pretrained and generally kept fixed.
- The student is trained using the teacher's outputs and usually the original ground-truth labels.
- The student can use a different architecture from the teacher.

Updating the teacher alongside the student is not required and would change the standard distillation setup.

### Question 3. Independent student training versus distillation

A student model trained independently achieves 70% validation accuracy. After knowledge distillation, the same student architecture achieves 76%.

Which conclusion is best supported?

- A. Distillation has improved the student's validation accuracy by six percentage points.
- B. Distillation has improved the student's accuracy by 6% relative to every possible baseline.
- C. The teacher must have 100% accuracy.
- D. The student has necessarily become larger.

Correct answer: A

Explanation:

The absolute improvement is

$$
76\%-70\%=\boxed{6\text{ percentage points}}.
$$

The relative improvement is different:

$$
\frac{76-70}{70}\times100
\approx 8.57\%.
$$

Therefore, the student's accuracy improves by six percentage points, equivalent to approximately an 8.57% relative increase over its baseline accuracy.

Distillation does not guarantee that the student will outperform the teacher or that every experiment will show an improvement.

### Question 4. What information does the teacher provide?

Suppose an image belongs to the class horse. The teacher predicts the following probabilities:

| Class | Probability |
| --- | --- |
| Horse | 0.80 |
| Dog | 0.10 |
| Bird | 0.07 |
| Ship | 0.02 |
| Truck | 0.01 |

What additional information does this distribution provide compared with a hard label stating only horse?

- A. It provides only the correct class and no other information.
- B. It reveals the teacher's relative preferences among incorrect classes.
- C. It guarantees that the teacher is correct.
- D. It eliminates the need to train the student.

Correct answer: B

Explanation:

A hard label identifies the target class but does not express how the model relates the target to other classes.

The teacher's distribution indicates that dog and bird are more plausible alternatives than ship and truck. These relative similarities provide additional information that the student can learn from.

This information is often called dark knowledge: useful structure encoded in the teacher's output probabilities, particularly among the non-target classes.

The teacher's predictions are not guaranteed to be correct, and the student still requires training.

### Question 5. What are soft labels?

Which statement best describes soft labels in knowledge distillation?

- A. Binary labels that can only be zero or one.
- B. Probabilistic targets that assign values to multiple classes.
- C. Labels that contain no information about the target class.
- D. Ground-truth labels that have been randomly shuffled.

Correct answer: B

Explanation:

Soft labels are probability distributions over classes. Unlike one-hot hard labels, they can assign nonzero probabilities to multiple classes.

For example, a hard label for three classes might be

$$
y=[1,0,0],
$$

whereas a soft label might be

$$
y=[0.75,0.20,0.05].
$$

The second distribution communicates that the first class is preferred, but the second class is more plausible than the third.

Soft labels can provide a richer learning signal than hard labels alone.

### Question 6. Why can a teacher's predictions be more informative than hard labels?

A teacher predicts that an image is 95% likely to be a cat, 4% likely to be a dog and 1% likely to be a truck. The ground-truth label identifies only the cat class.

Which interpretation is correct?

- A. The teacher distribution contains information about similarities between classes that the hard label does not explicitly represent.
- B. The ground-truth label is invalid because it contains only one nonzero entry.
- C. The student should predict the truck class because it has a nonzero probability.
- D. Soft labels are always more accurate than hard labels.

Correct answer: A

Explanation:

The teacher's output expresses relative uncertainty. In this example, it considers dog a more plausible alternative to cat than truck.

This provides the student with information about the teacher's learned class boundaries.

Soft labels are not always more accurate than hard labels, because the teacher itself can make mistakes. Their value comes from providing additional structure, not from guaranteeing correctness.

## Part B — Softmax, Temperature Scaling and Probability Distributions

### Question 7. Understanding softmax

For a vector of logits $z$, which formula correctly defines the softmax probability for class $i$?

- A. $\displaystyle p_i=\frac{z_i}{\sum_j z_j}$
- B. $\displaystyle p_i=\frac{\exp(z_i)}{\sum_j\exp(z_j)}$
- C. $\displaystyle p_i=\exp\left(\sum_jz_j\right)$
- D. $\displaystyle p_i=\frac{1}{z_i}$

Correct answer: B

Explanation:

Softmax converts logits into a probability distribution:

$$
p_i=\frac{\exp(z_i)}{\sum_j\exp(z_j)}.
$$

Exponentiation makes the terms positive, and division by their sum ensures that the probabilities add to one:

$$
\sum_i p_i=1.
$$

Softmax preserves the ranking of logits: the class with the largest logit also has the largest probability.

### Question 8. Calculating softmax probabilities

A model produces two logits:

$$
z=[0,\ln 3].
$$

What are the softmax probabilities?

- A. $[0.25,0.75]$
- B. $[0.50,0.50]$
- C. $[0.75,0.25]$
- D. $[0,1]$

Correct answer: A — $[0.25,0.75]$

Explanation:

First exponentiate the logits:

$$
\exp(z)=[1,3].
$$

The sum is $1+3=4$. Therefore,

$$
p=\left[\frac14,\frac34\right].
$$

Thus,

$$
\boxed{p=[0.25,0.75]}.
$$

The second class receives the larger probability because its logit is larger.

### Question 9. What is temperature scaling?

Which formula correctly defines temperature-scaled softmax?

- A. $\displaystyle p_i=\frac{\exp(Tz_i)}{\sum_j\exp(Tz_j)}$
- B. $\displaystyle p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}$
- C. $\displaystyle p_i=\frac{z_i+T}{\sum_jz_j}$
- D. $\displaystyle p_i=\frac{\exp(z_i)}{T}$

Correct answer: B

Explanation:

Temperature scaling divides each logit by a positive temperature $T$ before applying softmax:

$$
p_i(T)=
\frac{\exp(z_i/T)}
{\sum_j\exp(z_j/T)}.
$$

When $T=1$, the formula becomes ordinary softmax.

When $T>1$, the relative differences between logits are reduced, usually producing a softer distribution.

When $0<T<1$, those differences become more pronounced, usually producing a sharper distribution.

Exam tip: Temperature divides the logits. It does not multiply them.

### Question 10. Effect of increasing temperature

A teacher produces logits

$$
z=[5,2,0].
$$

What generally happens when the temperature increases from $T=1$ to $T=10$?

- A. The largest-logit class changes automatically.
- B. The probability distribution becomes more uniform.
- C. All probabilities except the largest become exactly zero.
- D. The logits become the ground-truth labels.

Correct answer: B

Explanation:

Dividing logits by a larger temperature reduces their differences before softmax.

At $T=1$, the difference between the first and second logits is 3. At $T=10$, it becomes 0.3.

The resulting probabilities are generally less concentrated on the highest-logit class.

As the temperature approaches infinity, the distribution approaches a uniform distribution over the classes, assuming a finite number of classes and finite logits.

For any finite positive temperature, the ordering of logits remains unchanged.

### Question 11. Temperature-scaled softmax calculation

A model has logits

$$
z=[4,2,0].
$$

If the temperature is $T=2$, what logits are passed into the ordinary softmax operation?

- A. $[8,4,0]$
- B. $[4,2,0]$
- C. $[2,1,0]$
- D. $[0.5,0.25,0]$

Correct answer: C — $[2,1,0]$

Explanation:

Temperature scaling divides each logit by $T$:

$$
z_T=\frac{z}{T}.
$$

Therefore,

$$
z_T=\left[\frac42,\frac22,\frac02\right]
=\boxed{[2,1,0]}.
$$

Softmax is then applied to the scaled logits.

Option A multiplies the logits by temperature, which makes the distribution sharper rather than softer.

### Question 12. Why use a large temperature during distillation?

In the notebook, a temperature such as $T=20$ is used to produce softer teacher outputs. What is the main reason?

- A. To force the teacher to predict the wrong class.
- B. To expose more information about the relative probabilities of non-target classes.
- C. To remove all uncertainty from the teacher's predictions.
- D. To increase the number of classes in the dataset.

Correct answer: B

Explanation:

A confident teacher can assign nearly all probability to one class. The remaining class probabilities may be so small that little information is available about the teacher's preferences among incorrect classes.

Increasing the temperature softens the distribution, making relative differences between classes easier to observe.

This can give the student a richer signal about the teacher's learned class relationships.

A large temperature is not always optimal. Its usefulness depends on the teacher's logits, the dataset, and the student model.

### Question 13. Comparing hard and soft distributions

A teacher's ordinary softmax output is

$$
p=[0.99,0.009,0.001].
$$

After applying a higher temperature, its output becomes approximately

$$
q=[0.60,0.25,0.15].
$$

Which interpretation is correct?

- A. The temperature has changed the ground-truth class.
- B. The softer distribution reveals more of the relative structure among classes.
- C. The softer distribution must be incorrect because it is less confident.
- D. Temperature scaling changes the model's weights directly.

Correct answer: B

Explanation:

Temperature scaling changes the probability distribution computed from the logits. It does not directly modify the model parameters or ground-truth labels.

The ordinary distribution is dominated by the first class. The temperature-scaled distribution assigns more visible probability to the alternatives, revealing how the teacher distributes its preference across classes.

This is one of the central motivations for using temperature in knowledge distillation.

### Question 14. Implementing temperature softmax in PyTorch

Which implementation correctly computes temperature-scaled softmax for an input tensor `x` with shape `(batch, num_classes)`?

- A. `torch.softmax(x * T, dim=1)`
- B. `torch.softmax(x / T, dim=1)`
- C. `torch.softmax(x, dim=0) / T`
- D. `torch.log_softmax(x + T, dim=1)`

Correct answer: B

Explanation:

The notebook's `temperature_softmax` function should rescale the logits by temperature and then apply softmax over the class dimension.

python
```
def temperature_softmax(x, T):
    return torch.softmax(x / T, dim=1)
```

The input shape is `(batch, num_classes)`.

- `dim=1` applies softmax across the classes for each example.
- `dim=0` would normalise across examples in the batch instead.
- Multiplying by $T$ sharpens the distribution when $T>1$.
- `log_softmax` returns log-probabilities, not ordinary probabilities.

For valid temperature scaling, $T$ must be positive.

## Part C — Distillation Loss and KL Divergence

### Question 15. What is KL divergence?

Which statement best describes Kullback–Leibler (KL) divergence?

- A. It measures the number of parameters in a neural network.
- B. It measures the difference between probability distributions.
- C. It measures classification accuracy directly.
- D. It is always symmetric in its two arguments.

Correct answer: B

Explanation:

KL divergence measures how one probability distribution differs from another.

For distributions $P$ and $Q$,

$$
D_{\mathrm{KL}}(P\parallel Q) = \sum_i P(i)\log\frac{P(i)}{Q(i)}.
$$

In knowledge distillation, the teacher's temperature-scaled output acts as the target distribution, while the student's temperature-scaled output is the distribution being trained to match.

KL divergence is not symmetric:

$$
D_{\mathrm{KL}}(P\parallel Q)
\ne D_{\mathrm{KL}}(Q\parallel P)
$$

in general.

### Question 16. Which KL divergence direction is appropriate?

Let $p_t$ be the teacher distribution and $p_s$ be the student distribution. Which expression represents the standard teacher-to-student distillation objective?

- A. $D_{\mathrm{KL}}(p_s\parallel p_t)$
- B. $D_{\mathrm{KL}}(p_t\parallel p_s)$
- C. $D_{\mathrm{KL}}(p_t\parallel p_t)$
- D. $D_{\mathrm{KL}}(p_s\parallel p_s)$

Correct answer: B

Explanation:

The standard distillation loss uses the teacher as the target distribution:

$$
L_d=T^2D_{\mathrm{KL}}(p_t^{(T)}
\parallel p_s^{(T)}).
$$

Here, $p_t^{(T)}$ and $p_s^{(T)}$ denote the teacher's and student's temperature-scaled softmax distributions.

The teacher distribution appears first because the student is being trained to approximate it.

The notebook describes the KL divergence between student and teacher outputs. When implementing that objective using PyTorch's `kl_div`, it is important to follow PyTorch's argument convention: the first argument is the student's log-probabilities, and the second argument is the teacher's target probabilities.

This distinction prevents an accidental reversal of the KL divergence.

### Question 17. Calculating a simple KL divergence

Suppose the teacher distribution is

$$
P=[0.5,0.5]
$$

and the student distribution is

$$
Q=[0.5,0.5].
$$

What is $D_{\mathrm{KL}}(P\parallel Q)$?

- A. 0
- B. 0.5
- C. 1
- D. 2

Correct answer: A — 0

Explanation:

Using the KL divergence formula,

$$
D_{\mathrm{KL}}(P\parallel Q)
=\sum_i P(i)\log\frac{P(i)}{Q(i)}.
$$

Since $P(i)=Q(i)$ for every class,

$$
\log\frac{P(i)}{Q(i)}=\log 1=0.
$$

Therefore,

$$
\boxed{D_{\mathrm{KL}}(P\parallel Q)=0}.
$$

KL divergence is nonnegative for valid probability distributions and equals zero when the two distributions are identical.

### Question 18. Calculating KL divergence for different distributions

Let

$$
P=[0.75,0.25],\qquad Q=[0.5,0.5].
$$

What is $D_{\mathrm{KL}}(P\parallel Q)$, using natural logarithms?

- A. 0
- B. Approximately 0.131
- C. Approximately 0.693
- D. Approximately 1.386

Correct answer: B — approximately 0.131

Explanation:

Substitute into the formula:

$$
\begin{aligned}
D_{\mathrm{KL}}(P\parallel Q)
&=0.75\ln\frac{0.75}{0.5}
+0.25\ln\frac{0.25}{0.5}\\
&=0.75\ln(1.5)+0.25\ln(0.5)\\
&\approx0.3041-0.1733\\
&\approx\boxed{0.131}.
\end{aligned}
$$

The distributions differ, so the KL divergence is positive.

### Question 19. Why multiply the distillation loss by $T^2$?

The notebook defines the distillation loss as

$$
L_d=T^2D_{\mathrm{KL}}
\left(p_t^{(T)}\parallel p_s^{(T)}\right).
$$

Why is the $T^2$ factor included?

- A. To increase the number of training examples.
- B. To compensate approximately for the gradient scaling introduced by temperature.
- C. To make the KL divergence negative.
- D. To ensure the teacher's weights are updated.

Correct answer: B

Explanation:

Temperature scaling softens the distributions, but it also changes the scale of the gradients used to train the student.

In the standard distillation formulation, the gradients from the soft-target term can shrink approximately with a factor proportional to $1/T^2$ at sufficiently high temperatures.

Multiplying the KL term by $T^2$ compensates for this scaling.

The corrected loss is

$$
\boxed{L_d=T^2D_{\mathrm{KL}}
(p_t^{(T)}\parallel p_s^{(T)})}.
$$

This is a gradient-scaling correction, not a guarantee that every temperature will yield better validation accuracy.

### Question 20. Implementing the distillation loss in PyTorch

Which implementation correctly follows the standard teacher-to-student KL objective?

- A. `F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * T**2`
- B. `F.kl_div(teacher_probs, student_log_probs, reduction='sum')`
- C. `F.cross_entropy(student_probs, teacher_probs) * T`
- D. `torch.mean(student_logits - teacher_logits)`

Correct answer: A

Explanation:

A suitable implementation is:

python
```
import torch
import torch.nn.functional as F

def distillation_output_loss(student_pred, teacher_pred, temperature):
    student_log_probs = F.log_softmax(
        student_pred / temperature, dim=1
    )
    teacher_probs = F.softmax(
        teacher_pred / temperature, dim=1
    )

    loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    )

    return loss * (temperature ** 2)
```

The key implementation details are:

- Apply temperature scaling to both sets of logits.
- Use student log-probabilities as the first argument to `F.kl_div`.
- Use teacher probabilities as the target distribution.
- Use `reduction="batchmean"` to sum over classes and average over the batch.
- Multiply by $T^2$.

The teacher distribution is the target, and the student distribution is what the optimisation changes.

### Question 21. Why use `reduction="batchmean"`?

A batch contains 16 examples and 10 classes. Why is `reduction="batchmean"` appropriate for the distillation loss?

- A. It divides the total KL divergence by the number of classes only.
- B. It averages the summed KL divergence across examples in the batch.
- C. It averages only the teacher logits.
- D. It removes the need for temperature scaling.

Correct answer: B

Explanation:

For each example, KL divergence sums the contributions from the classes. The batch loss then averages those per-example divergences over the batch.

Conceptually,

$$
L=\frac1B\sum_{b=1}^{B}
\sum_{i=1}^{C}
p_{t,b,i}
\log\frac{p_{t,b,i}}{p_{s,b,i}}.
$$

Here, $B$ is the batch size and $C$ is the number of classes.

`reduction="batchmean"` implements the summed class divergence divided by the batch size.

By contrast, `reduction="mean"` in PyTorch's KL-divergence function averages over all tensor elements, including the class dimension. That changes the scale of the loss.

### Question 22. Understanding the loss tensor shape

A student and teacher each produce logits with shape `(32, 10)`. After calculating the distillation loss using `reduction="batchmean"`, what should the loss tensor's shape be?

- A. `(32, 10)`
- B. `(32,)`
- C. `(10,)`
- D. `()`

Correct answer: D — scalar shape `()`

Explanation:

The KL divergence sums over classes and averages over the batch. Therefore, the result is a single scalar representing the batch loss.

In PyTorch:

python
```
loss.shape
```

should return

python
```
torch.Size([])
```

This is different from a tensor of shape `(1,)`, which contains one element but has one dimension.

A scalar loss is appropriate for ordinary backpropagation:

python
```
loss.backward()
```

### Question 23. Why must the student's and teacher's class dimensions match?

The student outputs logits of shape `(32, 10)`, while the teacher outputs logits of shape `(32, 100)`. What is the most important issue when calculating a direct KL distillation loss?

- A. The batch size is too large.
- B. The teacher and student distributions cover different numbers of classes and cannot be directly compared element by element.
- C. KL divergence only supports binary classification.
- D. The student must always have 100 layers.

Correct answer: B

Explanation:

A direct elementwise KL comparison requires corresponding probabilities to refer to the same classes.

Here, the teacher has 100 class outputs, while the student has only 10. Their probability distributions are therefore not aligned.

Possible solutions include using teacher and student models with the same class vocabulary or designing an appropriate mapping or distillation objective for the different output spaces.

Matching the batch sizes alone is not enough.

### Question 24. What if the teacher and student predictions are identical?

Suppose the student and teacher produce identical logits for every example, and both use the same temperature. What should the ideal distillation loss be?

- A. 0
- B. 1
- C. $T$
- D. $T^2$

Correct answer: A — 0

Explanation:

Identical logits produce identical temperature-scaled softmax distributions.

The KL divergence between identical distributions is zero. Multiplying zero by $T^2$ still gives zero:

$$
L_d=T^2\times0=\boxed{0}.
$$

In a numerical implementation, tiny rounding errors may produce a value extremely close to zero rather than exactly zero.

## Part D — Hard Loss, Soft Loss and Combined Training

### Question 25. What is the hard loss?

In the notebook, the hard loss is calculated using

python
```
hard_loss = nn.CrossEntropyLoss()(
    student_pred, labels
)
```

What does this loss measure?

- A. The difference between the teacher's and student's hidden-layer activations.
- B. The student's classification error relative to the ground-truth class labels.
- C. The number of incorrect classes predicted by the teacher.
- D. The difference between the model parameter counts.

Correct answer: B

Explanation:

Cross-entropy compares the student's predicted class distribution with the ground-truth labels.

For a single example with true class $y$, its loss is

$$
L_h=-\log p_s(y).
$$

If the student assigns a high probability to the true class, the loss is small. If it assigns a low probability to the true class, the loss is large.

In PyTorch, `nn.CrossEntropyLoss` expects raw logits and integer class labels. It applies the necessary log-softmax computation internally.

### Question 26. Why combine hard and soft losses?

Why does the notebook recommend training the student with both the hard loss and the distillation loss?

- A. The hard loss teaches ground-truth correctness, while the soft loss transfers the teacher's class relationships.
- B. The hard loss updates the teacher, while the soft loss updates the student.
- C. The two losses always have opposite signs and cancel each other.
- D. The soft loss makes the ground-truth labels unnecessary in every setting.

Correct answer: A

Explanation:

The two objectives provide complementary signals.

- Hard loss: Encourages the student to predict the correct ground-truth class.
- Soft loss: Encourages the student to reproduce the teacher's temperature-scaled distribution.

A common combined objective is

$$
L=(1-\alpha)L_h+\alpha L_d,
$$

where $\alpha$ controls the strength of distillation.

This formulation is one standard approach, although some notebooks use different conventions for the weighting parameter.

### Question 27. Calculating the combined loss

Suppose

$$
L_h=0.8,\qquad L_d=0.4,\qquad \alpha=0.25.
$$

Using

$$
L=(1-\alpha)L_h+\alpha L_d,
$$

what is the combined loss?

- A. 0.10
- B. 0.40
- C. 0.70
- D. 1.20

Correct answer: C — 0.70

Explanation:

Substitute the values:

$$
\begin{aligned}
L
&=(1-0.25)(0.8)+0.25(0.4)\\
&=0.75(0.8)+0.25(0.4)\\
&=0.60+0.10\\
&=\boxed{0.70}.
\end{aligned}
$$

The hard loss contributes 75% of the combined objective, while the soft loss contributes 25%.

### Question 28. Interpreting the weighting parameter

For the objective

$$
L=(1-\alpha)L_h+\alpha L_d,
$$

which statements are correct? Select all that apply.

- A. If $\alpha=0$, the student uses only the hard loss.
- B. If $\alpha=1$, the student uses only the soft loss.
- C. Increasing $\alpha$ increases the relative contribution of the distillation loss.
- D. The teacher's weights are automatically updated when $\alpha$ increases.
- E. A larger $\alpha$ always improves student accuracy.

Correct answers: A, B and C

Explanation:

The weighting parameter controls the balance between the two objectives.

- At $\alpha=0$, the loss is $L=L_h$.
- At $\alpha=1$, the loss is $L=L_d$.
- Intermediate values combine the two terms.

Changing $\alpha$ does not automatically update the teacher. Nor is a larger value guaranteed to improve accuracy: too much emphasis on teacher predictions can be harmful if the teacher's errors or biases dominate.

Exam tip: Distinguish between a hyperparameter that changes the objective and a mechanism that changes which model parameters receive gradients.

### Question 29. Summing versus averaging losses

A notebook adds hard and soft losses directly:

$$
L=L_h+L_d.
$$

Both losses are approximately 0.8. What is the combined loss, and what is a potential concern?

- A. The loss is 0.8, and the gradients must be zero.
- B. The loss is 1.6, and the gradient scale may be larger than when using either loss alone.
- C. The loss is 0.0, because the terms cancel.
- D. The loss is 2.0, and the teacher is necessarily updated.

Correct answer: B — 1.6

Explanation:

The combined loss is

$$
L=0.8+0.8=\boxed{1.6}.
$$

Adding two loss terms adds their gradients as well. This can increase the magnitude of the parameter updates.

One alternative is to average the losses:

$$
L=\frac{L_h+L_d}{2}.
$$

Another is to use a weighted combination. Either approach may require tuning the learning rate and loss weights.

The important point is to compare training setups fairly and understand how changes to the loss affect optimisation.

### Question 30. Which model should receive gradients?

In standard offline distillation, which model should have its parameters updated by the optimiser?

- A. The teacher only.
- B. The student only.
- C. Both teacher and student must always be updated.
- D. Neither model.

Correct answer: B

Explanation:

The teacher is pretrained and provides target distributions. The student learns to match those distributions and the ground-truth labels.

The optimiser should therefore update the student parameters only.

A typical setup is:

python
```
teacher.eval()
teacher.requires_grad_(False)

student.train()
optimizer = torch.optim.Adam(
    student.parameters(), lr=learning_rate
)
```

The teacher's forward pass can also be wrapped in `torch.no_grad()` to avoid constructing a gradient computation graph.

This reduces unnecessary computation and prevents teacher gradients from being accumulated.

## Part E — Training Loop, PyTorch and Debugging

### Question 31. Why use `teacher.eval()` during distillation?

The teacher contains BatchNorm layers. Why should it generally be placed in evaluation mode during student training?

- A. To make the teacher's predictions stable by using its running BatchNorm statistics.
- B. To force the teacher to predict the same class for every input.
- C. To disable all teacher forward passes.
- D. To increase the student's number of channels.

Correct answer: A

Explanation:

BatchNorm behaves differently in training and evaluation modes.

In training mode, BatchNorm uses batch statistics and updates its running estimates. In evaluation mode, it uses its stored running statistics.

The teacher is supposed to provide a stable target distribution. Calling

python
```
teacher.eval()
```

helps ensure that its outputs do not change because of unintended BatchNorm updates during distillation.

Evaluation mode does not disable the forward pass or make all predictions identical.

### Question 32. Why use `torch.no_grad()` for the teacher?

Consider this code:

python
```
with torch.no_grad():
    teacher_pred = teacher(inputs)

student_pred = student(inputs)
```

Why is `torch.no_grad()` useful here?

- A. It prevents the student from learning.
- B. It avoids recording gradients for the teacher's forward computation.
- C. It removes the teacher's learned weights.
- D. It automatically computes the KL divergence.

Correct answer: B

Explanation:

The teacher's predictions are targets, not parameters to optimise. Recording gradients through the teacher is unnecessary in ordinary offline distillation.

Using `torch.no_grad()` reduces memory usage and computational overhead.

The student forward pass is outside the context manager, so gradients can still be computed for the student.

This is different from wrapping the entire training step in `torch.no_grad()`, which would prevent the student loss from building the graph needed for backpropagation.

### Question 33. Debugging a temperature-softmax implementation

A student writes:

python
```
def temperature_softmax(x, T):
    return torch.softmax(x / T, dim=0)
```

The input shape is `(batch, num_classes)`. What is the main bug?

- A. The input should always be a NumPy array.
- B. Softmax is normalising across examples instead of classes.
- C. Temperature must be an integer.
- D. Softmax cannot accept a two-dimensional tensor.

Correct answer: B

Explanation:

For a batch of classification logits, the class dimension is `dim=1`.

The corrected implementation is:

python
```
def temperature_softmax(x, T):
    return torch.softmax(x / T, dim=1)
```

Using `dim=0` normalises each class across the batch rather than normalising each example across its classes.

Consequently, the resulting rows generally do not form valid per-example probability distributions.

### Question 34. Debugging a reversed KL-divergence implementation

A student writes:

python
```
loss = F.kl_div(
    F.softmax(student_pred / T, dim=1),
    F.log_softmax(teacher_pred / T, dim=1),
    reduction="batchmean"
)
```

What is the main issue?

- A. The first argument should be the student's log-probabilities, and the target should be the teacher's probabilities.
- B. KL divergence cannot be used for distillation.
- C. The batch size must equal the number of classes.
- D. The teacher must be in training mode.

Correct answer: A

Explanation:

PyTorch's `F.kl_div` convention expects the first argument to contain log-probabilities. By default, its second argument contains target probabilities.

The correct pattern is:

python
```
student_log_probs = F.log_softmax(
    student_pred / T, dim=1
)
teacher_probs = F.softmax(
    teacher_pred / T, dim=1
)

loss = F.kl_div(
    student_log_probs,
    teacher_probs,
    reduction="batchmean"
) * T**2
```

Using probabilities as the first argument and log-probabilities as the second argument is incorrect for the default convention.

### Question 35. Debugging an unexpectedly large loss

A student implements distillation using:

python
```
loss = F.kl_div(
    F.log_softmax(student_pred / T, dim=1),
    F.softmax(teacher_pred / T, dim=1),
    reduction="sum"
)
```

The loss increases substantially when the batch size is doubled, even though the average prediction quality appears similar.

What is the most likely explanation?

- A. The loss uses `reduction="sum"` instead of `reduction="batchmean"`.
- B. Softmax is unavailable for large batches.
- C. The teacher has too few parameters.
- D. The model's class labels must be floating-point values.

Correct answer: A

Explanation:

`reduction="sum"` sums the KL divergence over all examples and classes. If the average per-example divergence stays similar while the batch size doubles, the total loss will generally increase.

Using

python
```
reduction="batchmean"
```

divides the summed divergence by the batch size, making the loss an average over examples.

This is the reduction specified in the notebook assignment.

### Question 36. Training mode versus evaluation mode

A student model contains BatchNorm layers. The evaluation function calls `model.eval()` at the end of every epoch, but the next training epoch begins without restoring training mode.

What should be done?

- A. Call `student.train()` at the beginning of every training epoch.
- B. Keep the student in evaluation mode for all training.
- C. Delete every BatchNorm layer.
- D. Call `teacher.train()` to restore the student.

Correct answer: A

Explanation:

`model.eval()` changes the behaviour of layers such as BatchNorm and Dropout. It does not disable gradients, but it changes the model's forward-pass behaviour.

Before training an epoch, call:

python
```
student.train()
teacher.eval()
```

This ensures that the student uses training-mode behaviour while the teacher provides stable evaluation-mode targets.

The notebook explicitly warns that failing to manage these modes correctly can harm distillation performance.

### Question 37. Why freeze the teacher's parameters?

Which implementation correctly prevents the teacher's parameters from being updated by the optimiser?

- A. `optimizer = torch.optim.Adam(teacher.parameters())`
- B. `optimizer = torch.optim.Adam(student.parameters())`
- C. `optimizer = torch.optim.Adam(list(student.parameters()) + list(teacher.parameters()))`
- D. `optimizer = torch.optim.Adam(inputs)`

Correct answer: B

Explanation:

The optimiser should receive the student parameters only:

python
```
optimizer = torch.optim.Adam(
    student.parameters(), lr=learning_rate
)
```

The teacher may additionally be frozen with:

python
```
teacher.requires_grad_(False)
```

and evaluated inside `torch.no_grad()`.

Including teacher parameters in the optimiser is unnecessary in standard offline distillation and can result in unintended updates if gradients are enabled.

### Question 38. What should happen during one training batch?

Which sequence best describes a standard knowledge distillation training step?

- A. Update the teacher → generate random labels → calculate student accuracy → stop.
- B. Generate teacher targets without gradients → compute student predictions → calculate hard and soft losses → combine losses → backpropagate → update the student.
- C. Calculate the validation accuracy → update both models → discard the training batch.
- D. Calculate the hard loss → skip the soft loss → update the teacher.

Correct answer: B

Explanation:

A standard training batch proceeds as follows:

1. Put the student in training mode and the teacher in evaluation mode.
2. Obtain teacher logits without recording teacher gradients.
3. Obtain student logits with gradients enabled.
4. Calculate the hard loss using the ground-truth labels.
5. Calculate the temperature-scaled distillation loss.
6. Combine the two losses.
7. Clear old student gradients.
8. Backpropagate the combined loss.
9. Update the student parameters.

For example:

python
```
optimizer.zero_grad()

with torch.no_grad():
    teacher_pred = teacher(inputs)

student_pred = student(inputs)

hard_loss = criterion(student_pred, labels)
soft_loss = distillation_output_loss(
    student_pred, teacher_pred, temperature
)

loss = (1 - alpha) * hard_loss + alpha * soft_loss

loss.backward()
optimizer.step()
```

This example assumes `distillation_output_loss` implements the teacher-target KL divergence with the $T^2$ factor.

## Part F — Experimental Design and Performance Evaluation

### Question 39. What is the objective of Task 3?

The notebook asks you to train a distilled student and compare its validation accuracy with an independently trained student. What is the main objective?

- A. Prove that the teacher has more parameters than the student.
- B. Determine whether distillation improves the student's performance under comparable training conditions.
- C. Increase the student's training loss.
- D. Change the dataset so that the student predicts fewer classes.

Correct answer: B

Explanation:

The independent student is the baseline. The distilled student uses the same general student architecture but also receives teacher-derived training signals.

The experiment tests whether those additional signals improve generalisation.

A fair comparison should keep the relevant training conditions consistent, including learning rate, number of epochs, regularisation and data splits, except for the changes specifically required by the distillation method.

### Question 40. Why use the same student architecture?

The notebook initialises a new student using the same architecture as the independent student baseline. Why is this important?

- A. It guarantees the distilled student will achieve higher accuracy.
- B. It helps isolate the effect of knowledge distillation from the effect of changing the student architecture.
- C. It ensures that the teacher and student have identical weights.
- D. It eliminates the need for validation.

Correct answer: B

Explanation:

If the student architecture changes between the baseline and distilled experiments, performance differences may result from the architecture rather than distillation.

Keeping the architecture consistent makes the comparison more informative.

However, identical architecture does not guarantee identical initial weights, identical training trajectories or identical final accuracy. It simply helps control one important experimental variable.

### Question 41. Interpreting a distillation result

An independent student achieves 72% validation accuracy, a distilled student achieves 78%, and the teacher achieves 88%.

Which statements are correct? Select all that apply.

- A. The distilled student improves by six percentage points over the independent student.
- B. The student–teacher gap is 10 percentage points after distillation.
- C. The distilled student has closed 37.5% of the original student–teacher gap.
- D. The distilled student is guaranteed to outperform the teacher after more training.
- E. The distillation experiment shows an improvement over the baseline in this run.

Correct answers: A, B and E

Explanation:

The improvement over the baseline is

$$
78\%-72\%=6\text{ percentage points}.
$$

The remaining gap to the teacher is

$$
88\%-78\%=10\text{ percentage points}.
$$

The original gap was

$$
88\%-72\%=16\text{ percentage points}.
$$

Therefore, the fraction of the original gap closed is

$$
\frac{6}{16}\times100
=\boxed{37.5\%}.
$$

The distilled student improves on the baseline but remains below the teacher. There is no guarantee that additional training will make it outperform the teacher.

### Question 42. Calculating the percentage of the gap recovered

An independent student achieves 60% accuracy, a distilled student achieves 70%, and the teacher achieves 80%.

What percentage of the original student–teacher performance gap has been recovered?

- A. 10%
- B. 25%
- C. 50%
- D. 100%

Correct answer: C — 50%

Explanation:

The student's improvement is

$$
70\%-60\%=10\text{ percentage points}.
$$

The original performance gap is

$$
80\%-60\%=20\text{ percentage points}.
$$

The proportion recovered is

$$
\frac{10}{20}\times100
=\boxed{50\%}.
$$

The student has recovered half the original gap.

This metric helps distinguish the absolute improvement from the portion of the teacher's advantage that the student has regained.

### Question 43. Fair comparison of training hyperparameters

A researcher trains the independent student for 20 epochs with a learning rate of 0.001, but trains the distilled student for 100 epochs with a learning rate of 0.01. The distilled student performs better.

What is the main problem with this comparison?

- A. Knowledge distillation cannot use different learning rates.
- B. The performance improvement cannot be attributed solely to distillation because the training conditions also changed.
- C. Validation accuracy cannot be compared across epochs.
- D. The student and teacher must have the same learning rate.

Correct answer: B

Explanation:

The researcher changed two important training hyperparameters: the number of epochs and the learning rate.

Either change could explain some or all of the performance difference.

For a fair comparison, the experiment should keep the baseline training setup consistent and modify only the components required for distillation, unless the goal is explicitly to compare optimised configurations.

A separate hyperparameter-tuning experiment may be useful, but its results should not be confused with a controlled comparison of the effect of distillation itself.

### Question 44. Understanding validation curves

The distilled student's training loss decreases steadily, but its validation accuracy stops improving and then declines. What is the most likely interpretation?

- A. The student is necessarily learning better representations.
- B. The student may be overfitting, or the training objective may no longer align with improved validation accuracy.
- C. The teacher must have been deleted.
- D. The model has no trainable parameters.

Correct answer: B

Explanation:

A decreasing training loss does not guarantee improving validation accuracy.

The student may be fitting the training data increasingly well without improving its generalisation. Other possibilities include an unsuitable temperature, loss weighting or optimisation configuration.

Useful responses include:

- Monitoring validation accuracy throughout training.
- Selecting checkpoints based on validation performance.
- Tuning temperature and the hard/soft loss balance.
- Comparing the training and validation curves with the independent baseline.

The notebook's use of validation metrics is important because the final objective is improved generalisation, not simply a lower training loss.

### Question 45. Why plot both hard and soft losses?

The notebook suggests recording both hard loss and soft loss during training. Why is this useful?

- A. It proves that the teacher is always correct.
- B. It helps diagnose how the student's ground-truth classification objective and teacher-matching objective evolve separately.
- C. It eliminates the need to measure validation accuracy.
- D. It guarantees that the two losses will decrease at the same rate.

Correct answer: B

Explanation:

The two losses measure different aspects of student learning.

- The hard loss tracks how well the student fits the ground-truth labels.
- The soft loss tracks how well the student matches the teacher's temperature-scaled distribution.

If the hard loss decreases but the soft loss remains high, the student may be learning the labelled classification task without matching the teacher closely.

If the soft loss decreases but validation accuracy does not improve, matching the teacher may not be translating into better generalisation.

Neither loss must decrease at the same rate, and validation metrics remain essential.

### Question 46. What if the distilled student performs worse?

An independent student achieves 75% validation accuracy, but the distilled student achieves 71%. Which response is most appropriate?

- A. Conclude that knowledge distillation never works.
- B. Check the implementation, then investigate temperature, loss weighting and training stability while preserving a fair baseline comparison.
- C. Increase the teacher's learning rate so that it changes during student training.
- D. Report only the teacher's accuracy.

Correct answer: B

Explanation:

Distillation is not guaranteed to improve performance in every setup.

Potential issues include:

- An incorrect KL divergence argument order.
- Applying softmax along the wrong tensor dimension.
- Omitting the $T^2$ correction.
- Using an unsuitable temperature or loss weight.
- Incorrect teacher or student train/evaluation modes.
- Accidentally updating the teacher.
- Inconsistent training hyperparameters between the baseline and distilled runs.

The correct response is to check correctness first, then conduct controlled experiments to understand which settings help.

### Question 47. Which temperature should be selected?

A researcher tests $T=1$, $T=5$, $T=10$ and $T=20$. The results are:

| Temperature | Student validation accuracy |
| --- | --- |
| $T=1$ | 72% |
| $T=5$ | 76% |
| $T=10$ | 79% |
| $T=20$ | 75% |

Which conclusion is best supported?

- A. Temperature must always be 20.
- B. Temperature 10 gives the best validation accuracy among the tested configurations.
- C. Temperature 1 is always optimal because it is standard softmax.
- D. Increasing temperature always improves accuracy.

Correct answer: B

Explanation:

Among the four tested configurations, $T=10$ produces the highest validation accuracy at 79%.

The decline at $T=20$ shows that increasing temperature indefinitely is not necessarily beneficial.

Temperature controls the softness of the teacher's target distribution, so its effect depends on the model and data. The best value should be selected using validation performance rather than a universal rule.

### Question 48. What is the correct role of the test set?

A researcher tests 20 different temperatures and reports the temperature that achieves the highest test accuracy as the final result. What is the main issue?

- A. Temperature cannot be changed during model development.
- B. Repeatedly selecting hyperparameters using test accuracy leaks test information into model selection.
- C. Test accuracy is always lower than validation accuracy.
- D. KL divergence requires the test set to be empty.

Correct answer: B

Explanation:

The test set should provide an independent estimate of final performance.

Repeatedly selecting the best temperature based on test results turns the test set into part of the model-selection process. This can lead to an overly optimistic estimate of generalisation.

A better procedure is to select temperature and other hyperparameters using validation performance, then evaluate the chosen configuration on a reserved test set.

The notebook specifically asks for validation comparisons, so the same principle applies when interpreting its results.

## Part G — Advanced Application and Integrated Questions

### Question 49. Tensor shapes in a distillation pipeline

A teacher and student each classify a batch of 64 images into 20 classes. Which statements are correct? Select all that apply.

- A. Teacher logits have shape `(64, 20)`.
- B. Student logits have shape `(64, 20)`.
- C. The temperature-scaled teacher probabilities have shape `(64, 20)`.
- D. The batch-averaged KL loss must have shape `(64, 20)`.
- E. The final distillation loss should be a scalar.

Correct answers: A, B, C and E

Explanation:

Both models produce one logit per class for each example:

$$
\text{logits shape}=(B,C)=(64,20).
$$

Temperature-scaled softmax preserves this shape.

The KL divergence calculates a divergence for each example by summing over classes, and `reduction="batchmean"` averages over the 64 examples. Therefore, the final loss is a scalar with shape `()`.

A scalar loss is needed for the usual backward pass.

### Question 50. Deriving the standard distillation objective

Which expression correctly represents a combined hard-label and soft-label distillation objective?

- A. $\displaystyle L=L_h+T D_{\mathrm{KL}}(p_t\parallel p_s)$
- B. $\displaystyle L=(1-\alpha)L_h+\alpha T^2D_{\mathrm{KL}}(p_t^{(T)}\parallel p_s^{(T)})$
- C. $\displaystyle L=(1-\alpha)L_d-\alpha L_h$
- D. $\displaystyle L=T^2D_{\mathrm{KL}}(p_s^{(T)}\parallel p_s^{(T)})$

Correct answer: B

Explanation:

The standard weighted formulation is

$$
\boxed{
L=(1-\alpha)L_h+
\alpha T^2D_{\mathrm{KL}}
\left(p_t^{(T)}\parallel p_s^{(T)}\right)
}
$$

where:

- $L_h$ is the student cross-entropy loss against ground-truth labels.
- $p_t^{(T)}$ is the teacher's temperature-scaled distribution.
- $p_s^{(T)}$ is the student's temperature-scaled distribution.
- $T$ is the temperature.
- $\alpha$ controls the relative importance of distillation.

The weighting convention must be defined clearly: in this formulation, larger $\alpha$ gives more weight to the soft loss.

### Question 51. Diagnosing a student that copies the teacher but ignores labels

A researcher uses only the soft distillation loss. The student successfully approximates the teacher's probability distributions, but its accuracy on the ground-truth labels is poor.

Which explanation is most plausible?

- A. The teacher-matching objective does not guarantee that the student will correct the teacher's errors or optimise ground-truth classification directly.
- B. Soft labels cannot represent probability distributions.
- C. The student cannot be trained using KL divergence.
- D. A student trained with KL divergence must achieve 100% accuracy.

Correct answer: A

Explanation:

The soft loss encourages the student to reproduce the teacher's predictions. If the teacher is wrong or systematically biased on some examples, matching those outputs can reproduce those mistakes.

Including the hard loss provides a direct signal from the ground-truth labels:

$$
L=(1-\alpha)L_h+\alpha L_d.
$$

Balancing the two losses can help the student learn both the labelled task and the useful information encoded in the teacher's output distribution.

### Question 52. Why might a smaller student still perform worse than its teacher?

A student is trained successfully using a teacher's soft labels, but its final accuracy remains below the teacher's accuracy. Which explanation is most reasonable?

- A. Distillation has failed because a student must always match the teacher's accuracy.
- B. The student may have limited capacity, and the distillation objective does not guarantee that it can reproduce all of the teacher's knowledge.
- C. The teacher's outputs must be one-hot encoded.
- D. The student must use a larger temperature until it exceeds the teacher.

Correct answer: B

Explanation:

A smaller model may not have sufficient capacity to reproduce the teacher's decision boundaries or representations.

Other factors, including optimisation difficulty, temperature, the loss balance and the teacher's quality, can also affect the final result.

Knowledge distillation aims to improve the student's learning, not guarantee that it reaches or exceeds the teacher's performance.

A meaningful success criterion may be a measurable improvement over the independently trained student, even if the teacher remains more accurate.

### Question 53. Interpreting the teacher's non-maximal probabilities

A teacher predicts:

$$
p_t=[0.70,0.20,0.08,0.02].
$$

A student predicts:

$$
p_s=[0.70,0.02,0.08,0.20].
$$

Both models assign the highest probability to the first class. Why can the distillation loss still be substantial?

- A. The distributions differ significantly in their relative probabilities for the other classes.
- B. KL divergence compares only the most likely class.
- C. The distributions are identical because the highest probabilities match.
- D. KL divergence cannot compare four-class distributions.

Correct answer: A

Explanation:

The student and teacher agree on the most likely class, but they disagree on the relative likelihoods of the second and fourth classes.

The teacher assigns 20% to class 2 and 2% to class 4, whereas the student does the opposite.

KL divergence compares the full probability distributions, not just the highest-probability class. Consequently, it can penalise differences even when both models predict the same top-1 class.

This illustrates why soft-target distillation can provide learning signals that ordinary accuracy does not capture.

### Question 54. Evaluating the notebook's final experiment

A notebook experiment reports the following best validation accuracies:

| Model | Best validation accuracy |
| --- | --- |
| Independent student | 68% |
| Distilled student | 74% |
| Teacher | 82% |

Which conclusion is most appropriate?

- A. The distillation experiment has improved student accuracy by six percentage points, but the distilled student remains below the teacher.
- B. The distilled student has closed the entire student–teacher gap.
- C. The teacher's validation accuracy is irrelevant to interpreting the results.
- D. The result proves that all students should be trained using the same temperature.

Correct answer: A

Explanation:

The improvement over the baseline is

$$
74\%-68\%=\boxed{6\text{ percentage points}}.
$$

The remaining gap to the teacher is

$$
82\%-74\%=8\text{ percentage points}.
$$

The original gap was

$$
82\%-68\%=14\text{ percentage points}.
$$

Thus, the fraction of the original gap recovered is

$$
\frac{6}{14}\times100\approx42.9\%.
$$

The result supports the conclusion that distillation helped in this experiment, but it does not prove that the same temperature or loss weighting will work best for every model or dataset.

### Question 55. What should be included in the final assignment report?

Which report best satisfies the notebook's assessment requirements?

- A. Only the final training loss, without a comparison to the baseline.
- B. A plot comparing validation performance against the independent student and teacher, plus a brief discussion of implementation issues and insights.
- C. Only the teacher's accuracy and architecture.
- D. A screenshot of the code without any explanation of the results.

Correct answer: B

Explanation:

The notebook explicitly requests a plot or two comparing the distilled student's final validation accuracy with the independent student baseline and the teacher.

A good report should include:

1. The independent student's best validation accuracy.
2. The distilled student's best validation accuracy.
3. The teacher's best validation accuracy.
4. A plot showing their validation performance or training curves.
5. The absolute improvement over the baseline and, if useful, the fraction of the original student–teacher gap recovered.
6. A few sentences describing implementation problems, how they were solved and what was learned about distillation.

The discussion should be based on actual experimental results. Do not claim that distillation improved accuracy unless your results demonstrate that improvement.

## Final revision — Key concepts to know by heart

| Concept | Essential fact |
| --- | --- |
| Knowledge distillation | Transfers information from a teacher to a student |
| Teacher | Provides target predictions, usually kept fixed |
| Student | Learns from ground-truth labels and teacher outputs |
| Hard labels | Ground-truth class targets |
| Soft labels | Probability distributions over classes |
| Dark knowledge | Additional class relationships encoded in teacher outputs |
| Softmax | Converts logits into a probability distribution |
| Temperature scaling | Applies softmax to logits divided by $T$ |
| High temperature | Generally produces softer distributions |
| Distillation loss | KL divergence between teacher and student distributions |
| KL direction | Teacher target distribution compared against student distribution |
| Temperature correction | Multiply the soft KL loss by $T^2$ |
| Hard loss | Cross-entropy between student logits and true labels |
| Combined objective | $(1-\alpha)L_h+\alpha L_d$ |
| PyTorch KL input | Student log-probabilities first; teacher probabilities second |
| `batchmean` | Sums over classes and averages over the batch |
| Loss shape | Scalar, `torch.Size([])` |
| Teacher mode | Usually `teacher.eval()` |
| Student mode | `student.train()` during training |
| Teacher gradients | Normally disabled |
| Optimiser | Updates student parameters only |
| Fair comparison | Keep baseline training conditions comparable |
| Temperature selection | Tune using validation performance |
| Main success criterion | Improve student performance over independent training |
| Final assessment | Compare validation results and discuss implementation insights |

## Suggested study method

1. Answer all 55 questions without looking at the explanations.
2. Revisit incorrect answers, particularly the KL divergence direction, the $T^2$ correction, PyTorch tensor dimensions and loss weighting.
3. Practise writing both `temperature_softmax` and `distillation_output_loss` from memory.
4. Explain the training loop aloud, including why the teacher is frozen and why the student receives both hard and soft losses.
5. Practise calculating validation improvements and the percentage of the student–teacher gap recovered.

The most important distinction for the exam is that temperature scaling creates softer targets, KL divergence measures how well the student matches those targets, and the combined training objective balances teacher guidance with ground-truth supervision.
