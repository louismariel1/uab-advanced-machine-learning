# Advanced Machine Learning — Practice Exam: Knowledge Distillation

This mock exam follows the structure of your Advanced Machine Learning past paper: conceptual multiple-choice questions, short written explanations, algorithm interpretation, and theoretical reasoning. It focuses on the knowledge distillation chapter you provided, rather than requiring you to write code.

The questions are designed to test whether you understand what each component does, why it is needed, and what happens when its assumptions or parameters change.

## Exam conditions

Total marks

# 35 points

Suggested time

# 60 minutes

Format

Multiple choice + written

Materials

One double-sided cheat sheet

The 60-minute duration is scaled down from the original 90-minute, 50-point exam. No code writing is required.

Instructions

- For multiple-choice questions, select all correct answers. The number of correct answers is not disclosed.
- For written questions, explain the underlying reasoning, not just the definition.
- Show calculations when requested.
- The answer key is intentionally omitted so you can attempt the exam under realistic conditions.

## Section A — Multiple choice (10 points)

### Question 1 — Purpose of knowledge distillation (2 points)

Which statements about the three models in the practical are correct? Select all that apply.

- A. The independent student provides a baseline for measuring the effect of distillation.
- B. The teacher must have the same architecture as the student.
- C. The distilled student learns from both ground-truth labels and teacher predictions.
- D. The distilled student is guaranteed to outperform the teacher.
- E. The teacher's weights should remain fixed during student training.

### Question 2 — Soft labels and temperature (2 points)

Which statements about temperature-scaled softmax are correct? Select all that apply.

- A. Increasing the temperature generally produces a softer probability distribution.
- B. Setting the temperature to 1 recovers ordinary softmax.
- C. Increasing the temperature changes the predicted class in every example.
- D. Temperature scaling allows the student to receive information about relationships between classes.
- E. The probabilities cease to sum to 1 when the temperature is greater than 1.

### Question 3 — Distillation loss (2 points)

Which statements about the KL-divergence loss used in the practical are correct? Select all that apply.

- A. The student's logits are converted to log probabilities before being passed as the first argument to `F.kl_div()`.
- B. The teacher's predictions are used as the target probability distribution.
- C. The $T^2$ factor compensates for temperature-related changes in gradient scale.
- D. KL divergence is guaranteed to be strictly positive for every pair of distributions.
- E. With `reduction='batchmean'`, the loss is divided by the batch size rather than directly by the number of classes.

### Question 4 — Training and evaluation (2 points)

Which implementation choices are appropriate when training the distilled student? Select all that apply.

- A. Use `torch.no_grad()` when computing teacher predictions.
- B. Include the teacher's parameters in the student's optimizer.
- C. Use cross-entropy against the ground-truth labels.
- D. Set the teacher to evaluation mode when generating its predictions.
- E. Update the student using only the distillation loss, regardless of the value of $\alpha$.

### Question 5 — Interpreting experimental results (2 points)

The independent student achieves 65% validation accuracy, the teacher achieves 80%, and the distilled student achieves 72%.

Which statements are correct? Select all that apply.

- A. Distillation improved accuracy by 7 percentage points over the independent student.
- B. The distilled student closed approximately 47% of the original student–teacher accuracy gap.
- C. The distilled student achieved 72% of the teacher's accuracy.
- D. The distilled student recovered some, but not all, of the original accuracy gap.
- E. These results prove that distillation always improves small neural networks.

## Section B — Short written answers (12 points)

### Question 6 — Teacher and student architectures (3 points)

The practical uses a small ResNet as the student and a larger ResNet as the teacher. Both output predictions for the same 10 CIFAR-10 classes.

(a) Explain why the teacher is expected to achieve higher accuracy than the student. (1 point)

(b) Explain why the teacher and student can have different numbers of layers and different numbers of parameters while still using knowledge distillation. (1 point)

(c) Why is it important to train and evaluate the independent student before training the distilled student? (1 point)

### Question 7 — Temperature and probability distributions (4 points)

A teacher produces the following logits for three classes:

$$
z=[4,\ 2,\ 0]
$$

Consider applying temperature-scaled softmax with $T=1$ and then with $T=4$.

(a) Without calculating exact probabilities, explain which distribution will be more concentrated on the highest-scoring class. (1 point)

(b) Explain why a higher temperature can provide useful information about relationships between classes. (1 point)

(c) What happens to the temperature-scaled distribution as $T$ becomes very large? (1 point)

(d) Explain why temperature scaling the teacher's predictions does not mean the student is forced to predict the same class probabilities as the teacher at $T=1$. (1 point)

### Question 8 — Combining the two losses (5 points)

The distilled student is trained using

$$
L=\alpha L_h+(1-\alpha)L_d
$$

where $L_h$ is cross-entropy and $L_d$ is the temperature-scaled distillation loss.

(a) Explain what each loss teaches the student. (2 points)

(b) Describe the effect of setting $\alpha=1$. (1 point)

(c) Describe the effect of setting $\alpha=0$. (1 point)

(d) Explain why $\alpha=0.5$ does not necessarily mean that the two losses contribute equally to the actual parameter updates. (1 point)

## Section C — Algorithm interpretation and application (8 points)

### Question 9 — Identify the algorithm (4 points)

Consider the following training procedure:

1. Load a pretrained teacher and a newly initialised student.
2. For each training batch, compute the teacher's logits without tracking gradients.
3. Compute the student's logits for the same images.
4. Calculate cross-entropy between the student's logits and the true labels.
5. Calculate a temperature-scaled KL-divergence loss between the student and teacher predictions.
6. Combine the losses and update the student using backpropagation.
7. Evaluate the student on the validation set after each epoch.

(a) Name the learning technique being implemented. (1 point)

(b) Explain why the teacher predictions are calculated without tracking gradients. (1 point)

(c) What would happen if the teacher were accidentally included in the optimizer and its parameters were allowed to update? Explain why this would be undesirable for the intended procedure. (1 point)

(d) Suppose the teacher achieves poor accuracy and produces unreliable predictions. Explain how this could affect the distilled student. (1 point)

### Question 10 — Diagnose a failed experiment (4 points)

A student obtains the following results:

| Model | Best validation accuracy |
| --- | --- |
| Independent student | 68% |
| Teacher | 84% |
| Distilled student | 64% |

The implementation uses $T=20$ and $\alpha=0.5$.

(a) Calculate the change in validation accuracy between the independent and distilled students, in percentage points. (1 point)

(b) Give two plausible reasons why distillation might have reduced performance in this experiment. (2 points)

(c) Suggest one controlled experiment to investigate whether the loss weighting or temperature is responsible. State what you would change and what you would measure. (1 point)

## Section D — Theoretical reasoning (5 points)

### Question 11 — Why can a student learn more than the hard labels provide? (5 points)

Consider an image whose true class is horse. The teacher predicts the following probabilities:

| Class | Teacher probability |
| --- | --- |
| Horse | 0.75 |
| Deer | 0.10 |
| Dog | 0.07 |
| Cat | 0.04 |
| Truck | 0.01 |
| Other classes | 0.03 |

(a) Explain the difference between a hard label and a soft label in this example. (1 point)

(b) Explain why the teacher's prediction provides information that is missing from the one-hot ground-truth label. (2 points)

(c) A student is trained using both cross-entropy and distillation loss. Explain how these two objectives complement one another. (1 point)

(d) The distilled student has the same architecture as the teacher and is trained on the same dataset, yet it achieves better validation accuracy. Explain why this is theoretically possible, without claiming that it is guaranteed. (1 point)

## Before you finish: exam checklist

Use this checklist to assess whether you have covered the chapter's core concepts.

### Revision progress

0 of 9

[ ]I can explain the purpose of the independent student, teacher, and distilled student.

[ ]I can explain how temperature changes a softmax distribution.

[ ]I understand the direction and inputs of the KL-divergence loss used in this practical.

[ ]I can explain the T² factor and batchmean reduction.

[ ]I can interpret the loss-weight parameter α at its extreme values.

[ ]I understand why the teacher remains fixed during distillation.

[ ]I can diagnose why distillation might hurt performance.

[ ]I can explain how soft labels encode relationships between classes.

[ ]I can calculate percentage-point improvement and the fraction of the accuracy gap closed.

## How to get feedback

Attempt the exam without looking at the chapter, and send me your answers in a format such as:

`Q1: A, C, E`

`Q2: A, B, D`

For written questions, include your reasoning in a few sentences. I'll grade your responses out of 35 points, explain each mistake, provide model answers, and give feedback in the style of the original course examiner.
