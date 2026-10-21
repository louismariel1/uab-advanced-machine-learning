# Interactive Online Exam Sheet - Interactive Exam Practice Form.

Below is a regenerated, clean version of the Advanced Machine Learning — Previous Exam (35 points), reconstructed from your LaTeX document. I have preserved the original questions, point values, multiple-choice options, equations, and question order, while improving the formatting so you can use it directly for exam preparation.

The original document contains the previous-exam questions on pages 7–12. The two introductory pages, exam-format instructions, and cheat-sheet information are not included in the regenerated exam.

# Advanced Machine Learning

Previous Exam — Autumn 2026 course material · Total: 35 points · Reconstructed practice paper

Exam instructions

- Answer all questions.
- For multiple-choice questions, select all applicable answers.
- Explain your reasoning where requested.
- No code writing is required; pseudocode or natural-language algorithm descriptions are acceptable.
- The original questions below total 35 points.

## Question 1 — Deployment Performance (2 points)

Which of the following situations would cause a model to perform worse in deployment than on its validation split? Mark all that apply.

A. The model has insufficient capacity to learn the patterns in the data.

B. There is insufficient labelled data to attain good training accuracy.

C. The model overfits on the training split instead of learning generic features.

D. The data distribution seen in deployment is different from that used in training.

## Question 2 — Fine-Tuning (2 points)

Which of the following are true when fine-tuning a pre-trained model from source task \\(A\\) to target task \\(B\\)? Mark all that apply.

A. Fine-tuning is more effective if A and B have similar domains.

B. Early layers need more fine-tuning than later layers because they process raw input.

C. The model architecture should remain the same for pre-training and for fine-tuning.

D. The learning rate should remain the same for pre-training and for fine-tuning.

## Question 3 — Knowledge Distillation (2 points)

Which of the following are true properties of knowledge distillation by teacher–student training? Mark all that apply.

A. The student learns both from ground-truth labels and the teacher's supervision.

B. Temperature scaling is applied to the logits of the teacher, but not the student.

C. The student should have the same number of layers as the teacher.

D. The student model can inherit biases and adversarial vulnerabilities from the teacher.

E. Distillation is more effective when the teacher has high confidence on all predictions.

## Question 4 — Different Output Dimensions in Knowledge Distillation (3 points)

Can knowledge distillation be used in a setting where the teacher has a different number of output neurons from the student?

Explain your answer.

Answer space: explain whether it is possible and under what conditions.

## Question 5 — Contrastive Learning (3 points)

You aim to pre-train a vision backbone model using unsupervised data and contrastive learning. The training scheme is as follows:

1. At batch time, sample two different images, $a$ and $b$, from the training set.
2. Apply random augmentations to $a$, producing $a^*$. Apply other random augmentations to $b$, producing $b^*$.
3. Pass all four images, $[a, a^*, b, b^*]$, through the model and retrieve their embeddings in the final layer's feature space.
4. Calculate the loss using distances in feature space between these embeddings.

Fill the matrix with HIGH or LOW, according to whether the model should be optimised to have high or low distances between the corresponding feature embeddings.

Rows and columns represent the respective input images.

|     | a | a\* | b | b\* |
| --- | - | --- | - | --- |
| a   | — | —   | — | —   |
| a\* | — | —   | — | —   |
| b   | — | —   | — | —   |
| b\* | — | —   | — | —   |

## Question 6 — Reptile and MAML (7 points)

Reptile is a meta-learning algorithm for few-shot learning. Its update rule can be written as:

$$
\theta_M \leftarrow \theta_M + \beta(\theta_j - \theta_M)
$$

**(a) (2 points)** Explain what the terms $\theta_M$ and $\theta_j$ represent in this notation.

**(b) (2 points)** Describe the effect of setting the $\beta$ parameter to $1.0$.

**(c) (3 points)** Model-Agnostic Meta-Learning (MAML) is a conceptually similar algorithm to Reptile that has the same overall objective. Explain the main difference between the two algorithms.

## Question 7 — Ensemble Learning (7 points)

You initialise five neural networks with the same architecture and train them all on the same task using the same data. All five have similar training and test accuracy of around 80%.

However, you discover that you can improve performance at inference time by averaging the predictions of the five networks on the same input, achieving a test accuracy of 85%.

(a) (4 points) Give a theoretical explanation for this phenomenon, and explain why averaging the predictions results in better performance than simply using the single best network.

(b) (3 points) Would this approach work for other machine learning models, such as linear regressions or decision trees? Explain your answer.

## Question 8 — Self-Paced Curriculum Learning (4 points)

In self-paced curriculum learning, the model being trained provides a difficulty score for each example in a training batch, and these scores are used to selectively update its gradients.

Describe a basic implementation of this algorithm, in pseudocode or in English, including:

- The scoring function used to measure example difficulty.
- How examples are selected based on their scores.
- How the selected examples are used to update the model.

## Question 9 — Self-Distillation (5 points)

The phenomenon of self-distillation is a special case of teacher–student knowledge distillation, where the teacher and student models have the same architecture and are trained on the same data.

Explain how it is possible for the student to exceed the performance of the teacher in this setting.

## Exam completion

Progress

## 0/12

This counts completed response fields, not exam marks. There are 9 numbered questions and 35 points in total.

Copy exam outline

Suggested next step: Try the paper under a 60-minute practice limit, then send me your answers. I can grade them against the expected concepts, award marks question by question, and explain how to improve each answer for the actual 90-minute exam.
