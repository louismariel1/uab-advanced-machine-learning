# Practical 7 — Continual Learning

## Complete MCQ Collection: Parts 1–9

We will follow your batch-mode workflow exactly.

- Generate MCQ 1 for Part 1.
- Continue automatically with MCQ 2 for Part 2.
- Repeat until MCQ 9 for Part 9.
- Include four answer options, the correct answer and a detailed explanation for every question.
- No confirmation requests between parts.
- No interactive practice exams at this stage.

The questions will focus on conceptual understanding, mathematical reasoning, algorithmic principles and practical PyTorch implementation.

# MCQ 1 — Part 1: Understanding Continual Learning

Topic: Fundamental concepts, catastrophic forgetting, sequential learning and the stability-plasticity dilemma.

### Question 1. What is the primary objective of Continual Learning?

A. Train a neural network on all available tasks simultaneously, ensuring that all training examples are processed in one batch.

B. Enable a neural network to learn new tasks sequentially while retaining knowledge acquired from previously learned tasks.

C. Continuously increase the size of the neural network whenever a new task is introduced.

D. Train a separate neural network from scratch for every new task and discard all previous models.

Correct answer: BExplanation:

Continual Learning aims to develop models that can learn from a sequence of tasks without forgetting previously acquired knowledge.

Unlike conventional supervised learning, where training typically assumes a fixed dataset and task, continual learning considers a changing stream of information.

For example:

1. Task 1: Learn to classify cats and dogs.
2. Task 2: Learn to classify cars and trucks.
3. Task 3: Learn to classify birds and airplanes.

The challenge is to learn Task 3 without losing the knowledge acquired during Tasks 1 and 2.

Important concept: Catastrophic Forgetting

Catastrophic forgetting occurs when learning a new task significantly reduces performance on previously learned tasks.

Important concept: Stability-Plasticity Dilemma

- Stability: Preserve previously acquired knowledge.
- Plasticity: Remain capable of learning new information.

An effective continual learning algorithm must balance both.

Exam takeaway: Continual Learning is not simply learning new tasks. It is learning new tasks while preserving knowledge from previous tasks.

# MCQ 2 — Part 2: Learning Scenarios and Experimental Setup

Topic: Task-IL, Class-IL, Domain-IL, task sequences and CIFAR datasets.

### Question 2. What is the fundamental difference between Task-Incremental Learning (Task-IL) and Class-Incremental Learning (Class-IL)?

A. Task-IL requires the model to learn without labels, whereas Class-IL uses supervised learning.

B. Task-IL assumes that the task identity is available during inference, whereas Class-IL requires the model to distinguish between all previously encountered classes without knowing the task identity.

C. Task-IL uses a single classification head, whereas Class-IL always requires a separate neural network for each task.

D. Task-IL is exclusively designed for CIFAR-10, whereas Class-IL is exclusively designed for CIFAR-100.

Correct answer: BExplanation:

The distinction concerns what information is available when the model makes predictions.

Task-Incremental Learning (Task-IL)

The model knows which task an input belongs to during inference.

For example:

- Task 1 contains cats and dogs.
- Task 2 contains cars and trucks.

When an image is presented, the model knows which task-specific classification problem it must solve.

Class-Incremental Learning (Class-IL)

The model does not know the task identity during inference.

It must distinguish between all classes learned so far.

For example, after learning:

- Task 1: Cats and dogs.
- Task 2: Cars and trucks.

The model must classify an image among all four categories without being told whether it comes from Task 1 or Task 2.

Domain-Incremental Learning (Domain-IL)

The classes remain the same, but the input distribution changes.

For example, a model might classify the same object categories under different image styles, lighting conditions or environments.

CIFAR-10 and CIFAR-100 examples:

- CIFAR-10 contains 10 classes, making it convenient for a five-task experiment with two classes per task.
- CIFAR-100 contains 100 classes, allowing more extensive task sequences.

Exam takeaway: Task-IL provides task identity at inference; Class-IL does not. Class-IL is generally a more challenging evaluation setting.

# MCQ 3 — Part 3: Baseline Sequential Training

Topic: CNN architecture, sequential supervised training and catastrophic forgetting.

### Question 3. A CNN is trained sequentially on five tasks. After completing Task 1, the model achieves 90% accuracy on Task 1. After training Tasks 2–5, its accuracy on Task 1 drops to 25%.

What does this result primarily demonstrate?

A. Overfitting on Task 1.

B. Catastrophic forgetting caused by parameter updates during subsequent task training.

C. Successful transfer learning because the model has learned five tasks.

D. A reduction in the number of classes in the training dataset.

Correct answer: BExplanation:

This is a typical example of catastrophic forgetting.

During sequential training, the model updates its parameters to minimise the loss on the current task.

For example, the training procedure might look like this:

1. Train on Task 1: Cats and dogs.
2. Train on Task 2: Cars and trucks.
3. Train on Task 3: Birds and airplanes.
4. Train on Task 4: Ships and horses.
5. Train on Task 5: Frogs and deer.

The optimisation process for a new task can modify parameters that were important for previous tasks.

Consequently, the model may become increasingly specialised in the most recently learned task.

The important distinction is that good performance on the current task does not guarantee good performance on previous tasks.

Typical PyTorch implementation:

python

```python
for task in tasks:
    for images, labels in task:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
```

This is ordinary sequential training. It does not explicitly protect previously acquired knowledge.

Exam takeaway: Sequential training without a forgetting-prevention mechanism provides the baseline against which continual learning algorithms such as EWC and Experience Replay can be compared.

# MCQ 4 — Part 4: Evaluation Metrics

Topic: Accuracy matrix, Average Accuracy, Forgetting, Backward Transfer and Forward Transfer.

### Question 4. A continual learning experiment involves three sequential tasks. After training all three tasks, the model obtains the following accuracies:

| Task | Final accuracy |
| --- | --- |
| Task 1 | 60% |
| Task 2 | 70% |
| Task 3 | 90% |

What is the final Average Accuracy across the three tasks?

A. 60%

B. 70%

C. 73.33%

D. 90%

Correct answer: CExplanation:

Average Accuracy measures the model's overall performance across all tasks learned so far.

It is calculated as:

$$
A_{\text{avg}}=
\frac{1}{T}\sum_{i=1}^{T}a_i
$$

where:

- $T$ is the number of tasks.
- $a_i$ is the accuracy on task $i$ after training the complete sequence.

Substituting the values:

$$
A_{\text{avg}}=
\frac{60+70+90}{3}
$$

$$
\boxed{A_{\text{avg}}=73.33\%}
$$

Other important evaluation metrics:

- Forgetting: Measures how much performance on previous tasks deteriorates after learning subsequent tasks.
- Backward Transfer (BWT): Measures how learning subsequent tasks affects performance on previous tasks.
- Forward Transfer (FWT): Measures whether knowledge from previous tasks helps learning new tasks.

Exam takeaway: Final Average Accuracy measures overall retained performance, whereas Forgetting and BWT provide information about how learning new tasks affects previously learned tasks.

# MCQ 5 — Part 5: Regularisation-Based Methods — EWC

Topic: Elastic Weight Consolidation, Fisher Information Matrix and parameter importance.

### Question 5. What is the main purpose of Elastic Weight Consolidation (EWC) in Continual Learning?

A. Store all previous training images and replay them during every new task.

B. Prevent important parameters for previous tasks from changing significantly when learning new tasks.

C. Replace the CNN backbone with a nearest-neighbour classifier.

D. Increase the learning rate whenever the model encounters a new task.

Correct answer: BExplanation:

Elastic Weight Consolidation is a regularisation-based continual learning method.

Its purpose is to protect parameters that are important for previously learned tasks.

After learning Task 1, EWC estimates the importance of model parameters using the Fisher Information Matrix.

When learning Task 2, it penalises changes to parameters that were particularly important for Task 1.

The EWC objective is:

$$
\boxed{
L(\theta)=L_{\text{new}}(\theta)
+
\frac{\lambda}{2}
\sum_i F_i(\theta_i-\theta_i^*)^2
}
$$

Where:

- $L_{\text{new}}$: Loss on the new task.
- $\theta_i$: Current parameter.
- $\theta_i^*$: Parameter value after learning the previous task.
- $F_i$: Estimated importance of parameter $i$.
- $\lambda$: Regularisation strength.

Understanding the Fisher Information:

- Large $F_i$: The parameter is important for the previous task, so changing it is penalised strongly.
- Small $F_i$: The parameter is less important, so it can change more freely.

Example:

Suppose two parameters have different Fisher importance:

| Parameter | Fisher importance |
| --- | --- |
| Weight 1 | 100 |
| Weight 2 | 0.1 |

EWC strongly discourages changing Weight 1, while allowing Weight 2 to adapt more easily.

Typical PyTorch implementation:

python

```python
ewc_loss = 0

for name, param in model.named_parameters():
    ewc_loss += (
        fisher[name]
        * (param - old_params[name]) ** 2
    ).sum()

loss = task_loss + (lambda_ewc / 2) * ewc_loss
```

Exam takeaway: EWC protects important parameters through a Fisher-weighted quadratic penalty. It does not require storing all previous training images.

# MCQ 6 — Part 6: Replay-Based Methods

Topic: Experience Replay, memory buffers and rehearsal.

### Question 6. How does Experience Replay reduce catastrophic forgetting?

A. It freezes all neural network parameters after learning the first task.

B. It stores selected examples from previous tasks and combines them with current-task examples during training.

C. It deletes the previous task's classifier before training the next task.

D. It calculates the Fisher Information Matrix and penalises changes to important weights.

Correct answer: BExplanation:

Experience Replay is a replay-based continual learning method.

The main idea is to retain a limited collection of examples from previous tasks in a memory buffer.

When the model learns a new task, it trains using both:

- New examples from the current task.
- Stored examples from previous tasks.

For example:

- Task 1: Store selected cat and dog images.
- Task 2: Train on cars and trucks while replaying stored cat and dog images.
- Task 3: Train on birds and airplanes while replaying examples from earlier tasks.

This helps the model maintain performance on previously learned categories.

Typical PyTorch training concept:

python

```python
for images, labels in current_task_loader:

    replay_images, replay_labels = memory.sample()

    combined_images = torch.cat(
        [images, replay_images]
    )

    combined_labels = torch.cat(
        [labels, replay_labels]
    )

    optimizer.zero_grad()

    outputs = model(combined_images)
    loss = criterion(outputs, combined_labels)

    loss.backward()
    optimizer.step()
```

Important considerations:

- Buffer capacity: How many examples can be stored?
- Sampling strategy: Which examples should be retained?
- Memory management: How should the buffer be updated?
- Replay ratio: How many old examples should be combined with new examples?

Exam takeaway: Experience Replay reduces forgetting by revisiting representative examples from previous tasks while learning new tasks.

# MCQ 7 — Part 7: Knowledge Distillation and Learning Without Forgetting

Topic: Teacher-student learning, soft targets, distillation loss and LwF.

### Question 7. What is the central idea behind Learning Without Forgetting (LwF)?

A. Train a completely new model for every task and discard the previous model.

B. Preserve knowledge from previous tasks by encouraging the updated model to maintain predictions similar to those of the previous model.

C. Store every previous training image in a replay buffer.

D. Prevent all parameter updates when a new task is introduced.

Correct answer: BExplanation:

Learning Without Forgetting uses knowledge distillation to preserve previously acquired knowledge.

It maintains an older model as a teacher and uses its predictions to guide the updated student model.

The general process is:

1. Train the original model on Task 1.
2. Keep a copy of the trained model as the teacher.
3. Train the student model on Task 2.
4. Encourage the student to maintain the teacher's predictions on previously learned classes.
5. Combine the new-task learning objective with the distillation objective.

A simplified objective is:

$$
L_{\text{total}}=
L_{\text{new}}+
\lambda L_{\text{distill}}
$$

The distillation loss can use temperature-scaled probability distributions:

$$
p_i^{(T)}
=
\frac{\exp(z_i/T)}
{\sum_j\exp(z_j/T)}
$$

A common distillation loss is:

$$
L_{\text{distill}}
=
T^2
D_{\mathrm{KL}}
\left(
p_{\text{teacher}}^{(T)}
\parallel
p_{\text{student}}^{(T)}
\right)
$$

Where:

- $T$: Temperature parameter.
- $D_{\mathrm{KL}}$: Kullback-Leibler divergence.
- Teacher: Previous model.
- Student: Model being updated.

The temperature softens the probability distributions, allowing the student to learn from the teacher's relative preferences among classes.

Important distinction:

- EWC protects important parameters.
- Experience Replay revisits stored previous examples.
- LwF preserves previous model outputs through distillation.

Exam takeaway: LwF attempts to preserve previous knowledge without requiring a replay buffer of old training images.

# MCQ 8 — Part 8: Experimental Comparison

Topic: Hyperparameters, performance comparison, stability-plasticity and experimental interpretation.

### Question 8. In a continual learning experiment, three methods obtain the following results after learning five sequential tasks:

| Method | Average Accuracy | Average Forgetting |
| --- | --- | --- |
| Sequential baseline | 55% | 35% |
| EWC | 68% | 18% |
| Experience Replay | 75% | 10% |

Which conclusion is most appropriate?

A. The sequential baseline is the best method because it has the simplest implementation.

B. EWC and Experience Replay improve performance compared with the baseline, and Experience Replay performs best according to both reported metrics.

C. EWC is necessarily better because it does not require storing previous images.

D. Experience Replay is worse because its Average Accuracy is higher.

Correct answer: BExplanation:

The results demonstrate how different continual learning methods can improve knowledge retention.

Let's examine the two metrics.

Average Accuracy

Higher is better.

- Baseline: 55%.
- EWC: 68%.
- Experience Replay: 75%.

Experience Replay achieves the highest overall accuracy.

Average Forgetting

Lower is better.

- Baseline: 35%.
- EWC: 18%.
- Experience Replay: 10%.

Experience Replay also achieves the lowest forgetting.

Therefore, under this particular experimental configuration, Experience Replay performs best according to both metrics.

However, this does not establish that Experience Replay will always outperform EWC.

Other considerations include:

- Memory requirements.
- Computational cost.
- Buffer selection strategy.
- Dataset complexity.
- Number of sequential tasks.
- Hyperparameter selection.

The stability-plasticity dilemma:

A model must preserve previously acquired knowledge while remaining adaptable to new tasks.

A method that prevents too many parameter changes might preserve old knowledge but struggle to learn new tasks. Conversely, a highly adaptable model might learn new tasks quickly but forget previous ones.

Exam takeaway: Experimental conclusions must consider multiple metrics and computational constraints, rather than relying on Average Accuracy alone.

# MCQ 9 — Part 9: Advanced Understanding and Exam Preparation

Topic: Algorithm comparison, mathematical reasoning, implementation and conceptual understanding.

### Question 9. Consider the following three continual learning strategies:

1. Method A penalises changes to parameters according to their estimated importance for previous tasks.
2. Method B stores representative examples from previous tasks and reuses them during training.
3. Method C uses predictions from an older model to guide the updated model.

Which combination correctly identifies these methods?

A. A = Experience Replay; B = EWC; C = LwF.

B. A = EWC; B = Experience Replay; C = LwF.

C. A = LwF; B = Experience Replay; C = EWC.

D. A = EWC; B = LwF; C = Experience Replay.

Correct answer: BExplanation:

This question tests whether you understand the fundamental differences between the three principal methods.

Method A — EWC

Regularisation-basedEWC estimates parameter importance using Fisher Information.

Its objective is to discourage changes to parameters that are important for previous tasks.

$$
L=L_{\text{new}}+L_{\text{EWC}}
$$

Method B — Experience Replay

Replay-basedExperience Replay maintains a memory buffer containing examples from previous tasks.

The model revisits these examples when learning new tasks.

$$
L=L_{\text{current}}+L_{\text{replay}}
$$

This is a simplified representation of the combined training objective.

Method C — Learning Without Forgetting

Distillation-basedLwF uses an older model's predictions to guide the updated model.

$$
L=L_{\text{new}}+\lambda L_{\text{distill}}
$$

This helps preserve the previous model's behaviour.

### Final algorithm comparison

| Characteristic | EWC | Experience Replay | LwF |
| --- | --- | --- | --- |
| Main strategy | Parameter regularisation | Memory replay | Knowledge distillation |
| Stores old images | Not required | Yes | Not required |
| Uses Fisher Information | Yes | No | No |
| Uses a memory buffer | No | Yes | No |
| Uses a teacher model | No | No | Yes |
| Main objective | Protect important weights | Revisit previous examples | Preserve previous predictions |

Exam takeaway: You should be able to identify each continual learning method from its underlying mechanism, mathematical objective and implementation strategy.

# Final overview — MCQ 1–9

All 9 MCQs generated| MCQ | Part | Main subject | Correct answer |
| --- | --- | --- | --- |
| 1 | Part 1 | Understanding Continual Learning | B |
| 2 | Part 2 | Learning Scenarios and Experimental Setup | B |
| 3 | Part 3 | Baseline Sequential Training | B |
| 4 | Part 4 | Evaluation Metrics | C |
| 5 | Part 5 | Regularisation-Based Methods — EWC | B |
| 6 | Part 6 | Replay-Based Methods | B |
| 7 | Part 7 | Knowledge Distillation and LwF | B |
| 8 | Part 8 | Experimental Comparison | B |
| 9 | Part 9 | Advanced Understanding and Exam Preparation | B |

## What we have accomplished

We now have nine question-focused MCQs covering the complete proposed Practical 7 Continual Learning syllabus.

Each MCQ includes:

- A four-option examination question.
- The correct answer.
- A detailed conceptual explanation.
- Relevant mathematical formulations or PyTorch examples where appropriate.
- An examination takeaway.

Next phase: We can use these nine MCQs as the source material for nine separate interactive practice exams, following the same workflow we established for Practical 6.

For now, the MCQ generation phase is complete.
