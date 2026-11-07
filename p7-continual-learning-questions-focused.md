# Advanced Machine Learning — Practical 7: Continual Learning

## Questions-Focused Summary — Exam Preparation and Practical Understanding

New topic: Practical 7 Proposed structure: 9 parts

I agree with this approach. We can follow exactly the same learning and examination preparation workflow that we established for Practical 6.

The important difference is that, since the actual Practical 7 notebook is not yet available, we will construct a representative Continual Learning practical, based on a standard PyTorch notebook and commonly used datasets such as CIFAR-10 and CIFAR-100.

We will treat this as our working reference notebook, not as a claim about the actual contents of the course's future Practical 7.

The objective is to understand:

- The theoretical concepts behind continual learning.
- The practical implementation of continual learning algorithms.
- The purpose of the important Python/PyTorch code.
- The mathematical equations behind the algorithms.
- The experimental results and evaluation metrics.
- The concepts and implementation details most likely to be examined.

## 1. Our learning workflow

We will use the same two-step process as Practical 6 for every part.

# 1

Step 1 — Questions-focused MCQ

Generate multiple-choice questions with four answer options, correct answers and detailed explanations.

# 2

Step 2 — Interactive practice exam

Convert the MCQs into an interactive examination with selectable answers, scoring, feedback and explanations.

We will complete both steps for Part 1 before moving to Part 2, and so on.

## 2. What is Continual Learning?

The central question of Practical 7 is:

Can a neural network learn new tasks or categories sequentially without forgetting what it previously learned?

In ordinary supervised learning, a model is typically trained using a dataset containing all the training examples available for its task.

Continual learning is different.

A model receives data over time, usually in a sequence of tasks. It must learn each new task while preserving as much knowledge as possible from previous tasks.

For example:

Illustrative sequence of learning tasks

1. Task 1 — Learn animals
   Classification on CIFAR-10. Earlier we have discussed about Deep… | by The AI Guy | Nerd For Tech | Medium
   [medium.com](<https://medium.com/nerd-for-tech/classification-on-cifar-10-32abe456302?utm_source=chatgpt.com>)
   Learn to recognise cats, dogs, frogs and deer.
2. Task 2 — Learn vehicles
   Lesson1@CS231n:Image Classification Pipeline | eKuL
   [kuangliu.github.io](<https://kuangliu.github.io/2016/06/28/1.Image_Classification_Pipeline/?utm_source=chatgpt.com>)
   Learn automobiles, trucks, airplanes and ships, while retaining the ability to recognise animals.
3. Task 3 — Learn additional categories
   Content-based image retrieval via transfer learning - Iqra Toheed Chughtai, Asma Naseer, Maria Tamoor, Saara Asif, Mamoona Jabbar, Rabia Shahid, 2023
   [journals.sagepub.com](<https://journals.sagepub.com/doi/10.3233/JIFS-223449?utm_source=chatgpt.com>)
   Learn additional categories without substantially degrading performance on earlier tasks.

GoalPreserve previously acquired knowledge while learning new information.

The main challenge is called catastrophic forgetting.

When a neural network is trained on a new task, its parameters may change in ways that damage the knowledge acquired from previous tasks.

Continual learning investigates how to reduce this forgetting.

# Proposed Practical 7 — Questions-Focused Summary

We will organise the assumed notebook into nine parts, following the progression from fundamental concepts to implementation, experiments and exam preparation.

## Part 1 — Understanding Continual Learning

Foundations and terminologyThis part establishes the fundamental concepts before looking at the code.

### Q1. What is Continual Learning?

Answer: Continual Learning (CL) is a machine learning approach in which a model learns from a sequence of tasks or experiences over time while attempting to retain previously acquired knowledge.

Unlike conventional supervised learning, the model does not necessarily have access to all previous training data when learning a new task.

The objective is to balance:

- Learning new information.
- Retaining previously acquired knowledge.
- Adapting to changes in the data distribution.

### Q2. What is catastrophic forgetting?

Answer: Catastrophic forgetting occurs when learning a new task causes a neural network to lose performance on previously learned tasks.

For example:

- Train a model on Task 1: cats and dogs.
- Train the same model on Task 2: cars and trucks.
- The model becomes good at recognising cars and trucks but loses its ability to recognise cats and dogs.

This happens because neural network parameters are shared between tasks. Updating them for a new task can interfere with previously learned representations.

### Q3. Why is continual learning important?

Answer: Many real-world systems receive new information continuously.

Examples include:

- Autonomous vehicles learning to recognise new road situations.
- Medical systems adapting to new imaging patterns.
- Robots learning new environments.
- Recommendation systems adapting to changing user preferences.
- Image classification systems learning new categories over time.

Retraining a model from scratch whenever new information becomes available can be expensive and inefficient.

### Q4. What is the difference between traditional learning and continual learning?

| Traditional supervised learning | Continual learning |
| --- | --- |
| Usually trains on a fixed dataset. | Learns from a sequence of tasks. |
| Training data is often available together. | Data may arrive incrementally. |
| No explicit requirement to retain earlier tasks after sequential training. | Must minimise forgetting. |
| Typically uses one training phase. | Uses multiple learning phases. |

### Q5. What are tasks in continual learning?

Answer: A task represents a learning problem associated with a particular dataset, class subset or data distribution.

For example, with CIFAR-10, we could divide the ten classes into five tasks:

| Task | Classes |
| --- | --- |
| Task 1 | Airplane, automobile |
| Task 2 | Bird, cat |
| Task 3 | Deer, dog |
| Task 4 | Frog, horse |
| Task 5 | Ship, truck |

The model learns these tasks sequentially rather than receiving all classes simultaneously.

### Q6. What is a task sequence?

Answer: A task sequence is the ordered collection of learning tasks presented to the model.

For example:

$$
T_1\rightarrow T_2\rightarrow T_3\rightarrow T_4
$$

The order matters because learning a new task can affect performance on earlier tasks.

Exam tip: Continual learning is not simply learning multiple tasks. The important challenge is learning them sequentially while retaining previous knowledge.

## Part 2 — Continual Learning Scenarios and Experimental Setup

Task types and datasetsBefore implementing an algorithm, we must understand how continual learning experiments are organised.

### Q7. What are the three main continual learning scenarios?

Answer: The three commonly discussed scenarios are:

1. Task-Incremental Learning (Task-IL).
2. Class-Incremental Learning (Class-IL).
3. Domain-Incremental Learning (Domain-IL).

### Q8. What is Task-Incremental Learning?

Answer: In Task-IL, the model learns multiple tasks sequentially and is given information about which task is currently being evaluated.

Each task can have its own classification head or task-specific output configuration.

For example:

- Task 1: Recognise animals.
- Task 2: Recognise vehicles.
- Task 3: Recognise household objects.

The model knows which task it is solving.

### Q9. What is Class-Incremental Learning?

Answer: In Class-IL, new classes are introduced over time, and the model must classify examples across all previously learned classes and newly introduced classes.

For example:

- First learn cats and dogs.
- Then learn cars and trucks.
- Finally classify all four categories without being told which task an image belongs to.

This is generally more challenging than Task-IL because the model must distinguish between all available classes.

### Q10. What is Domain-Incremental Learning?

Answer: In Domain-IL, the model learns from different distributions or environments while the prediction task remains essentially the same.

For example, an image classifier may encounter:

- Normal lighting.
- Low-light images.
- Different camera conditions.
- Images with different visual styles.

The classes remain the same, but the input distribution changes.

### Q11. What is the difference between Task-IL, Class-IL and Domain-IL?

| Scenario | What changes? | Task identity available? |
| --- | --- | --- |
| Task-IL | Task-specific classes or objectives | Yes |
| Class-IL | New classes are introduced | No |
| Domain-IL | Input distribution changes | Generally no |

### Q12. Which datasets are suitable for continual learning experiments?

Answer: Several standard datasets can be used.

Image Dataset(1)

CIFAR-10

60,000 colour images, 10 classes, 32 × 32 pixels.

A common starting point for class-incremental experiments.

(PDF) Classification of Images Using CNN Model and its Variants

CIFAR-100

60,000 colour images, 100 classes, 32 × 32 pixels.

More challenging because it contains many more categories.

Introduction to Deep Learning

MNIST

70,000 grayscale images of handwritten digits.

Often used for introductory continual learning experiments.

Fashion\_MNIST\_Data\_Image Classification in TensorFlow | by sankar channa | Medium

Fashion-MNIST

70,000 grayscale clothing images across 10 classes.

Useful for studying sequential classification.

### Q13. How can CIFAR-10 be divided into sequential tasks?

Answer: One approach is to split the ten classes into five groups of two classes.

python

```python
task_classes = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9]
]
```

Each group becomes a separate learning task.

The model learns Task 1 first, followed by Task 2, and so on.

The exact class order should be recorded because different task orders can produce different forgetting results.

### Q14. What is a task dataset or task-specific dataloader?

Answer: It is a dataset or DataLoader containing the examples belonging to a particular task.

For example:

python

```python
task_dataset = Subset(
    train_dataset,
    task_indices
)

task_loader = DataLoader(
    task_dataset,
    batch_size=64,
    shuffle=True
)
```

The `Subset` selects the relevant training examples, while the `DataLoader` provides batches for model training.

Exam tip: Understand the difference between the learning scenario and the dataset partition. CIFAR-10 is the dataset; Task-IL or Class-IL describes the experimental learning setting.

## Part 3 — The Baseline: Sequential Training and Catastrophic Forgetting

First implementationThis part introduces the simplest continual learning baseline: sequentially training the same model on each task without an explicit forgetting-prevention mechanism.

### Q15. What is sequential training?

Answer: Sequential training means training a model on one task, then moving to the next task using the model parameters learned from the previous task.

The model is not reset between tasks.

The sequence is:

$$
\theta_0
\rightarrow \theta_1
\rightarrow \theta_2
\rightarrow \theta_3
$$

Each new task changes the model parameters.

### Q16. How does the baseline model work?

Answer: A standard CNN can be used as the feature extractor and classifier.

A representative PyTorch implementation is:

python

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

        self.classifier = nn.Linear(
            64 * 8 * 8,
            num_classes
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

The CNN extracts image features, and the linear layer generates classification logits.

This example assumes 32 × 32 RGB images and a ten-class output.

### Q17. How is the baseline trained on each task?

Answer: The same model is trained sequentially on each task.

python

```python
for task_id, task_loader in enumerate(
    task_loaders
):

    model.train()

    for images, labels in task_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = F.cross_entropy(
            logits, labels
        )

        loss.backward()

        optimizer.step()
```

Important operations:

- `optimizer.zero_grad()` clears previous gradients.
- `model(images)` generates predictions.
- `F.cross_entropy()` calculates classification loss.
- `loss.backward()` calculates gradients.
- `optimizer.step()` updates model parameters.

### Q18. Why does sequential training cause catastrophic forgetting?

Answer: The model updates its shared parameters to minimise the loss of the current task.

Those parameter updates may damage the representations needed for earlier tasks.

The model therefore becomes increasingly specialised in the most recently learned task.

### Q19. How can we demonstrate catastrophic forgetting experimentally?

Answer: Evaluate the model on previously learned tasks after training each new task.

For example:

| Training stage | Task 1 accuracy | Task 2 accuracy | Task 3 accuracy |
| --- | --- | --- | --- |
| After Task 1 | 90% | — | — |
| After Task 2 | 55% | 88% | — |
| After Task 3 | 35% | 60% | 87% |

Illustrative results only, not experimental measurements.

The decreasing accuracy on Task 1 demonstrates forgetting.

### Q20. Why is the baseline important?

Answer: It establishes how severe forgetting becomes when no explicit continual learning mechanism is used.

More advanced algorithms can then be compared against this baseline.

Exam tip: A model that performs well on the most recent task is not necessarily a good continual learning model. We must also evaluate its performance on earlier tasks.

## Part 4 — Measuring Continual Learning Performance

Evaluation and mathematical metricsThis is one of the most important parts because continual learning requires more than measuring final classification accuracy.

### Q21. Why is ordinary classification accuracy insufficient?

Answer: Ordinary accuracy measures how many examples are correctly classified, but it does not necessarily show how much knowledge the model has forgotten.

A continual learning experiment must measure performance across multiple tasks and learning stages.

### Q22. What is the accuracy matrix?

Answer: The accuracy matrix records performance on different tasks after each sequential training stage.

Let:

$$
R_{i,j}
$$

represent the accuracy on task $j$ after training up to task $i$.

For example:

$$
R=
\begin{bmatrix}
90 & 0 & 0\\
55 & 88 & 0\\
35 & 60 & 87
\end{bmatrix}
$$

The rows represent training stages, and the columns represent evaluation tasks.

The matrix helps us observe how learning new tasks affects previous tasks.

### Q23. What is Average Accuracy?

Answer: Average Accuracy measures the mean performance across all tasks learned so far.

After training task $t$:

$$
A_t=\frac{1}{t}\sum_{j=1}^{t}R_{t,j}
$$

It provides an overall measure of performance across the learned tasks.

### Q24. What is Forgetting?

Answer: Forgetting measures how much performance on an earlier task decreases after learning subsequent tasks.

One common definition is:

$$
F_j=
\max_{i\in\{j,\ldots,T\}}R_{i,j}
-R_{T,j}
$$

Here, $F_j$ measures the difference between the best recorded accuracy on task $j$ and its final accuracy.

Average forgetting is:

$$
F=\frac{1}{T-1}
\sum_{j=1}^{T-1}F_j
$$

A smaller value generally indicates better retention.

### Q25. What is Backward Transfer (BWT)?

Answer: Backward Transfer measures how learning subsequent tasks affects performance on previously learned tasks.

A common definition is:

$$
BWT=
\frac{1}{T-1}
\sum_{j=1}^{T-1}
(R_{T,j}-R_{j,j})
$$

- Negative BWT indicates that later learning has harmed previous-task performance.
- Positive BWT indicates that later learning has improved previous-task performance.

### Q26. What is Forward Transfer (FWT)?

Answer: Forward Transfer measures whether knowledge from earlier tasks helps the model perform on future tasks before learning them.

It is usually defined relative to an appropriate initial-model or reference baseline:

$$
FWT=
\frac{1}{T-1}
\sum_{j=2}^{T}
(R_{j-1,j}-b_j)
$$

Here, $b_j$ is the reference performance on task $j$ before learning it.

Positive forward transfer suggests that previous learning benefits future tasks.

### Q27. Which metrics are most important?

| Metric | What it measures | Desired direction |
| --- | --- | --- |
| Average Accuracy | Overall task performance | Higher |
| Forgetting | Loss of previous knowledge | Lower |
| BWT | Effect of later tasks on earlier tasks | Less negative or higher |
| FWT | Benefit of previous learning to future tasks | Higher |

Exam tip: Be able to explain why a model can have high final accuracy on the last task but poor average accuracy across all tasks.

## Part 5 — Regularisation-Based Methods: Elastic Weight Consolidation (EWC)

Important algorithmEWC is one of the most important classical continual learning methods.

### Q28. What is Elastic Weight Consolidation?

Answer: Elastic Weight Consolidation (EWC) is a regularisation-based continual learning method that protects parameters considered important for previously learned tasks.

Instead of allowing all model parameters to change freely when learning a new task, EWC penalises large changes to important parameters.

### Q29. What is the main idea behind EWC?

Answer: Some neural network parameters are more important than others for solving a particular task.

EWC estimates parameter importance using the Fisher Information Matrix.

Parameters with high estimated importance should be changed less during subsequent learning.

### Q30. What is the Fisher Information Matrix?

Answer: The Fisher Information Matrix measures how sensitive the model's likelihood is to changes in its parameters.

In a practical EWC implementation, we often use a diagonal approximation:

$$
F_i\approx
\mathbb{E}
\left[
\left(
\frac{\partial \log p(y|x,\theta)}
{\partial\theta_i}
\right)^2
\right]
$$

A larger Fisher value indicates that the corresponding parameter is considered more important for the task.

### Q31. What is the EWC loss function?

Answer: EWC adds a regularisation penalty to the ordinary task loss:

$$
\boxed{
L(\theta)=
L_{\text{new}}(\theta)
+
\frac{\lambda}{2}
\sum_i
F_i(\theta_i-\theta_i^*)^2
}
$$

Where:

- $L_{\text{new}}$: Classification loss on the new task.
- $\lambda$: Regularisation strength.
- $F_i$: Importance of parameter $i$.
- $\theta_i^*$: Parameter value learned on the previous task.
- $\theta_i$: Current parameter value.

The penalty discourages important parameters from moving too far away from their previous values.

### Q32. What does the EWC regularisation parameter $\lambda$ control?

Answer: It controls the strength of the protection against forgetting.

- Small $\lambda$: More freedom to learn new tasks, but potentially more forgetting.
- Large $\lambda$: Stronger protection of previous knowledge, but potentially weaker learning of new tasks.

This creates a stability-plasticity trade-off.

### Q33. How is EWC implemented in PyTorch?

A simplified Fisher estimation procedure looks like this:

python

```python
fisher = {
    name: torch.zeros_like(param)
    for name, param in model.named_parameters()
}

model.eval()

for images, labels in previous_task_loader:

    model.zero_grad()

    logits = model(images)

    loss = F.cross_entropy(
        logits, labels
    )

    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            fisher[name] += param.grad.detach() ** 2
```

This accumulates squared gradients as an approximation to the diagonal Fisher information.

A real implementation should normalise the accumulated estimates and handle the data and model state consistently.

### Q34. How is the EWC penalty calculated?

python

```python
ewc_loss = 0.0

for name, param in model.named_parameters():

    ewc_loss += (
        fisher[name]
        * (param - old_parameters[name]) ** 2
    ).sum()

total_loss = (
    task_loss
    + (lambda_ewc / 2) * ewc_loss
)
```

The EWC penalty is added to the ordinary classification loss.

Exam tip: Remember the essential principle:

EWC protects important old parameters by penalising changes to them.

## Part 6 — Replay-Based Methods: Experience Replay

Memory-based continual learning### Q35. What is Experience Replay?

Answer: Experience Replay is a continual learning strategy that stores a limited collection of examples from previous tasks and reuses them during training on new tasks.

Rather than learning only from the current task, the model trains using both:

- New task examples.
- Stored examples from previous tasks.

This helps preserve earlier knowledge.

### Q36. What is a replay buffer?

Answer: A replay buffer is a memory structure that stores selected examples from previously encountered tasks.

For example:

python

```python
memory = {
    "images": [],
    "labels": []
}
```

The buffer has a limited capacity because storing every previous training example may be impractical.

### Q37. How does Experience Replay work?

Answer: A typical training iteration combines current task data with replay-buffer examples.

python

```python
for images, labels in current_loader:

    replay_x, replay_y = sample_memory(
        memory,
        batch_size=32
    )

    images = torch.cat(
        [images, replay_x], dim=0
    )

    labels = torch.cat(
        [labels, replay_y], dim=0
    )

    optimizer.zero_grad()

    logits = model(images)

    loss = F.cross_entropy(
        logits, labels
    )

    loss.backward()

    optimizer.step()
```

The model receives training signals from both old and new examples.

### Q38. Why does replay reduce catastrophic forgetting?

Answer: Because the model is periodically reminded of previous tasks.

When learning a new task, the model must also maintain good predictions on stored examples from earlier tasks.

This discourages parameter changes that would completely destroy old knowledge.

### Q39. What are the advantages and disadvantages of Experience Replay?

| Advantages | Disadvantages |
| --- | --- |
| Often effective at reducing forgetting. | Requires memory storage. |
| Conceptually straightforward. | Must decide which examples to retain. |
| Can be combined with other methods. | Performance depends on buffer size and selection strategy. |
| Uses actual previous examples. | May raise privacy or data-retention concerns. |

### Q40. What is the difference between EWC and Experience Replay?

Answer:

- EWC protects important model parameters through a mathematical regularisation penalty.
- Experience Replay retains selected previous examples and uses them again during training.

This is a fundamental distinction between regularisation-based and replay-based approaches.

## Part 7 — Knowledge Distillation and Learning Without Forgetting (LwF)

Teacher-student learning### Q41. What is Knowledge Distillation?

Answer: Knowledge Distillation is a technique in which a student model learns to reproduce the predictions or knowledge of a teacher model.

Instead of learning exclusively from hard labels, the student can also learn from the teacher's probability distributions.

### Q42. What is Learning Without Forgetting (LwF)?

Answer: Learning Without Forgetting is a continual learning approach that uses knowledge distillation to help preserve the previous model's behaviour while learning a new task.

The old model acts as a teacher, while the updated model acts as a student.

The new model learns:

- The new task using ordinary classification loss.
- To preserve previous predictions using a distillation loss.

### Q43. What is the LwF loss function?

A simplified form is:

$$
L_{\text{total}}=
L_{\text{new}}+
\lambda L_{\text{distill}}
$$

The distillation loss can use temperature-scaled probability distributions:

$$
L_{\text{distill}}
=
T^2
D_{\mathrm{KL}}
\left(
p_{\mathrm{old}}^{T}
\parallel
p_{\mathrm{new}}^{T}
\right)
$$

Where:

- $D_{\mathrm{KL}}$: Kullback-Leibler divergence.
- $T$: Distillation temperature.
- $p_{\mathrm{old}}^T$: Old model's softened predictions.
- $p_{\mathrm{new}}^T$: Updated model's softened predictions.

The temperature allows the model to learn from relative probabilities across classes rather than only the highest-probability prediction.

### Q44. What is the teacher model's role?

Answer: The teacher model preserves information about previously learned tasks.

It provides reference predictions that the updated model attempts to reproduce.

### Q45. What is the difference between EWC, Experience Replay and LwF?

| Method | Main mechanism |
| --- | --- |
| EWC | Protect important parameters |
| Experience Replay | Reuse stored previous examples |
| LwF | Preserve previous model predictions through distillation |

Exam tip: These three methods address catastrophic forgetting through three different mechanisms. Be able to explain each without confusing them.

## Part 8 — Comparing Continual Learning Methods and Practical Experiments

Experimental analysis### Q46. Why do we compare multiple continual learning algorithms?

Answer: Different algorithms manage forgetting in different ways.

Comparing them helps determine how well they balance learning new tasks and retaining old knowledge.

### Q47. What should the experimental comparison include?

A representative experiment could compare:

1. Sequential training without protection.
2. EWC.
3. Experience Replay.
4. LwF.

The same dataset split, model architecture and evaluation procedure should be used wherever possible.

### Q48. What are the main hyperparameters?

| Method | Important hyperparameters |
| --- | --- |
| Baseline | Learning rate, epochs, batch size |
| EWC | $\lambda$, Fisher estimation samples |
| Experience Replay | Buffer capacity, replay batch size |
| LwF | Distillation temperature, distillation weight |

### Q49. Why is the stability-plasticity dilemma important?

Answer: It describes the challenge of balancing the ability to learn new information (plasticity) with the ability to preserve existing knowledge (stability).

- Too much plasticity can lead to catastrophic forgetting.
- Too much stability can prevent the model from learning new tasks effectively.

A successful continual learning algorithm must balance both.

### Q50. How should we interpret experimental results?

Consider the following illustrative results:

| Method | Average accuracy | Average forgetting |
| --- | --- | --- |
| Sequential baseline | 48% | 35% |
| EWC | 57% | 24% |
| Experience Replay | 67% | 12% |
| LwF | 59% | 21% |

These numbers are hypothetical examples for explaining result interpretation.

Experience Replay performs best in this example because it achieves higher average accuracy and lower forgetting.

However, the best method depends on the dataset, task sequence, architecture, memory constraints and hyperparameters.

### Q51. What plots should be included in the practical report?

Recommended plots:

- Accuracy after each task.
- Average accuracy across learning stages.
- Forgetting across tasks.
- Comparison of different continual learning methods.
- Training loss for the different methods.

These plots help explain not only which method performs best, but also how its performance changes as additional tasks are learned.

## Part 9 — Advanced Understanding and Exam Preparation

Oral examination and written assessmentThis final part consolidates the key ideas and tests whether you understand the mechanisms behind the algorithms.

### Q52. Why is catastrophic forgetting difficult to solve?

Answer: Neural networks generally use shared parameters to represent information from multiple tasks.

When parameters are updated to improve performance on new data, those changes can interfere with previously acquired knowledge.

The challenge is to preserve useful old representations while allowing sufficient parameter adaptation for new tasks.

### Q53. What is the difference between stability and plasticity?

Answer:

- Stability is the ability to preserve previously acquired knowledge.
- Plasticity is the ability to learn new knowledge.

Continual learning aims to balance these competing requirements.

### Q54. Which continual learning method uses the Fisher Information Matrix?

Answer: Elastic Weight Consolidation (EWC).

It estimates parameter importance using Fisher information and penalises changes to important parameters.

### Q55. Which method uses a memory buffer?

Answer: Experience Replay.

It stores examples from earlier tasks and mixes them with current training data.

### Q56. Which method uses teacher-student knowledge distillation?

Answer: Learning Without Forgetting (LwF).

It attempts to preserve the previous model's predictions while learning a new task.

### Q57. Why is Class-Incremental Learning particularly challenging?

Answer: The model must classify examples from all previously learned and newly introduced classes without receiving the task identity.

It must therefore maintain a common classification space across the complete class sequence.

### Q58. Can a model have high accuracy on the latest task but still be a poor continual learning model?

Answer: Yes.

A model could perform extremely well on the latest task while having forgotten most earlier tasks.

This is why average accuracy and forgetting are important evaluation metrics.

### Q59. What is the most important difference between EWC and Replay?

Answer: EWC protects parameter values based on estimated importance, whereas Replay uses actual examples from previous tasks to remind the model of earlier knowledge.

### Q60. What are the five most important things to remember for an oral examination?

1. Catastrophic forgetting: Learning new tasks can damage performance on previous tasks.
2. Continual learning scenarios: Task-IL, Class-IL and Domain-IL define different learning and evaluation settings.
3. EWC: Protects important parameters using Fisher information.
4. Experience Replay: Stores and reuses examples from previous tasks.
5. Evaluation: Average accuracy and forgetting measure how well the model learns new tasks while retaining old knowledge.

# 3. Summary of the proposed nine-part structure

Here is the complete roadmap we will use to organise Practical 7.

| Part | Subject | Main concepts / algorithms |
| --- | --- | --- |
| 1 | Understanding Continual Learning | Catastrophic forgetting, tasks, sequential learning |
| 2 | Learning Scenarios and Experimental Setup | Task-IL, Class-IL, Domain-IL, CIFAR-10/100 |
| 3 | Baseline Sequential Training | CNN, PyTorch training, catastrophic forgetting |
| 4 | Evaluation Metrics | Accuracy matrix, Average Accuracy, Forgetting, BWT, FWT |
| 5 | Regularisation-Based Methods | Elastic Weight Consolidation (EWC), Fisher Information |
| 6 | Replay-Based Methods | Experience Replay, replay buffer, memory management |
| 7 | Knowledge Distillation | Knowledge Distillation, Learning Without Forgetting (LwF) |
| 8 | Experimental Comparison | Hyperparameters, plots, results, stability-plasticity |
| 9 | Advanced Understanding and Exam Preparation | Algorithm comparisons, mathematical reasoning, oral examination |

## 4. Important assumptions about our reference notebook

To make the practical realistic, I suggest the following experimental configuration.

| Component | Proposed configuration |
| --- | --- |
| Framework | PyTorch |
| Dataset | CIFAR-10 initially |
| Optional dataset | CIFAR-100 |
| Backbone | CNN |
| Learning setup | Sequential task learning |
| Number of tasks | 5 tasks |
| Classes per task | 2 for CIFAR-10 |
| Baseline | Sequential training |
| Method 1 | EWC |
| Method 2 | Experience Replay |
| Method 3 | Learning Without Forgetting |
| Main metric | Average accuracy |
| Forgetting metric | Average forgetting |
| Additional metrics | BWT and FWT |

A few implementation details will need to be made explicit when we eventually construct the notebook:

- Whether we use Task-IL or Class-IL, since the output-head design and evaluation protocol differ.
- Whether the replay buffer is allowed to store previous training examples.
- Whether LwF uses a frozen teacher and how old-class logits are handled.
- Whether EWC uses online or task-specific Fisher estimates.
- Whether we evaluate every previously learned task after each new task.

These decisions matter because two experiments using the same algorithm can produce different results under different continual learning protocols.

## 5. How we will proceed

I suggest we adopt the following sequence:

1. Complete the questions-focused summary
   Use the nine parts above as our reference study material. We can refine individual explanations and code when needed.
2. Generate MCQs for Part 1
   Four options per question, correct answer and detailed explanation, following the Practical 6 format.
3. Generate the interactive Part 1 practice exam
   Include answer selection, submission, score calculation, feedback and explanations.
4. Repeat for Parts 2–9
   Complete Step 1 and Step 2 for each part before moving to the next.
5. Final examination preparation
   Combine the material into a comprehensive revision and examination practice covering Practical 7.

## 6\. Regarding your Practical 6 notebook example

Yes, please paste the Practical 6 notebook example when convenient.

It will help establish the expected structure of a course practical, particularly:

- How datasets are loaded and divided.
- How tasks are generated.
- How models and training loops are organised.
- How evaluation functions are implemented.
- How students are expected to complete the missing code.
- How results and plots are presented.

We can then make our assumed Practical 7 notebook follow a similar teaching and implementation style, rather than producing an entirely theoretical continual learning tutorial.

My recommendation: Use the nine-part structure above as our initial Practical 7 study framework. Once you provide the Practical 6 notebook example, we can refine the assumed Practical 7 structure to make it closer to the actual course conventions.

The ultimate goal remains the same as Practical 6: understand the theory, understand the code, be able to explain the algorithms, and be well prepared for the examination.
