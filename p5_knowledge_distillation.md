# Advanced Machine Learning — Practical 5: Knowledge Distillation

Let's explain this assignment using a question-and-answer approach, as if you were preparing for a lab session, an oral exam, or a written assessment.

The main idea is simple:

Can we train a small neural network to perform better by teaching it what a larger, more powerful neural network has learned?

You will use PyTorch and CIFAR-10 images to investigate this. The assignment has three tasks:

1. Implement temperature-scaled softmax.
2. Implement the knowledge distillation loss using KL divergence.
3. Implement the teacher–student training loop and compare the results.

Before examining the code, let's build an understanding of the concepts behind it.

# Part 1 — Understanding the assignment

## Q1. What is knowledge distillation?

Answer: Knowledge distillation is a machine learning technique in which a larger model, called the teacher, helps train a smaller model, called the student.

Imagine that you want a computer to classify images into ten categories, such as cats, dogs, cars, and airplanes.

You have two models:

Hundeschule Sarah Neumann Köln-Nord und Umgebung

Teacher model

- Larger and more powerful.
- Has more parameters and learning capacity.
- Learns useful patterns from the training data.
- Provides additional information to the student.

Hundeschule Sarah Neumann Köln-Nord und Umgebung

Student model

- Smaller and computationally cheaper.
- Has fewer parameters.
- Learns from the true labels and the teacher's predictions.
- Tries to achieve higher accuracy than it would independently.

The teacher is not necessarily teaching the student through explanations in natural language. Instead, it provides a probability distribution over the possible classes.

For example, given an image of a dog, the teacher might predict:

| Class | Teacher probability |
| --- | --- |
| Dog | 75% |
| Cat | 12% |
| Horse | 7% |
| Bird | 4% |
| Other classes | 2% |

The correct label is simply dog. However, the teacher also indicates that the image resembles a cat or a horse more than it resembles unrelated objects.

This extra information is sometimes called dark knowledge. It can help the student learn relationships between classes that are not explicitly represented by the hard labels.

## Q2. What are the three models or training stages in the assignment?

Answer: You will compare three training outcomes.

Stage 1 — Independent student

Train the small model using only the true labels. This establishes the baseline.

Objective: ordinary supervised learning.

Stage 2 — Teacher

Train the larger model using the same training data and true labels.

Objective: establish a stronger model that can teach the student.

Stage 3 — Distilled student

Train a new small model using both the true labels and the teacher's predictions.

Objective: outperform the independent student while keeping the small architecture.

The comparison is important because it allows you to investigate whether the improvement comes from knowledge distillation rather than simply using a larger network.

## Q3. What is CIFAR-10, and why does the assignment use only 30% of its training data?

Answer: CIFAR-10 is an image classification dataset containing 60,000 colour images, each of size $32 \times 32$ pixels. It has ten classes, including airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

The notebook creates this split:

| Split | Percentage | Number of images |
| --- | --- | --- |
| Training | 30% | 18,000 |
| Validation | 20% | 12,000 |
| Discarded | 50% | 30,000 |

The reduced dataset makes the practical faster and helps illustrate the value of transferring knowledge when labelled data is limited.

The training data updates the model's weights. The validation data measures performance on images that were not used for those updates.

Important: The validation set is not used as a source of teaching labels. The teacher and student learn from the same training examples.

## Q4. What is the difference between the teacher and the student architectures?

Answer: Both use the custom `BasicResnet` architecture, but the teacher has wider and deeper convolutional layers.

Your notebook specifies:

python
```
student_conv_layers = [32, 64, 128]

teacher_conv_layers = [64, 128, 128, 256, 256]
teacher_dropout = 0.2
```

These lists specify the number of output channels in successive residual blocks.

- The student has three residual blocks.
- The teacher has five residual blocks.
- The teacher has more channels in several layers and therefore more parameters.
- The teacher uses spatial dropout to help reduce overfitting.

A residual block combines a transformed input with a shortcut connection:

$$
h(x)=\operatorname{ReLU}(F(x)+x)
$$

Here, $F(x)$ represents the learned convolutional transformation. If the input and output dimensions differ, the implementation uses a projection shortcut instead of adding the original input directly.

The teacher's greater capacity gives it the potential to learn more complex patterns. However, a larger model is not guaranteed to be better unless it is trained appropriately.

# Part 2 — The predictions the models produce

## Q5. What are logits, and how are they different from probabilities?

Answer: Logits are the raw numerical outputs of a model's final classification layer, before softmax is applied.

Suppose the student produces the following logits for three classes:

$$
z=[4,\ 2,\ 0]
$$

These numbers are not probabilities. They can be negative, positive, or greater than one, and they do not have to sum to one.

To convert them into probabilities, apply softmax:

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

The resulting probabilities are approximately:

| Class | Logit | Probability |
| --- | --- | --- |
| A | 4 | 86.7% |
| B | 2 | 11.7% |
| C | 0 | 1.6% |

The largest logit corresponds to the most probable class.

In PyTorch:

python
```
logits = model(images)
probabilities = torch.softmax(logits, dim=1)
```

The argument `dim=1` means softmax is applied across the class dimension for each example in the batch.

Why does this matter? The assignment uses logits for ordinary cross-entropy training, but it compares probability distributions when calculating the distillation loss.

## Q6. What are hard labels and soft labels?

Answer: They are two different ways of representing the target for a classification problem.

A hard label gives one correct class:

$$
y=[0,0,1,0]
$$

This represents class 2, assuming zero-based indexing.

A soft label assigns probabilities to several classes:

$$
p=[0.05,0.10,0.75,0.10]
$$

The teacher considers class 2 most likely, but also assigns some probability to other classes.

The key difference is that a hard label tells the student which answer is correct, while a soft label also conveys how the teacher views the alternatives.

In the assignment, the hard labels come from CIFAR-10, whereas the soft labels come from the teacher network.

# Part 3 — Task 1: Temperature-scaled softmax

## Q7. Why do we need temperature?

Answer: The teacher can be very confident in its predictions. When the probability of the correct class is close to 100%, the probabilities of the other classes can become extremely small.

That makes it harder for the student to learn the relationships between different classes.

Temperature, denoted by $T$, controls how soft or spread out the probability distribution is.

The formula is:

$$
p_i=\frac{\exp(z_i/T)}
{\sum_j\exp(z_j/T)}
$$

where:

- $z_i$ is the logit for class $i$.
- $T$ is the temperature.
- $p_i$ is the resulting probability for class $i$.

Consider the logits $[4,2,0]$.

Explore the effect of temperature

Temperature

## 1

1

5

10

20

Class A

86.7%

Class B

11.7%

Class C

1.6%

Notice that increasing the temperature makes the probabilities more uniform. The most likely class remains the same for these logits.

When $T=1$, you get ordinary softmax. As $T$ increases, the differences between the logits become less pronounced, so the distribution becomes softer.

As $T$ approaches infinity, the distribution approaches a uniform distribution over the classes. Very large temperatures can therefore make the teacher's predictions less informative.

### Q8. How do you implement Task 1 in PyTorch?

Answer: The assignment asks you to implement softmax after dividing the logits by the temperature.

Complete the function as follows:

python
```
def temperature_softmax(x, T):
    """Apply temperature-scaled softmax over the class dimension."""
    return F.softmax(x / T, dim=1)
```

That's all you need for Task 1.

Let's examine each part:

- `x / T` rescales the logits.
- `F.softmax(...)` converts the rescaled logits into probabilities.
- `dim=1` applies softmax across classes for each image.

For a batch of 256 images and ten classes, the input and output shapes are both `(256, 10)`.

Each output row sums to approximately 1.

### Q9. What should you check to verify Task 1?

Answer: Run the notebook's plotting code:

python
```
temp = 20
teacher_soft_temp = temperature_softmax(teacher_pred, temp)

plot_distribution(
    teacher_soft_temp[0],
    f'Teacher softmax with temperature={temp}'
)
```

You should see a distribution that is generally flatter than the ordinary softmax distribution.

You can also run these checks:

python
```
assert teacher_soft_temp.shape == teacher_pred.shape
assert torch.allclose(
    teacher_soft_temp.sum(dim=1),
    torch.ones(teacher_pred.shape[0], device=teacher_pred.device),
    atol=1e-5
)
assert torch.all(teacher_soft_temp >= 0)
```

These verify that the shape is preserved, probabilities sum to one, and no probability is negative.

# Part 4 — Task 2: Distillation loss

## Q10. What is a loss function, and why do we need a new one?

Answer: A loss function measures how different a model's predictions are from a target.

During ordinary supervised training, your student uses cross-entropy loss:

$$
L_{\text{hard}}=\operatorname{CE}(y,z_s)
$$

Here, $y$ represents the true class labels and $z_s$ represents the student's logits.

This encourages the student to predict the correct CIFAR-10 class.

Knowledge distillation introduces a second objective: make the student's output distribution resemble the teacher's distribution.

We call this the distillation loss, or soft loss:

$$
L_{\text{soft}}=D_{\mathrm{KL}}(p_t\parallel p_s)
$$

Here, $p_t$ is the teacher's temperature-scaled distribution and $p_s$ is the student's temperature-scaled distribution.

The two losses teach different things:

- Hard loss: learn from the dataset's true labels.
- Soft loss: learn from the teacher's predictions.

The final student will learn from both.

## Q11. What is KL divergence?

Answer: Kullback–Leibler (KL) divergence measures how different one probability distribution is from another.

Its general formula is:

$$
D_{\mathrm{KL}}(P\parallel Q)
=\sum_i P_i\log\frac{P_i}{Q_i}
$$

For example, suppose the teacher predicts:

$$
P=[0.7,0.2,0.1]
$$

and the student predicts:

$$
Q=[0.4,0.4,0.2]
$$

The distributions differ, so their KL divergence is positive. If the distributions are identical, the divergence is zero.

For distillation, we want to reduce the discrepancy between the teacher and student.

A subtle but important point: KL divergence is asymmetric. In this assignment, the usual teacher-to-student objective is $D_{\mathrm{KL}}(p_t\parallel p_s)$. This makes the teacher the target distribution and the student the distribution being adjusted.

The assignment's displayed formula labels the distributions in the opposite order in one place. For standard response-based distillation, use the teacher as the KL target and the student as the predicted distribution.

## Q12. Why multiply the KL divergence by $T^2$?

Answer: Increasing the temperature softens the distributions, but it also reduces the magnitude of the gradients used to train the student.

For the standard temperature-scaled distillation formulation, gradient magnitudes can decrease roughly in proportion to $1/T^2$. Multiplying the loss by $T^2$ compensates for this effect.

The complete formula is:

$$
\boxed{
L_{\text{soft}}=
T^2D_{\mathrm{KL}}(p_t^{(T)}\parallel p_s^{(T)})
}
$$

where

$$
p_t^{(T)}=\operatorname{softmax}(z_t/T)
$$

and

$$
p_s^{(T)}=\operatorname{softmax}(z_s/T)
$$

The superscript $T$ here denotes temperature, not a matrix transpose.

Without the $T^2$ factor, changing the temperature would also change the scale of the distillation gradients, complicating comparisons between temperature settings.

## Q13. How do we implement the distillation loss?

Answer: PyTorch provides `F.kl_div()`, which expects the first argument to contain log probabilities and the second argument to contain target probabilities.

This detail is important because passing two ordinary probability distributions directly to `F.kl_div()` would be incorrect.

Here is a correct implementation for Task 2:

python
```
def distillation_output_loss(
    student_pred,
    teacher_pred,
    temperature
):
    # Convert logits into temperature-scaled distributions.
    student_log_probs = F.log_softmax(
        student_pred / temperature, dim=1
    )

    teacher_probs = F.softmax(
        teacher_pred / temperature, dim=1
    )

    # Standard teacher-to-student KL divergence.
    soft_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='batchmean'
    )

    # Compensate for temperature-related gradient scaling.
    soft_loss = soft_loss * (temperature ** 2)

    return soft_loss
```

Let's understand the critical lines.

| Code | Purpose |
| --- | --- |
| `F.log_softmax(student_pred / temperature, dim=1)` | Computes the student's log probabilities. |
| `F.softmax(teacher_pred / temperature, dim=1)` | Computes the teacher's target probabilities. |
| `F.kl_div(..., reduction='batchmean')` | Computes KL divergence averaged over the batch. |
| `* (temperature ** 2)` | Applies the temperature compensation. |

### Why use `reduction='batchmean'`?

Suppose a batch contains 256 images and ten classes. We want the sum of the KL contributions across the classes, averaged over the 256 images.

`batchmean` divides by the batch size, rather than dividing by the total number of class entries.

This produces a loss scale that is more suitable for comparing with cross-entropy.

### Q14. Why is the teacher distribution the second argument to `F.kl_div()`?

Answer: PyTorch's convention is effectively:

python
```
F.kl_div(log_Q, P)
```

which computes:

$$
D_{\mathrm{KL}}(P\parallel Q)
$$

Therefore:

- `log_Q` is the student's log probability distribution.
- `P` is the teacher's probability distribution.

The loss encourages the student to match the teacher.

The teacher's outputs should not receive gradients during distillation. We will handle that in Task 3 using `torch.no_grad()`.

### Q15. How do you check Task 2?

Answer: Run the notebook's existing test code:

python
```
temp = 20

hard_loss = nn.CrossEntropyLoss()(
    student_pred[:1],
    example_y[:1].to(device)
)

soft_loss = distillation_output_loss(
    student_pred[:1],
    teacher_pred[:1],
    temp
)

assert soft_loss.shape == ()
assert soft_loss >= 0

print(f"Hard loss: {hard_loss.item():.4f}")
print(f"Soft loss: {soft_loss.item():.4f}")
```

The loss should be a scalar tensor, not a vector. It should be nonnegative up to possible tiny floating-point errors.

The hard loss and soft loss do not have to be numerically identical. The notebook's suggestion that they should be of a similar scale is a useful practical guideline, not a mathematical guarantee.

# Part 5 — Task 3: Teacher–student distillation training

## Q16. What is the objective of Task 3?

Answer: Train a new student model with the same architecture as the independent student, but now use both the hard loss and the soft loss.

The total training loss is:

$$
L_{\text{total}}
=\alpha L_{\text{hard}}
+(1-\alpha)L_{\text{soft}}
$$

Here, $\alpha$ controls the relative importance of the two objectives.

- If $\alpha=1$, the student ignores the teacher and trains only on hard labels.
- If $\alpha=0$, the student trains only to imitate the teacher.
- If $\alpha=0.5$, both losses receive equal weighting.

For this practical, a reasonable starting point is $\alpha=0.5$, with temperature $T=20$.

These are starting hyperparameters, not guaranteed optimal values.

## Q17. What happens during one training iteration?

Answer: Each batch follows this process:

1. Load a batch of CIFAR-10 images

Images and ground-truth labels

Teacher

Forward pass only

Teacher logits

No weight updates

Student

Forward pass

Student logits

Trainable weights

Hard loss

Student vs. true labels

Soft loss

Student vs. teacher

3. Combine the losses

$\alpha L_{\text{hard}}+(1-\alpha)L_{\text{soft}}$

4. Backpropagate and update the student

Repeat for every batch and epoch.

The key is that the teacher's predictions are used as training targets, but the teacher's parameters remain fixed.

## Q18. How should we implement `distil_model()`?

Answer: We can reuse the structure of `train_model()`, adding the teacher forward pass, the two losses, and the combined optimization objective.

The following is a complete implementation compatible with the notebook's existing `TrainingMetrics`, `evaluate_model()`, and `get_batch_acc()` helpers.

python
```
def distil_model(
    student: nn.Module,
    teacher: nn.Module,
    temperature: float,
    alpha: float,
    train_loader,
    val_loader,
    num_epochs: int,
    lr: float,
    l2_reg: float = 0.0,
    gamma: float = 1.0,
    baseline_accs=None,
    baseline_metrics=None,
    dynamic_plot: bool = True,
    run_name: str = "distilled student",
    seed: int = 0,
):
    """Train a student using hard labels and teacher predictions."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    torch.manual_seed(seed)

    hard_loss_func = nn.CrossEntropyLoss()

    # Only the student parameters are optimized.
    opt = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
        weight_decay=l2_reg
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        opt, gamma=gamma
    )

    metrics = TrainingMetrics()

    # The teacher is fixed and its BatchNorm layers use
    # their stored running statistics.
    teacher.eval()

    for epoch in range(num_epochs):

        # Student is in training mode.
        student.train()
        teacher.eval()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()

            # Teacher provides targets, not trainable outputs.
            with torch.no_grad():
                teacher_pred = teacher(x)

            # Student prediction.
            student_pred = student(x)

            # Learn from the ground-truth labels.
            hard_loss = hard_loss_func(student_pred, y)

            # Learn from the teacher's probability distribution.
            soft_loss = distillation_output_loss(
                student_pred,
                teacher_pred,
                temperature
            )

            # alpha weights the hard loss.
            total_loss = (
                alpha * hard_loss
                + (1.0 - alpha) * soft_loss
            )

            # Update student weights using both objectives.
            total_loss.backward()
            opt.step()

            batch_acc = get_batch_acc(student_pred, y)

            metrics.log_train(
                loss=float(hard_loss.item()),
                acc=float(batch_acc),
                soft_loss=float(soft_loss.item())
            )

        scheduler.step()

        # Evaluate using the true labels and ordinary CE loss.
        val_loss, val_acc = evaluate_model(
            student,
            val_loader,
            hard_loss_func
        )

        metrics.log_val(
            loss=float(val_loss),
            acc=float(val_acc)
        )

        if dynamic_plot:
            metrics.plot(
                title=f"Training ({run_name})",
                live=True,
                max_step=len(train_loader) * num_epochs,
                baseline_accs=baseline_accs,
                baseline_metrics=baseline_metrics
            )

        else:
            train_loss, epoch_val_loss = metrics.epoch_loss
            train_acc, epoch_val_acc = metrics.epoch_acc

            print(
                f"epoch: {epoch + 1:>3} | "
                f"loss: {train_loss:.3f} | "
                f"{epoch_val_loss:.3f} | "
                f"acc: {train_acc:.1%} | "
                f"{epoch_val_acc:.1%}"
            )

    if not dynamic_plot:
        metrics.plot(
            title=f"Training ({run_name})",
            live=False,
            baseline_accs=baseline_accs,
            baseline_metrics=baseline_metrics
        )

    print(
        f"Best validation accuracy ({run_name}): "
        f"{metrics.best_val_acc:.2%}"
    )

    return metrics
```

### Q19. Why are these implementation details important?

Answer: Several lines protect the experiment from common mistakes.

- `teacher.eval()` disables training behaviour such as BatchNorm statistic updates and dropout.
- `with torch.no_grad()` prevents the teacher forward pass from building a gradient computation graph.
- `opt = AdamW(student.parameters(), ...)` ensures that only the student is optimized.
- `total_loss.backward()` computes gradients from both learning objectives.
- `evaluate_model()` evaluates the student using the actual class labels, not the teacher's predictions.

One subtle detail is that `metrics.log_train()` records the hard loss and soft loss separately, which allows the plotting helper to visualize both. The optimizer, however, uses the combined loss.

The validation loss remains ordinary cross-entropy, so the final validation accuracy measures actual classification performance rather than how closely the student imitates the teacher.

## Q20. How do we launch the distillation experiment?

Answer: Replace the unfinished training section with the following:

python
```
# Distillation hyperparameters
temperature = 20
alpha = 0.5

# Create a fresh student with the same architecture as the baseline.
student = BasicResnet(student_conv_layers).to(device)

# Restore the pretrained teacher.
teacher_state_dict = torch.load(
    teacher_save_name,
    map_location=device
)
teacher.load_state_dict(teacher_state_dict)

# Freeze the teacher's parameters.
teacher.requires_grad_(False)
teacher.eval()

# Train the distilled student using the baseline hyperparameters.
student_metrics = distil_model(
    student=student,
    teacher=teacher,
    temperature=temperature,
    alpha=alpha,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=student_epochs,
    lr=student_lr,
    l2_reg=student_l2,
    gamma=gamma,
    baseline_accs={
        "independent student": ind_student_metrics.best_val_acc,
        "teacher": teacher_metrics.best_val_acc
    },
    baseline_metrics=ind_student_metrics,
    dynamic_plot=True,
    run_name="teacher-student distillation"
)
```

This uses the same student architecture and the same main training hyperparameters as the independent student, making the comparison fairer.

The new student is initialized from scratch rather than from the baseline student's weights. Otherwise, the experiment would combine distillation with continued training of the existing student.

# Part 6 — Evaluating your results

## Q21. How do we know whether knowledge distillation worked?

Answer: Compare the best validation accuracies of all three models:

1. Independent student — baseline.
2. Teacher — upper performance reference.
3. Distilled student — model we want to improve.

Run the notebook's comparison code:

python
```
student_metrics.plot(
    title="Teacher-student distillation",
    live=False,
    baseline_accs={
        "baseline": ind_student_metrics.best_val_acc,
        "teacher": teacher_metrics.best_val_acc
    },
    baseline_metrics=ind_student_metrics
)

print(
    "Independent student:",
    f"{ind_student_metrics.best_val_acc:.2%}"
)

print(
    "Teacher:",
    f"{teacher_metrics.best_val_acc:.2%}"
)

print(
    "Distilled student:",
    f"{student_metrics.best_val_acc:.2%}"
)
```

The plots show how the training and validation losses and accuracies evolve. The horizontal accuracy references help you see whether the distilled student surpasses the baseline and how close it gets to the teacher.

### Q22. How do we calculate the improvement?

Answer: Suppose your experiment produces the following illustrative results. These are examples, not measured results from your notebook.

Example validation accuracies

Independent student

### 67%

Distilled student

### 73%

Teacher

### 78%

The distilled student has improved by:

$$
73\%-67\%=6\text{ percentage points}
$$

The teacher–student baseline gap is:

$$
78\%-67\%=11\text{ percentage points}
$$

The fraction of the original gap recovered is:

$$
\frac{73-67}{78-67}\times100
\approx54.5\%
$$

This means the distilled student recovers about 54.5% of the performance gap between the independent student and the teacher.

Important distinction: An improvement from 67% to 73% is a gain of 6 percentage points, or approximately an 8.96% relative improvement. The assignment's target of a 5% accuracy boost is best interpreted as a gain of five percentage points.

# Part 7 — Common problems and how to solve them

## Q23. What if the distilled student performs worse than the baseline?

Answer: This can happen. Knowledge distillation does not guarantee better accuracy.

Try the following experiments, changing one parameter at a time:

| Experiment | What to try | Why? |
| --- | --- | --- |
| Temperature | $T=5,10,20$ | Controls how much information the teacher shares about alternative classes. |
| Loss weighting | $\alpha=0.3,0.5,0.7$ | Controls the balance between true labels and teacher supervision. |
| Training stability | Check training and validation curves | Identifies unstable learning or overfitting. |
| Teacher quality | Check teacher validation accuracy | A weak teacher may provide less useful guidance. |

For example, if the teacher's predictions are useful but imperfect, increasing the hard-loss weight may help the student prioritize the true labels.

If the teacher's class relationships provide useful additional information, a lower hard-loss weight may help the student benefit more from distillation.

Keep the architecture and other training settings unchanged when comparing experiments, and record each configuration and its best validation accuracy.

## Q24. What are the most important implementation mistakes to avoid?

Answer: Check these details before drawing conclusions from your results.

- Incorrect KL divergence direction: Use `F.kl_div(student_log_probs, teacher_probs)` for the standard teacher-to-student objective.
- Missing temperature scaling: Divide both teacher and student logits by the same temperature.
- Missing $T^2$ correction: Multiply the KL loss by `temperature ** 2`.
- Updating the teacher: Optimize only `student.parameters()` and calculate teacher predictions inside `torch.no_grad()`.
- Incorrect model mode: Use `student.train()` for training and `teacher.eval()` throughout distillation.
- Unfair comparisons: Keep the student architecture, learning rate, epoch count, regularization and data splits consistent with the baseline.
- Incorrect validation objective: Measure validation performance against the true labels using ordinary cross-entropy.

One additional detail: the supplied plotting code expects `metrics.epoch_loss` and `metrics.epoch_acc` to be properties, so they should be accessed without parentheses. The `distil_model()` implementation above handles this correctly.

# Part 8 — Understanding the assignment at an advanced level

## Q25. What is the fundamental machine learning concept being tested?

Answer: The assignment tests whether you understand how to transfer information from a high-capacity model into a smaller model through its output distribution.

There are three connected concepts:

1. Supervised learning: The independent student learns from hard labels.
2. Probabilistic knowledge transfer: The teacher produces soft labels, and KL divergence measures the mismatch between the teacher's and student's distributions.
3. Multi-objective optimization: The distilled student minimizes a weighted combination of the hard and soft losses.

The optimization objective is:

$$
\min_{\theta_s}
\left[
\alpha L_{\mathrm{hard}}(\theta_s)
+(1-\alpha)T^2
D_{\mathrm{KL}}
\left(
p_t^{(T)}\parallel p_s^{(T)}
\right)
\right]
$$

Here, $\theta_s$ represents the student parameters. The teacher distribution is fixed, while the student distribution changes as its parameters are updated.

The practical uses response-based distillation because knowledge is transferred through the teacher's output probabilities rather than through intermediate feature maps or hidden representations.

## Q26. What should you write in the final assessment report?

Answer: Your report should include two plots, your measured results, and a short discussion of what you learned.

You can use the following template after running the notebook.

### Results

| Model | Best validation accuracy |
| --- | --- |
| Independent student | `[your result]` |
| Teacher | `[your result]` |
| Distilled student | `[your result]` |

Include:

- A training plot comparing the independent and distilled students.
- A validation accuracy comparison showing all three models.
- The temperature and loss-weighting parameter used in the final experiment.

### Discussion template

> In this practical, I investigated response-based knowledge distillation on CIFAR-10. I first trained an independent student model and a larger teacher model, then trained a new student using a weighted combination of cross-entropy loss and temperature-scaled KL divergence. The temperature controlled the smoothness of the teacher's output distribution, while the weighting parameter controlled the balance between the true labels and teacher supervision. The independent student achieved \[X%\] validation accuracy, the teacher achieved \[Y%\], and the distilled student achieved \[Z%\]. This corresponds to an improvement of \[Z − X\] percentage points over the independent student. One issue I encountered was \[describe your actual issue\], which I addressed by \[describe your actual solution\]. Overall, the experiment demonstrated \[summarize what your measured results show about distillation\].

Replace the placeholders with your actual experimental results. If distillation does not improve the baseline, report that honestly and discuss the hyperparameters and possible causes.

## Final revision: five questions you should be able to answer

1What is the purpose of the teacher model?  2Why do we divide logits by temperature?  3Why use KL divergence?  4Why multiply the soft loss by T²?  5What proves that distillation was successful?

The most important takeaway is that knowledge distillation does not make the student larger. It changes the learning signal so that a small model can benefit from information learned by a larger model. Your experiment tests whether that additional information improves generalization on unseen CIFAR-10 images.
