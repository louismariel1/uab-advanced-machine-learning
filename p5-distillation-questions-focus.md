# Practical 5 — Knowledge Distillation: Assignment Explained

## 1. What is the main objective of this assignment?

Question: What are you trying to achieve in this notebook?

The goal is to improve the classification accuracy of a small neural network by teaching it using the predictions of a larger, more powerful neural network.

This technique is called knowledge distillation. You will train three models on CIFAR-10, a dataset of 32 × 32 colour images belonging to 10 classes, such as cats, dogs, cars, birds, and aeroplanes.

The notebook uses only 30% of the original CIFAR-10 dataset for training, 20% for validation, and discards the remaining 50%.

Khan Academy

Model 1: Independent student

A small ResNet trained normally using the correct image labels. This is the baseline against which you will measure improvement.

\[cs231n-lec9\] CNN Architectures - young log

Model 2: Teacher

A larger ResNet trained on the same data. It should learn better representations and achieve higher validation accuracy.

"Knowledge Distillation: How to Make Tiny AI Models as Smart as Giant Ones" - DEV Community

Model 3: Distilled student

A new small ResNet trained using both the true labels and the teacher's predictions. The objective is to outperform the independent student while approaching the teacher's accuracy.

The overall comparison is:

$$
\text{Independent student}
\quad \rightarrow \quad
\text{Distilled student}
\quad \rightarrow \quad
\text{Teacher}
$$

This represents the expected performance ordering, not a guaranteed result.

## 2. What is the difference between the student and teacher models?

Question: How are the two architectures different?

Both models use the custom `BasicResnet` class defined in chunk 1, which is built from residual convolutional blocks.

The notebook defines the following architectures:

| Property | Student | Teacher |
| --- | --- | --- |
| Convolutional block sizes | `[32, 64, 128]` | `[64, 128, 128, 256, 256]` |
| Number of residual blocks | 3 | 5 |
| Dropout | None | 0.2 |
| Classification output | 10 classes | 10 classes |
| Role | Compact model | More powerful model |

The teacher has more layers and wider convolutional blocks, so it has substantially more parameters than the student.

Question: Why train the small model first?

You need a baseline to establish how well the student performs without assistance. Otherwise, you cannot determine whether knowledge distillation actually improves its performance.

Question: Why train the teacher separately?

The teacher must first learn to make useful predictions. Its outputs will then provide additional training information for the student.

The teacher is trained first and its weights are saved in `teacher.ckpt`. These weights are reloaded before distillation.

## 3. How is the dataset prepared?

Question: What data preprocessing and splitting take place?

The notebook performs three main operations.

1. Split the data: `random_split` creates training, validation, and discarded subsets in the proportions 30%, 20%, and 50%.
2. Convert images to tensors: `transforms.ToTensor()` converts each image to a PyTorch tensor.
3. Normalize the images: each RGB channel is normalized using mean $0.5$ and standard deviation $0.2$.

The `collate` function processes a batch of images and labels, stacks them into tensors, and sends them to the selected device.

The notebook uses a batch size of 256 and selects a CUDA GPU when available, otherwise falling back to the CPU.

Question: Why use a reduced dataset?

A smaller dataset makes training faster and helps demonstrate the effect of distillation in a practical exercise.

## 4. What happens during baseline training?

Question: How is the independent student trained?

The function `train_model()` implements ordinary supervised training.

For every batch of training images, the notebook:

1. Passes the images through the student model.
2. Obtains the model's raw output logits.
3. Computes the cross-entropy loss against the correct labels.
4. Calculates the batch accuracy.
5. Backpropagates the loss and updates the model parameters.
6. Records training metrics.

At the end of every epoch, the model is evaluated on the validation set.

The main hyperparameters are:

| Hyperparameter | Value |
| --- | --- |
| Epochs | 20 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| LR decay factor | 0.95 |
| Batch size | 256 |

Question: What is the purpose of the baseline?

The independent student's best validation accuracy, stored in `ind_student_metrics.best_val_acc`, is the benchmark that the distilled student must beat.

The teacher is trained separately for 20 epochs, using the same learning rate and learning-rate decay, but with weight decay of `0.1`.

The notebook expects the teacher to outperform the baseline student. However, actual results depend on training and execution.

## 5. What is knowledge distillation?

Question: Why aren't the true labels enough?

Consider an image of a horse. Its true label tells the model only that the image belongs to the `horse` class.

However, a well-trained teacher might predict something like this:

| Class | Illustrative probability |
| --- | --- |
| Horse | 0.75 |
| Deer | 0.10 |
| Dog | 0.07 |
| Cat | 0.04 |
| Truck | 0.01 |
| Other classes | 0.03 |

These are illustrative values, not measured outputs from the notebook.

The teacher's predictions contain information about how similar it considers the different classes to be. For example, an image of a horse might share visual features with other animals.

These additional signals are called soft labels.

The original labels are called hard labels because they identify one correct class, normally represented as a one-hot vector.

Knowledge distillation trains the student using both sources of information:

- Hard labels: what the correct answer is.
- Soft labels: what the teacher has learned about the relationships between classes.

## 6. Task 1 — How do you implement temperature-scaled softmax?

Question: What is temperature, and why is it needed?

The teacher's predictions may be extremely confident, giving almost all probability to one class. This makes the other class probabilities very small, reducing the additional information available to the student.

Temperature $T$ softens the probability distribution.

The formula is:

$$
p_i = \frac{\exp(z_i/T)}
{\sum_j \exp(z_j/T)}
$$

where:

- $z_i$ is the logit for class $i$.
- $T$ is the temperature.
- $p_i$ is the resulting probability for class $i$.

When $T=1$, this is ordinary softmax. Increasing $T$ makes the distribution softer, while decreasing $T$ makes it more concentrated.

Question: What code should you write?

Replace the incomplete `temperature_softmax()` function with:

python
```

def temperature_softmax(x, T):
    """Applies softmax after scaling logits by temperature."""
    return F.softmax(x / T, dim=1)
```

Question: How do you check whether it works?

Run the existing plotting code with `temp = 20`.

Compare the teacher's original softmax distribution with its temperature-scaled version. You should see a less concentrated distribution with more probability assigned to non-maximal classes.

The output should still sum to approximately 1 for each example.

## 7. Task 2 — How do you implement the distillation loss?

Question: What is the loss function measuring?

The distillation loss measures the difference between the student and teacher probability distributions after temperature scaling.

The notebook uses Kullback–Leibler divergence, or KL divergence.

$$
D_{\mathrm{KL}}(P\parallel Q)
=\sum_i P(i)\log\frac{P(i)}{Q(i)}
$$

For this assignment, the required loss is:

$$
L_d =
D_{\mathrm{KL}}(p_s\parallel p_t)T^2
$$

where $p_s$ is the student's softened distribution and $p_t$ is the teacher's softened distribution.

The $T^2$ factor compensates for the scaling of gradients caused by temperature.

Question: How should you implement it in PyTorch?

PyTorch's `F.kl_div()` expects the first argument to be log probabilities and the second to be target probabilities when using its standard KL-divergence interface.

Use the following implementation:

python
```

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
        reduction='batchmean'
    )

    return loss * (temperature ** 2)
```

Why use `reduction='batchmean'`?

It sums the divergence across classes and examples, then divides by the batch size. This ensures that the loss is not scaled directly by the number of examples in the batch.

Question: Why use log-softmax for the student?

The KL-divergence function expects log probabilities in its first argument. `F.log_softmax()` computes these in a numerically stable way.

Question: How do you verify Task 2?

Run the provided sanity checks. They verify that:

- The loss is a scalar tensor, with shape `()`.
- The loss is positive for the example being tested.
- The loss differs from the ordinary cross-entropy loss.

Remember that KL divergence can be zero when the two distributions match exactly. It is not mathematically guaranteed to be positive for every possible pair of predictions.

## 8. Task 3 — How do you train the student using distillation?

Question: What changes compared with ordinary training?

In ordinary training, the student uses only cross-entropy loss:

$$
L_h = \operatorname{CrossEntropy}(z_s,y)
$$

In distillation training, the student combines this hard-label loss with the distillation loss:

$$
\boxed{L = \alpha L_h+(1-\alpha)L_d}
$$

Here:

- $L_h$ is the hard-label cross-entropy loss.
- $L_d$ is the temperature-scaled KL-divergence loss.
- $\alpha$ controls the relative importance of the two losses.

With this convention, $\alpha=0.5$ gives equal weight to the two terms. The assignment allows you to experiment with the weighting.

Question: What should the training loop do?

The `distil_model()` function should follow the same overall structure as `train_model()`, with a few additional steps.

1. Load a batch of images and labels

Teacher

Evaluation mode; no gradients

Student

Training mode; gradients enabled

2. Calculate both losses

Cross-entropy against the labels + KL divergence against the teacher

3. Combine the losses and update the student

Backpropagate the combined loss; the teacher's weights stay fixed

4. Validate the student at the end of each epoch

Question: Which implementation details are important?

- Call `student.train()` at the beginning of each training epoch.
- Call `teacher.eval()` before generating teacher predictions.
- Use `torch.no_grad()` when computing teacher predictions.
- Pass only `student.parameters()` to the optimizer.
- Calculate `hard_loss` using `nn.CrossEntropyLoss()`.
- Calculate `soft_loss` using your Task 2 function.
- Combine the losses using the selected $\alpha$.
- Call `backward()` and `optimizer.step()` on the combined loss.
- Use `evaluate_model()` to evaluate the student on the validation set.
- Record the metrics with `metrics.log_train()` and `metrics.log_val()`.

These details matter because the teacher must remain fixed, while the student learns from both the correct labels and the teacher's predictions.

### Suggested implementation of `distil_model()`

The following is a complete training-loop implementation consistent with the notebook's helper functions and logging conventions.

python
```

def distil_model(student: nn.Module,
                 teacher: nn.Module,
                 temperature: float,
                 alpha: float,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 num_epochs: int,
                 lr: float,
                 l2_reg: float = 0.0,
                 gamma: float = 1.0,
                 baseline_accs: dict = None,
                 baseline_metrics: TrainingMetrics = None,
                 dynamic_plot: bool = True,
                 run_name: str = 'distilled student',
                 seed: int = 0):

    torch.manual_seed(seed)

    hard_loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        student.parameters(), lr=lr, weight_decay=l2_reg
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        opt, gamma=gamma
    )

    metrics = TrainingMetrics()
    teacher.eval()

    for e in range(num_epochs):
        student.train()
        teacher.eval()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()

            # Teacher predictions are fixed targets
            with torch.no_grad():
                teacher_pred = teacher(x)

            student_pred = student(x)

            hard_loss = hard_loss_func(student_pred, y)
            soft_loss = distillation_output_loss(
                student_pred, teacher_pred, temperature
            )

            loss = alpha * hard_loss + (1 - alpha) * soft_loss
            batch_acc = get_batch_acc(student_pred, y)

            loss.backward()
            opt.step()

            metrics.log_train(
                loss=float(hard_loss.item()),
                acc=float(batch_acc),
                soft_loss=float(soft_loss.item())
            )

        scheduler.step()

        val_loss, val_acc = evaluate_model(
            student, val_loader, hard_loss_func
        )
        metrics.log_val(
            loss=float(val_loss), acc=float(val_acc)
        )

        if dynamic_plot:
            metrics.plot(
                title=f'Training ({run_name})',
                live=True,
                max_step=len(train_loader) * num_epochs,
                baseline_accs=baseline_accs,
                baseline_metrics=baseline_metrics
            )
        else:
            train_loss, val_loss_epoch = metrics.epoch_loss
            train_acc, val_acc_epoch = metrics.epoch_acc
            print(
                f'Epoch {e+1:>2} | '
                f'loss: {train_loss:.3f} | '
                f'val loss: {val_loss_epoch:.3f} | '
                f'acc: {train_acc:.1%} | '
                f'val acc: {val_acc_epoch:.1%}'
            )

    if not dynamic_plot:
        metrics.plot(
            title=f'Training ({run_name})',
            live=False,
            baseline_accs=baseline_accs,
            baseline_metrics=baseline_metrics
        )

    print(
        f'Best validation accuracy ({run_name}): '
        f'{metrics.best_val_acc:.2%}'
    )
    return metrics
```

One subtlety: the supplied `TrainingMetrics.plot()` function labels its first training-loss curve as the supervised loss whenever a soft-loss series is present. This implementation logs the hard loss and soft loss separately, as suggested by the notebook. The combined loss is what actually drives the optimizer updates.

## 9. Which temperature and loss weight should you choose?

Question: What values should you use initially?

A sensible first experiment is:

- Temperature: $T=20$, matching the example used in Tasks 1 and 2.
- Loss weight: $\alpha=0.5$.
- Epochs: 20.
- Learning rate: $10^{-3}$.
- Weight decay: $10^{-4}$.
- Learning-rate decay: $0.95$.

These values provide a starting point, not a guarantee of success. The notebook specifically encourages experimentation with temperature and loss weighting.

Question: How do you launch distillation training?

After implementing `distil_model()`, replace the unfinished cells with:

python
```

temperature = 20
alpha = 0.5

student = BasicResnet(student_conv_layers).to(device)

print(f'Loading pre-trained weights from {teacher_save_name}')
teacher_state_dict = torch.load(
    teacher_save_name, map_location=device
)
teacher.load_state_dict(teacher_state_dict)
teacher.eval()

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
        'ind. student': ind_student_metrics.best_val_acc,
        'teacher': teacher_metrics.best_val_acc
    },
    baseline_metrics=ind_student_metrics,
    dynamic_plot=True,
    run_name='distilled student'
)
```

This reuses the baseline student's hyperparameters, which makes the comparison fairer.

Note that `torch.load()` behavior can depend on the PyTorch version and how the checkpoint was saved. Here the notebook saves a plain state dictionary, which is what `load_state_dict()` expects.

## 10. How do you evaluate whether distillation worked?

Question: What should you compare?

Compare the best validation accuracy of all three models:

- `ind_student_metrics.best_val_acc`
- `student_metrics.best_val_acc`
- `teacher_metrics.best_val_acc`

The notebook already provides code to plot the training curves and compare the final student's performance against the two reference models.

The assignment's stated target is to gain at least 5 percentage points over the independent student. For example, if the independent student achieves 67%, the target would be 72%, not 70.35%.

Question: How should you interpret the results?

| Result | Interpretation |
| --- | --- |
| Distilled student is more accurate than the independent student | Distillation improved performance |
| Distilled student approaches teacher accuracy | Much of the performance gap has been closed |
| Distilled student matches or exceeds the teacher | Excellent result, though not guaranteed |
| Distilled student is worse than the independent student | The distillation setup may need tuning |

The fraction of the student–teacher performance gap recovered can be calculated as:

$$
\text{Gap closed} =
\frac{A_d-A_s}{A_t-A_s}
$$

where $A_d$ is the distilled student's accuracy, $A_s$ is the independent student's accuracy, and $A_t$ is the teacher's accuracy.

This calculation is most meaningful when the teacher outperforms the baseline. It can be negative if distillation hurts performance and can exceed 100% if the distilled student surpasses the teacher.

## 11. What should you write in the assessment?

Question: What does the final submission require?

The notebook requests two things.

### A. One or two plots

Include plots comparing validation accuracy for:

- The independent student baseline.
- The distilled student.
- The teacher.

You can use the existing `student_metrics.plot()` call and its horizontal accuracy reference lines. Ensure that your plot or accompanying text identifies the best validation accuracy achieved by each model.

### B. A short written discussion

Your discussion should explain:

1. Whether the distilled student outperformed the independent student.
2. How close the distilled student came to the teacher.
3. Any implementation issues and how you resolved them.
4. What you learned about temperature, soft labels, and the effect of the distillation-loss weight.

Here is a fill-in template you can adapt after running your notebook.

Writing

## Practical 5: Knowledge Distillation — Results and Discussion

### Results

The independent student achieved a best validation accuracy of **\[X%\]**, while the teacher achieved **\[Y%\]**. After knowledge distillation, the student achieved **\[Z%\]** validation accuracy, corresponding to an improvement of **\[Z − X percentage points\]** over the independent student.

The training curves show **\[describe whether the distilled student's validation accuracy improved and how it compared with the teacher\]**.

### Discussion

I implemented temperature-scaled softmax to produce softer probability distributions and used KL divergence, multiplied by the squared temperature, as the distillation loss. The student was trained using a combination of the cross-entropy loss against the true labels and the distillation loss against the teacher's predictions.

One issue I encountered was **\[describe an actual issue, or state that no major implementation issues occurred\]**. I addressed this by **\[explain the solution\]**.

This practical demonstrated that a teacher's soft predictions can provide additional information beyond hard labels. The effect of distillation depends on the temperature, the relative weighting of the two losses, and the quality of the teacher. **\[Summarise what your results showed about these factors.\]**

Only fill in the results after running the notebook; no model accuracies can be determined from the code alone.

## 12. What are the three tasks you must complete?

The assignment boils down to three coding tasks and one reporting task.

[ ]Task 1: Temperature-scaled softmaxImplement \`temperature\_softmax(x, T)\` using \`F.softmax(x / T, dim=1)\`. Check that the distributions sum to 1.

[ ]Task 2: Distillation lossImplement temperature-scaled student and teacher distributions, KL divergence with \`reduction='batchmean'\`, and the \`T\*\*2\` factor.

[ ]Task 3: Distillation trainingImplement \`distil\_model()\`, keep the teacher fixed, combine hard and soft losses, and train a fresh student.

[ ]Assessment: Results and discussionPlot and compare validation accuracies, report the measured results, and discuss the implementation and insights.

0 of 4 completed

The key idea to remember is that the teacher is not teaching the student by providing new labels manually; it is teaching through the structure of its predicted probability distribution. Your job is to implement that signal correctly, combine it with the original supervised objective, and demonstrate whether it improves the small model's validation performance.
