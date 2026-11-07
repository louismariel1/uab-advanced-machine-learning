# Advanced Machine Learning – Practical 6: Meta-Learning

## Part 5 – Understanding the Reptile algorithm and interpreting the results

We have now completed the first meta-learning algorithm in the practical: Reptile.

This is an important stage because we can compare three things:

1. The conventional supervised learning baseline.
2. The Reptile meta-learning model.
3. Later, the Prototypical Network (ProtoNet).

Let's understand what the code does, why it works, and, most importantly, what your results tell us.

## Question 19. What is Reptile?

Reptile is a gradient-based meta-learning algorithm.

Its objective is to find a neural network initialisation that can adapt quickly to new classification tasks using only a few labelled examples.

Remember the problem we are trying to solve:

- We have 80 classes available for meta-training.
- We generate many small classification tasks from those classes.
- Each task has only six support examples per class.
- We want the model to learn a good starting point that can be adapted to completely new classes.

The central idea of Reptile is:

Train on a task, then move the original model parameters towards the parameters obtained after task-specific training.

This process is repeated thousands of times.

### Understanding the intuition

Imagine that we start with a randomly initialised neural network.

We then give it a task:

- Class 0: Butterflies
- Class 1: Mice
- Class 2: Couches

We fine-tune the network using the support examples.

After fine-tuning, the network has adapted to this particular task.

We then compare:

- The original network parameters.
- The parameters after fine-tuning.

Reptile moves the original network towards the adapted parameters.

We repeat this with thousands of randomly generated tasks.

Eventually, we hope the original network becomes a useful initialisation for many different classification problems.

### Reptile versus ordinary supervised learning

| Ordinary supervised learning | Reptile |
| --- | --- |
| Learns to classify fixed classes | Learns a useful initialisation across tasks |
| Uses one large classification problem | Uses many small classification tasks |
| Optimises performance on training classes | Optimises the ability to adapt to new tasks |
| Normally keeps the trained classifier | Reuses the backbone with a new task-specific classifier |

## Question 20. What is the Reptile update equation?

This is the most important mathematical concept in the Reptile implementation.

The algorithm uses:

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
+
\epsilon
(\theta_{\text{task}}-\theta_{\text{old}})
$$

Where:

- $\theta_{\text{old}}$: parameters before task-specific training.
- $\theta_{\text{task}}$: parameters after fine-tuning on the current task.
- $\epsilon$: meta-learning step size.
- $\theta_{\text{new}}$: updated meta-model parameters.

The notebook defines:

python

```python
meta_step_size = 0.5
```

Therefore:

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
+
0.5(\theta_{\text{task}}-\theta_{\text{old}})
$$

This means the meta-model moves halfway towards the task-adapted model.

### Numerical example

Imagine one neural network parameter initially has the value:

$$
\theta_{\text{old}}=2
$$

After fine-tuning on a task:

$$
\theta_{\text{task}}=6
$$

The Reptile update becomes:

$$
\theta_{\text{new}}
=
2+0.5(6-2)
$$

$$
\theta_{\text{new}}=4
$$

The new meta-model parameter is 4.

It has moved halfway from 2 towards 6.

This is repeated for all parameters in the neural network.

Important: Reptile does not simply keep the model trained on the latest task. It updates the original meta-model towards the task-adapted model.

## Question 21. Understanding the Reptile training hyperparameters

Let's examine the first part of your code.

python

```python
max_episodes = 5000

meta_evaluate_interval = 200

meta_step_size = 0.5
```

These are the three main hyperparameters.

### 1. max\_episodes = 5000

This specifies the number of meta-training episodes.

An episode represents one randomly generated classification task.

For each episode:

1. Generate a random task.
2. Fine-tune the backbone on its support set.
3. Calculate the Reptile update.
4. Update the meta-model.
5. Record the task's query performance.

The algorithm repeats this process 5,000 times.

### 2. meta\_evaluate\_interval = 200

The model is evaluated every 200 episodes.

This means:

- Episode 0: evaluation.
- Episode 200: evaluation.
- Episode 400: evaluation.
- Episode 600: evaluation.
- And so on.

This allows us to monitor whether the model is becoming better at adapting to unseen tasks.

### 3. meta\_step\_size = 0.5

This controls how far the meta-model moves towards the task-adapted parameters.

A larger value means a larger outer-loop update.

A smaller value means a more conservative update.

The meta-step size is different from the inner learning rate.

| Hyperparameter | Purpose | Value |
| --- | --- | --- |
| FT\_LR | Learning rate during task fine-tuning | 0.01 |
| meta\_step\_size | Reptile outer-loop update | 0.5 |
| FT\_EPOCHS | Fine-tuning epochs per task | 10 |
| max\_episodes | Number of meta-training tasks | 5,000 |

This distinction is important:

- The inner learning rate controls how quickly the model adapts to one task.
- The meta-step size controls how the general initialisation changes across tasks.

## Question 22. How does the notebook initialise the Reptile model?

python

```python
reptile_backbone = ConvBackbone()

reptile_meta_metrics = TrainingMetrics(meta=True)

current_best_acc = 0

verbose = False
```

Let's examine each instruction.

### ConvBackbone()

python

```python
reptile_backbone = ConvBackbone()
```

Creates a new CNN backbone.

Initially, its weights are randomly initialised.

This is the model that Reptile will train across multiple tasks.

### TrainingMetrics(meta=True)

python

```python
reptile_meta_metrics = TrainingMetrics(meta=True)
```

Creates an object for recording meta-training and meta-evaluation metrics.

It stores information such as:

- Meta-training query loss.
- Meta-training query accuracy.
- Meta-test query loss.
- Meta-test query accuracy.
- Best validation accuracy.

Notice that the meta-training accuracy is measured using the query set, not the support set.

This is useful because it measures how well the model performs on examples that were not used for its inner-loop updates.

### current\_best\_acc

python

```python
current_best_acc = 0
```

Stores the best meta-test query accuracy observed so far.

It is used to decide whether a checkpoint should be saved.

## Question 23. Why does the notebook save prev\_state.ckpt?

python

```python
torch.save(
    reptile_backbone.state_dict(),
    'prev_state.ckpt'
)
```

This saves the initial backbone parameters.

Why?

Because Reptile needs to compare the model before and after fine-tuning.

We therefore need access to the original parameter values.

The saved state represents:

$$
\theta_{\text{old}}
$$

After fine-tuning, the current backbone represents:

$$
\theta_{\text{task}}
$$

We can then calculate the difference between them.

This difference determines the Reptile update.

## Question 24. Understanding the Reptile training loop

The main loop is:

python

```python
for m in tqdm(range(max_episodes)):
```

It executes 5,000 times.

Let's break down what happens during each iteration.

### Step 1. Generate a random meta-training task

python

```python
metatrain_task = create_task(
    num_support=SUPPORT_SIZE,
    num_query=QUERY_SIZE
)
```

This generates a random task using CIFAR-80.

For example:

- Butterflies
- Mice
- Couches

The function randomly chooses the number of classes (3, 4 or 5).

It then creates:

- A support set containing six examples per class.
- A query set containing six examples per class.

The support set is used for fine-tuning.

The query set is used to evaluate the task-adapted model.

### Step 2. Fine-tune the backbone

python

```python
task_metrics, task_head = finetune_backbone_on_task(
    reptile_backbone,
    metatrain_task,
    max_epochs=FT_EPOCHS,
    lr=FT_LR,
    l2_reg=FT_L2,
    progress_bar=False
)
```

This is the inner training loop.

The function:

1. Creates a new classifier head.
2. Connects it to the backbone.
3. Trains the backbone and classifier using the support set.
4. Evaluates the adapted model on the query set.
5. Returns the training metrics and classifier.

The important point is that the backbone is modified in place.

After this function finishes, the backbone contains the task-adapted parameters.

This is exactly what Reptile needs.

### Step 3. Count the examples seen

python

```python
num_examples_seen += (
    len(metatrain_task.support) * task_metrics.step
)
```

This calculates the approximate number of support examples processed during training.

For example, suppose:

- The task contains three classes.
- Each class has six support examples.
- Training runs for ten epochs.

The support set contains:

$$
3 \times 6 = 18
$$

If each epoch processes the complete support set once:

$$
18 \times 10 = 180
$$

Therefore, this task contributes approximately 180 examples seen.

The notebook uses this counter to report training progress.

### Step 4. Calculate the Reptile update

This is the most important section.

python

```python
current_state = reptile_backbone.state_dict()

prev_state = torch.load('prev_state.ckpt')

new_state = dict()
```

We retrieve:

- The current backbone parameters.
- The previously saved backbone parameters.

Then:

python

```python
for name, current_param in current_state.items():

    prev_param = prev_state[name]

    param_task_vector = current_param - prev_param

    new_param = (
        prev_param +
        (param_task_vector * meta_step_size)
    )

    new_state[name] = new_param
```

Let's understand this mathematically.

First:

python

```python
param_task_vector = current_param - prev_param
```

Calculates the difference between the adapted and original parameters.

Mathematically:

$$
\Delta\theta
=
\theta_{\text{task}}-\theta_{\text{old}}
$$

Then:

python

```python
new_param = (
    prev_param +
    (param_task_vector * meta_step_size)
)
```

Moves the original parameter towards the adapted parameter.

Mathematically:

$$
\theta_{\text{new}}
=
\theta_{\text{old}}+
\epsilon\Delta\theta
$$

Since:

$$
\epsilon=0.5
$$

the meta-model moves halfway towards the adapted model.

### Step 5. Update the meta-model

python

```python
reptile_backbone.load_state_dict(new_state)
```

This loads the updated parameters into the backbone.

The backbone is now the updated meta-model.

### Step 6. Save the updated parameters

python

```python
torch.save(
    reptile_backbone.state_dict(),
    'prev_state.ckpt'
)
```

This is important because the next episode needs to compare its task-adapted parameters against the newly updated meta-model.

The process then repeats.

## Question 25. Why does Reptile use query accuracy during meta-training?

Consider:

python

```python
reptile_meta_metrics.log_train(
    loss=task_metrics.val_loss[-1],
    acc=task_metrics.val_acc[-1]
)
```

Notice something interesting.

The notebook records:

python

```python
task_metrics.val_loss[-1]
task_metrics.val_acc[-1]
```

It does not record the final training loss and accuracy.

Why?

Because:

- Training metrics measure performance on the support set.
- Validation metrics measure performance on the query set.

The query examples were not used for the inner-loop gradient updates.

Therefore, query accuracy provides a better indication of whether the model can generalise beyond the support examples.

This is central to few-shot learning.

A model that achieves 100% support accuracy but only 40% query accuracy may simply be memorising the small support set.

## Question 26. How does meta\_evaluate() measure Reptile performance?

Every 200 episodes, the notebook calls:

python

```python
meta_eval_results = meta_evaluate(
    reptile_backbone,
    model_name=f'Reptile meta-model (episode {m})',
    max_epochs=FT_EPOCHS,
    lr=FT_LR,
    l2_reg=FT_L2,
    early_stopping=False,
    inspect_tasks=False,
    progress_bars=False,
    show_taskwise_accuracy=False,
    baseline_avg=baseline_metatest_query_acc,
    freeze_backbone=True
)
```

This evaluates the current Reptile model on the 15 meta-test tasks.

Remember that these tasks contain classes that were excluded from meta-training.

### Why is freeze\_backbone=True important?

This is a particularly important detail.

When we evaluate Reptile, the notebook uses:

python

```python
freeze_backbone=True
```

This means that the backbone weights are not updated during meta-test fine-tuning.

Only the new task-specific classifier head is trained.

Consequently, this evaluation tests how useful the learned feature representations are without further adapting the backbone.

There is an important distinction here:

- During Reptile meta-training, the backbone is fine-tuned on each training task before the outer-loop update.
- During this meta-evaluation, the backbone is frozen and only the classifier is trained.

This is a specific evaluation configuration, not a requirement of the Reptile algorithm itself.

### Why do we use the same 15 tasks?

Because we want a fair comparison against the baseline.

The baseline and Reptile model are evaluated using the same meta-test tasks.

This makes the comparison more meaningful.

## Question 27. What is the purpose of saving best\_reptile.ckpt?

python

```python
if (test_query_acc > current_best_acc):

    print(
        f'Meta-test query accuracy {test_query_acc:.2%} '
        f'better than previous best {current_best_acc:.2%}, '
        'saving checkpoint'
    )

    torch.save(
        reptile_backbone.state_dict(),
        'best_reptile.ckpt'
    )
```

The model is evaluated repeatedly during meta-training.

Its performance may improve or deteriorate between evaluations.

Instead of automatically keeping the final model, the notebook saves the model that achieves the highest meta-test query accuracy.

This is called checkpointing.

The purpose is to preserve the best-performing version of the model.

A methodological note: The notebook uses the meta-test set repeatedly to select the best checkpoint. Strict experimental evaluation would normally use a separate meta-validation set for model selection and reserve meta-test tasks for the final evaluation. Repeatedly selecting checkpoints using test performance can introduce test-set selection bias.

For understanding this practical, however, the important point is that the code retains the best observed Reptile model.

# Part 6 – Understanding your Reptile results

You successfully executed the final Reptile evaluation.

Let's analyse the actual output.

## Question 28. What do the final Reptile results tell us?

Your output:

text

```text
Final meta-trained Reptile model evaluation on meta-test set:

AVERAGE META-TEST SUPPORT ACCURACY: 96.74%

AVERAGE META-TEST QUERY ACCURACY: 64.41%

(compared to baseline average: 69.93%)
```

Let's interpret these numbers.

Final performance comparison

BASELINE QUERY ACCURACY

# 69.93%

REPTILE QUERY ACCURACY

# 64.41%

Baseline support accuracy

97.72%

Reptile support accuracy

96.74%

QUERY ACCURACY DIFFERENCE

# -5.52 percentage points

Reptile performs worse than the conventional baseline in this experiment.

### 1. Baseline query accuracy: 69.93%

The baseline achieves an average accuracy of 69.93% on unseen few-shot classification tasks.

This is our reference performance.

### 2. Reptile query accuracy: 64.41%

Reptile achieves 64.41%.

This is lower than the baseline.

The difference is:

$$
64.41-69.93=-5.52
$$

Therefore, Reptile performs 5.52 percentage points worse than the baseline.

### 3. What does the support accuracy tell us?

Reptile achieves:

$$
96.74\%
$$

support accuracy.

This is very high.

However, its query accuracy is only:

$$
64.41\%
$$

The difference is:

$$
96.74-64.41=32.33
$$

percentage points.

This suggests that the model performs much better on the examples used for adaptation than on previously unseen examples from the same task.

This is consistent with overfitting to the small support sets.

It is not proof of overfitting by itself, but it is a strong indication that generalisation remains a problem.

## Question 29. Why did Reptile perform worse than the baseline?

This is an important analytical question.

Meta-learning is not automatically better than conventional supervised learning.

Several factors could explain your results.

### 1. The backbone may overfit to individual tasks

During each Reptile episode, the backbone is fine-tuned for ten epochs using a very small support set.

The model may learn the particular support examples very well without learning sufficiently transferable features.

Your support accuracy of 96.74% and query accuracy of 64.41% suggest this possibility.

### 2. The outer-loop step size may be too large

Your code uses:

python

```python
meta_step_size = 0.5
```

This means the model moves halfway towards the task-adapted parameters after every episode.

A relatively large outer-loop update can make training unstable or cause the meta-model to overemphasise individual tasks.

However, we cannot conclude that 0.5 is definitely too large without comparing other values.

### 3. The inner-loop training may be too extensive

Your inner-loop configuration is:

python

```python
FT_EPOCHS = 10
FT_LR = 1e-2
FT_L2 = 0
```

The backbone is trained for ten epochs on very few images.

This can result in substantial task-specific adaptation.

Reducing the number of inner-loop epochs or changing the inner learning rate could potentially improve generalisation.

### 4. The baseline has an advantage from conventional pre-training

The baseline first learns to classify 80 CIFAR classes using a much larger collection of labelled images.

This produces a useful feature extractor.

Reptile starts from a randomly initialised backbone and learns through many small tasks.

These are different training strategies, and the baseline can perform surprisingly well on unseen classes because of its transferable visual representations.

### 5. The evaluation protocol freezes the Reptile backbone

Remember:

python

```python
freeze_backbone=True
```

This means that Reptile's backbone cannot adapt during meta-test evaluation.

It only provides fixed features to the new classifier.

This is a meaningful test of the quality of the learned representation, but it does not measure the full adaptation capability of Reptile when the backbone itself is allowed to change.

Consequently, your results should be interpreted in the context of this particular evaluation protocol.

## Question 30. Why does the notebook load the best checkpoint at the end?

Your final cell contains:

python

```python
print('Loading best Reptile checkpoint...')

reptile_backbone.load_state_dict(
    torch.load('best_reptile.ckpt')
)
```

During the 5,000 episodes, the notebook repeatedly evaluates the model.

It saves the best-performing checkpoint whenever meta-test query accuracy improves.

At the end, it reloads that checkpoint.

This ensures that the final evaluation uses the best saved model rather than automatically using the model from episode 4,999.

The final evaluation then executes:

python

```python
final_meta_results = meta_evaluate(
    reptile_backbone,
    baseline_avg=baseline_metatest_query_acc,
    model_name='Final meta-trained Reptile model',
    freeze_backbone=True
)
```

This evaluates the selected Reptile model one final time.

Your final output confirms:

- Average support accuracy: 96.74%.
- Average query accuracy: 64.41%.
- Baseline query accuracy: 69.93%.

## Question 31. How should you interpret the individual task results?

Your output also reports the accuracy for each of the 15 meta-test tasks.

| Task | Reptile query accuracy |
| --- | --- |
| 0 | 62.5% |
| 1 | 70.0% |
| 2 | 44.4% |
| 3 | 66.7% |
| 4 | 66.7% |
| 5 | 63.3% |
| 6 | 77.8% |
| 7 | 72.2% |
| 8 | 61.1% |
| 9 | 61.1% |
| 10 | 43.3% |
| 11 | 62.5% |
| 12 | 70.0% |
| 13 | 61.1% |
| 14 | 83.3% |

Notice that the performance varies considerably.

For example:

- Task 14 achieves 83.3%.
- Task 6 achieves 77.8%.
- Task 10 achieves only 43.3%.
- Task 2 achieves only 44.4%.

This indicates that the model does not generalise equally well to all unseen classification tasks.

Some combinations of classes are easier to distinguish than others.

For example, classes with visually distinctive shapes and colours may be easier than classes with similar visual characteristics.

This is why averaging performance over multiple tasks is important.

A single task would not provide a reliable picture of the model's overall generalisation ability.

# Part 7 – Baseline versus Reptile: final comparison

Let's put your two experiments together.

| Metric | Baseline | Reptile |
| --- | --- | --- |
| Support accuracy | 97.72% | 96.74% |
| Query accuracy | 69.93% | 64.41% |
| Evaluation tasks | 15 | 15 |
| Backbone | Supervised pre-training | Reptile meta-training |
| Meta-test backbone | Frozen | Frozen |

Baseline vs Reptile

Average query accuracy on the 15 unseen meta-test tasks.

Expand chart

|  | Query accuracy |
| --- | --- |
| Baseline | 69.9% |
| Reptile | 64.4% |

### What can we conclude?

The conventional supervised baseline performs better than Reptile in your experiment.

This does not mean Reptile is inherently ineffective.

It means that, with the current architecture, hyperparameters, training procedure and evaluation protocol, Reptile did not outperform the baseline.

This is an important lesson in machine learning:

A more sophisticated algorithm does not necessarily produce better results.

The experimental configuration matters just as much as the choice of algorithm.

# Part 8 – What comes next: Prototypical Networks (ProtoNet)

We have now covered:

1. The conventional supervised baseline.
2. Gradient-based meta-learning using Reptile.

The final algorithm in this practical is ProtoNet.

It uses a fundamentally different approach.

Instead of adapting a classifier through repeated gradient updates, ProtoNet learns a feature space in which examples belonging to the same class are close together.

### How does ProtoNet work?

Imagine that we have a task containing three classes:

- Butterflies
- Mice
- Couches

The backbone converts every image into a feature vector.

ProtoNet then calculates a prototype for each class.

A prototype is simply the average feature vector of the support examples belonging to that class.

Mathematically:

$$
c_k =
\frac{1}{|S_k|}
\sum_{(x_i,y_i)\in S_k}
f_\phi(x_i)
$$

Where:

- $c_k$ is the prototype of class $k$.
- $S_k$ is the support set for class $k$.
- $f_\phi(x_i)$ is the feature representation of image $x_i$.

When a query image arrives, ProtoNet compares its feature vector against every class prototype.

The query image is assigned to the nearest prototype.

Building a Prototypical Network — Few-shot and Zero-shot Learning for Music Information Retrieval

[music-fsl-zsl.github.io](<https://music-fsl-zsl.github.io/tutorial/fsl-example/models.html?utm_source=chatgpt.com>)

### The fundamental difference

| Baseline | Reptile | ProtoNet |
| --- | --- | --- |
| Supervised feature learning | Gradient-based meta-learning | Metric-based meta-learning |
| Fine-tunes a classifier | Learns a useful initialisation | Learns an embedding space |
| Uses cross-entropy classification | Uses task adaptation and outer updates | Uses distances to prototypes |
| Task-specific classifier | Task-specific classifier | No separately trained classifier head required for each task |

ProtoNet is particularly interesting because it can classify new classes without requiring gradient-based fine-tuning of a new classifier.

This makes it computationally attractive for few-shot classification.

## Summary – What you should remember from Reptile

1. Reptile is a gradient-based meta-learning algorithm.
2. It learns a useful initialisation across multiple classification tasks.
3. Each episode generates a random task from the meta-training classes.
4. The model is fine-tuned using the support set.
5. The original model parameters are moved towards the task-adapted parameters.
6. The Reptile update is controlled by `meta_step_size`.
7. The model is periodically evaluated on unseen meta-test tasks.
8. The best-performing checkpoint is saved.
9. Your baseline achieved 69.93% query accuracy.
10. Your Reptile model achieved 64.41% query accuracy.
11. Reptile therefore underperformed the baseline by 5.52 percentage points in this experiment.
12. The high support accuracy and lower query accuracy suggest a generalisation problem.

The next important objective is to understand whether ProtoNet can achieve better generalisation by learning a suitable feature space rather than relying on gradient-based adaptation.
