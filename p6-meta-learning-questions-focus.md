# Advanced Machine Learning — Practical 6: Few-Shot Learning and Meta-Learning

Let's explain Practical 6 using the same question-and-answer approach as Practical 5, as if you were preparing for a lab session, an oral exam, or a written assessment.

The main idea is:

Can we train a neural network to learn new image categories from only a few labelled examples?

In ordinary supervised learning, a model learns from many labelled images belonging to known classes. In few-shot learning, the goal is to recognise new classes using only a small support set of examples.

This practical uses PyTorch and CIFAR image data to compare three approaches:

1. Baseline supervised learning: Pre-train a feature extractor on many classes, then adapt it to new tasks.
2. Reptile: Meta-train the model's initial weights so that they can adapt quickly to new tasks.
3. Prototypical Networks (ProtoNet): Learn an embedding space in which examples can be classified by comparing them with the average embeddings of a few labelled examples.

The practical contains two main implementation tasks: completing the ProtoNet classifier and implementing its meta-training loop. It also asks you to evaluate the results against the baseline and Reptile.

Before examining the code, let's build an understanding of the concepts behind it.

# Part 1 — Understanding the assignment

## Q1. What is few-shot learning?

Answer: Few-shot learning is a machine learning approach that aims to recognise new categories using only a small number of labelled examples per category.

Imagine that a model has already learned general visual features, such as edges, textures, shapes, and object parts. Now we give it a new task involving five categories, with only five labelled examples per category.

Instead of training a large classifier from scratch using thousands of examples, the model uses those few examples to recognise the new categories.

For example, a five-way, five-shot classification task could look like this:

Machine Learning cơ bản

Class 1: Dog

Five labelled support examples teach the model what this class looks like.

Tutorial 2: 94% accuracy on Cifar10 in 2 minutes | by David Yang | Fenwicks | Medium

Class 2: Ship

The model builds a representation from a few examples rather than thousands.

Red-eyed Tree Frog — Colour Studies

Class 3: Frog

The same feature extractor can be reused for another category.

The actual practical uses randomly sampled classification tasks, with classes and support/query examples determined by its task-generation code.

The essential challenge is generalisation to unseen classes with very little labelled data.

## Q2. What is the difference between ordinary supervised learning and few-shot learning?

Answer: The main difference is how the model learns and how it is evaluated.

| Ordinary supervised learning | Few-shot learning |
| --- | --- |
| Trains on many examples from known classes. | Adapts to a task using only a few labelled examples per class. |
| Typically has a fixed classification head. | Often uses a task-specific classifier or class prototypes. |
| Optimises classification on the training classes. | Aims to generalise to new classes and new tasks. |
| Usually evaluates on held-out examples from the same class vocabulary. | Evaluates on tasks containing classes excluded from meta-training. |

In this practical, the baseline is trained across the meta-training classes, but its final few-shot evaluation uses separate meta-test tasks.

This separation matters: good performance on the original training classes does not automatically mean that a model will adapt well to new classes.

## Q3. What do meta-training, meta-validation, and meta-testing mean?

Answer: They are three stages for developing and assessing a model's ability to learn new tasks.

- Meta-training: The model learns from many sampled tasks constructed from the meta-training classes.
- Meta-validation: Performance on held-out tasks is used to monitor progress, compare models, or select hyperparameters.
- Meta-testing: The model is evaluated on separate tasks to measure its ability to generalise.

The distinction between a dataset and a task is important.

A dataset contains images and labels. A task is a small classification problem constructed by selecting some classes and dividing their images into support and query sets.

A typical few-shot task has this structure:

Sample classes and images

Choose a small set of categories for one episode.

Support set

A few labelled examples per class. Used to construct the classifier or prototypes.

Query set

Additional examples whose labels are used to calculate the task's loss and accuracy.

Evaluate how well the model solves the task

During meta-training, the model repeatedly encounters new combinations of classes. During meta-testing, it must handle tasks from the held-out classes.

Exam tip: Meta-learning means learning how to learn across tasks, rather than simply learning one fixed classification problem.

## Q4. What are the support set and query set?

Answer: They are two different groups of examples within each few-shot task.

- The support set provides the labelled examples used to define or adapt the task-specific classifier.
- The query set provides examples used to evaluate how well the model classifies that task.

For ProtoNet, the support set determines the class prototypes, while the query set is classified by comparing its embeddings against those prototypes.

For Reptile, the model is fine-tuned on the support set, and the query loss and accuracy are recorded for evaluation.

A simple example is a five-way, five-shot task:

- Five classes.
- Five support images per class: $5\times5=25$ support images.
- If there are 15 query images per class, $5\times15=75$ query images.

These counts illustrate the terminology; the actual counts are determined by `SUPPORT_SIZE`, `QUERY_SIZE`, and the notebook's task-generation implementation.

## Q5. What is an episode in meta-learning?

Answer: An episode is one sampled few-shot learning task.

In the notebook, an episode begins with code such as:

python
```
metatrain_task = create_task(
    num_support=SUPPORT_SIZE,
    num_query=QUERY_SIZE
)
```

The generated task provides the support and query loaders. The model uses these examples to perform one meta-training update.

The distinction is:

- Epoch: Typically one pass through a training dataset.
- Episode: One sampled task in the meta-learning process.

In this practical, the Reptile and ProtoNet loops are organised around episodes rather than conventional epochs over the full image dataset.

# Part 2 — The baseline supervised model

## Q6. Why does the assignment train a baseline before implementing meta-learning?

Answer: The baseline establishes a reference point against which the meta-learning methods can be compared.

The baseline model contains two components:

1. A backbone, which extracts features from images.
2. A classification head, which maps those features to class scores.

The notebook combines them as follows:

python
```
baseline_backbone = ConvBackbone()

baseline_classifier = ClassifierHead(
    num_classes=len(meta_train_classes),
    zero_init=True,
    normalise_features=False
)

baseline_model = nn.Sequential(
    baseline_backbone,
    baseline_classifier
)
```

The backbone learns visual representations. The classifier learns to predict the meta-training classes from those representations.

The model is trained with supervised learning on the meta-training dataset, using cross-entropy loss and stochastic gradient descent.

The training parameters include:

python
```
max_epochs = 30
lr = 1e-2
l2_reg = 1e-2
annealing = True
```

Here, `lr` is the initial learning rate, `l2_reg` controls weight decay, and `annealing=True` enables cosine learning-rate scheduling.

## Q7. Why do we need a backbone and a classification head?

Answer: They perform different jobs.

The backbone converts each image into a feature vector:

$$
x\longrightarrow f_\theta(x)
$$

The classification head converts that vector into class logits:

$$
f_\theta(x)\longrightarrow z
$$

For example, a convolutional backbone might map an image into a 64-dimensional representation. A classification head then maps that representation into one score per training class.

The advantage of separating these components is that we can reuse the backbone for a new task without necessarily reusing the original classifier.

This is especially useful for few-shot learning because the original classification head is trained for the old classes, whereas the new task may contain entirely different classes.

## Q8. What are cross-entropy loss, accuracy, and weight decay doing here?

Answer: They control how the baseline learns and how its performance is measured.

- Cross-entropy loss measures how poorly the predicted class probabilities match the correct labels.
- Accuracy measures the proportion of correctly classified examples.
- Weight decay discourages unnecessarily large parameter values and can help regularise training.
- Momentum in SGD helps updates build on consistent gradient directions.
- Cosine annealing gradually reduces the learning rate according to a cosine schedule.

The baseline uses `top5_acc` because it classifies many meta-training categories. Top-5 accuracy counts a prediction as correct when the true class appears among the five highest-scoring classes.

However, the few-shot meta-test tasks are five-way classification problems, so their evaluation uses top-1 accuracy.

Important: Baseline validation accuracy on the meta-training classes is not the final metric of interest. The key comparison is the average query accuracy across the held-out meta-test tasks.

## Q9. How does the baseline adapt to a new few-shot task?

Answer: The notebook uses `finetune_backbone_on_task()` to fine-tune a model on each task's support set and evaluate it on the query set.

Its main parameters include:

python
```
FT_EPOCHS = 5
FT_PATIENCE = 3
FT_LR = 1e-3
FT_L2 = 0
```

The function can freeze the backbone or allow its parameters to be updated. It can also create a fresh task-specific classification head.

For the baseline evaluation, the notebook uses:

python
```
full_state=False
freeze_backbone=True
```

This means the pre-trained backbone is copied into a new model for each task, then kept fixed while a new classifier head is trained.

The result tests whether the features learned by ordinary supervised learning are useful for classifying unfamiliar categories.

# Part 3 — Reptile: learning a useful initialisation

## Q10. What is Reptile?

Answer: Reptile is a gradient-based meta-learning algorithm that tries to find model initial parameters from which many different tasks can be learned quickly.

Instead of only training on random batches from a large collection of classes, Reptile repeatedly samples a task, fine-tunes the model on that task, and then moves the meta-model's parameters toward the fine-tuned parameters.

The intuition is that the initialisation should be useful across many tasks, not just good at one fixed classification problem.

## Q11. How does the Reptile update work mathematically?

Answer: Let:

- $\theta$ be the current meta-model parameters.
- $\theta_i'$ be the parameters after fine-tuning on task $i$.
- $\epsilon$ be the meta-step size.

The Reptile update is:

$$
\boxed{
\theta\leftarrow
\theta+\epsilon(\theta_i'-\theta)
}
$$

Suppose one parameter starts at $2$, task fine-tuning changes it to $6$, and the meta-step size is $0.5$.

Then:

$$
\theta_{\mathrm{new}}
=2+0.5(6-2)=4
$$

The meta-model moves halfway from its original parameter value toward the task-adapted value.

Explore the update

Initial parameter $\theta$

### 2.0

0

5

10

Task-adapted parameter $\theta_i'$

### 6.0

0

5

10

Meta-step size $\epsilon$

### 0.50

0

0.5

1

Updated parameter

# 4.00

The update moves a fraction $\epsilon$ of the distance toward the task-adapted parameter.

A small step size makes a small adjustment. A step size of one moves all the way to the task-adapted parameter.

## Q12. What does the Reptile training loop do?

Answer: It performs four main operations.

1. Sample a task from the meta-training classes.
2. Fine-tune the current backbone on the task's support set.
3. Move the meta-model's parameters toward the fine-tuned parameters.
4. Record the query-set performance and periodically evaluate on meta-test tasks.

The notebook implements the update using a saved state dictionary:

python
```
current_state = reptile_meta_model.state_dict()
prev_state = torch.load("prev_state.ckpt")
new_state = {}

for name, current_param in current_state.items():
    prev_param = prev_state[name]
    param_task_vector = current_param - prev_param

    new_param = (
        prev_param
        + param_task_vector * meta_step_size
    )

    new_state[name] = new_param

reptile_meta_model.load_state_dict(new_state)
```

The difference between the adapted and original parameters is the task vector. Multiplying it by `meta_step_size` controls how far the meta-model moves.

The state dictionary contains more than trainable weights; it may also contain buffers, such as BatchNorm running statistics. The notebook's simple update interpolates all entries, which is worth keeping in mind when debugging Reptile implementations.

## Q13. Why does the Reptile implementation create a new classifier head for each task?

Answer: Different tasks contain different classes, so their classifiers must assign outputs to different categories.

The call:

python
```
task_metrics, task_head = finetune_backbone_on_task(
    reptile_meta_model,
    metatrain_task,
    max_epochs=FT_EPOCHS,
    lr=FT_LR,
    l2_reg=FT_L2,
    reinit=True,
    progress_bar=False
)
```

creates a task-specific head while using the meta-backbone to extract features.

The task-specific head is trained alongside the backbone during inner-loop fine-tuning. The Reptile update is then applied to the backbone's parameters.

This is why the notebook meta-trains `ConvBackbone()` rather than the original full classification model.

## Q14. How do we know whether Reptile is working?

Answer: The notebook periodically evaluates the Reptile backbone on the meta-test tasks.

The important metric is:

$$
\text{Average query accuracy}
=\frac{1}{N}\sum_{i=1}^{N}
\operatorname{Accuracy}_i
$$

where $N$ is the number of meta-test tasks.

In the notebook, the query accuracy is compared with `baseline_metatest_query_acc`. The training plots also show the meta-training query performance and the meta-validation performance.

The code saves a checkpoint when the meta-test query accuracy improves. This helps preserve the best-performing model encountered during training.

The notebook's intended lesson is that Reptile is relatively straightforward to implement, but its performance depends on the task distribution, backbone, and inner-loop hyperparameters. The practical suggests that it may struggle to outperform the pre-trained baseline.

### An implementation issue to notice

At the end of the Reptile section, the notebook contains:

python
```
final_meta_results = meta_evaluate(meta_model, ...)
```

But the Reptile model in the preceding code is called `reptile_backbone`. Unless `meta_model` has been assigned elsewhere, this is a variable-name error. The final evaluation should use the intended Reptile model, for example:

python
```
final_meta_results = meta_evaluate(
    reptile_backbone,
    model_name="Final meta-trained Reptile model",
    baseline_avg=baseline_metatest_query_acc,
    full_state=False,
    freeze_backbone=True
)
```

This assumes the saved best checkpoint has already been loaded into `reptile_backbone`.

# Part 4 — Task 1: Implementing Prototypical Networks

## Q15. What is a Prototypical Network (ProtoNet)?

Answer: A Prototypical Network is a few-shot learning model that classifies examples by comparing their feature embeddings with the prototypes of the available classes.

Unlike an ordinary classification head, ProtoNet does not need a learned linear layer that has a fixed output neuron for every possible class.

Instead, it constructs a temporary representation for each class using the labelled support examples.

The procedure is:

1. Pass the support images through the backbone.
2. Calculate the mean embedding for each class.
3. Pass the query images through the same backbone.
4. Calculate the distance from every query embedding to every class prototype.
5. Use the negative distances as logits for classification.

The prototype is the average feature representation of a class.

## Q16. What is an embedding?

Answer: An embedding is a numerical representation of an input in a learned feature space.

Let $f_\theta(x)$ be the output of the backbone for image $x$. This output is the image's embedding.

Ideally, images from the same class have similar embeddings, while images from different classes have embeddings that are farther apart.

For example, suppose three support images of the same class produce the following two-dimensional embeddings:

$$
z_1=(1,2),\quad z_2=(2,3),\quad z_3=(3,1)
$$

Their prototype is the mean:

$$
c=\frac{z_1+z_2+z_3}{3}
$$

Therefore,

$$
c=(2,2)
$$

The prototype summarises the feature representation of that class. In the actual practical, embeddings will generally have more than two dimensions.

## Q17. How do we implement `make_prototypes()`?

Answer: First, embed the support images. Then group the embeddings by their class labels and calculate the mean for each group.

Complete the method as follows:

python
```
def make_prototypes(self, support_x, support_y):
    # Extract one embedding per support image.
    support_embeddings = self.backbone(support_x)

    # Find the class labels represented in the support set.
    classes = torch.unique(support_y, sorted=True)

    # Calculate the mean embedding for each class.
    prototypes = torch.stack([
        support_embeddings[support_y == cls].mean(dim=0)
        for cls in classes
    ])

    return prototypes
```

Let's examine the important lines:

- `self.backbone(support_x)` converts images into feature embeddings.
- `torch.unique()` identifies the classes represented in the support set.
- `support_y == cls` selects embeddings belonging to one class.
- `.mean(dim=0)` averages across the selected support examples.
- `torch.stack()` combines the class means into one prototype tensor.

If there are five classes and each embedding has 64 features, the prototype tensor has shape `(5, 64)`.

This implementation assumes the backbone returns one feature vector per image, with shape `(batch_size, embedding_dim)`.

## Q18. How do we calculate distances between query embeddings and prototypes?

Answer: We use Euclidean distance.

For a query embedding $z$ and class prototype $c_k$, the Euclidean distance is:

$$
d(z,c_k)=\sqrt{\sum_j(z_j-c_{k,j})^2}
$$

A smaller distance means the query embedding is closer to the prototype.

The model can calculate all query-to-prototype distances in one operation using `torch.cdist()`.

If there are 75 query images and five class prototypes, the distance matrix has shape `(75, 5)`.

Each row contains the distances from one query image to all five classes.

## Q19. Why do we use negative distances as logits?

Answer: Ordinary classifiers produce logits where larger values indicate stronger support for a class.

Distances have the opposite interpretation: smaller values indicate a better match.

Therefore, we negate the distances:

$$
\boxed{
z_k=-d(f_\theta(x),c_k)
}
$$

The closest prototype now has the largest logit.

We can then pass these logits directly to cross-entropy loss. PyTorch internally applies the appropriate log-softmax operation.

For example, suppose a query has distances:

$$
d=[1.0,\ 3.0,\ 2.0]
$$

The logits become:

$$
z=[-1.0,\ -3.0,\ -2.0]
$$

Class 1 has the largest logit because its prototype is closest to the query.

## Q20. What is temperature in ProtoNet?

Answer: Temperature controls how sharply the logits distinguish between the prototypes.

The notebook defines a temperature parameter $T$, so the classification logits are:

$$
\boxed{
z_k=-\frac{d(f_\theta(x),c_k)}{T}
}
$$

- When $T=1$, the negative Euclidean distances are used directly.
- When $T>1$, the logits become less extreme, generally producing softer class probabilities.
- When $T$ is very small and positive, the model places relatively more emphasis on the nearest prototype.

Temperature does not change which prototype is nearest, provided it is positive. It changes the scale of the logits and thus the confidence of the resulting probability distribution.

## Q21. How do we implement `predict_query()`?

Answer: Embed the query images, calculate their distances to the prototypes, and return temperature-scaled negative distances.

python
```
def predict_query(self, query_x, prototypes):
    # Extract embeddings for the query images.
    query_embeddings = self.backbone(query_x)

    # Calculate Euclidean distances to every prototype.
    distances = torch.cdist(
        query_embeddings,
        prototypes,
        p=2
    )

    # Convert distances to classification logits.
    logits = -distances / self.temperature

    return logits
```

The method returns logits, not probabilities. This is exactly what we need for `F.cross_entropy()`.

### Q22. What is the complete ProtoNet implementation?

Answer: Combine the two methods with the supplied constructor and `forward()` method.

python
```
class ProtoNet(nn.Module):
    def __init__(self, backbone, temperature=1.0):
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.backbone = backbone
        self.temperature = temperature

        self.to(device)

    def make_prototypes(self, support_x, support_y):
        support_embeddings = self.backbone(support_x)

        classes = torch.unique(support_y, sorted=True)

        prototypes = torch.stack([
            support_embeddings[support_y == cls].mean(dim=0)
            for cls in classes
        ])

        return prototypes

    def predict_query(self, query_x, prototypes):
        query_embeddings = self.backbone(query_x)

        distances = torch.cdist(
            query_embeddings,
            prototypes,
            p=2
        )

        logits = -distances / self.temperature

        return logits

    def forward(self, support_set, query_set):
        support_x, support_y = support_set
        query_x, query_y = query_set

        prototypes = self.make_prototypes(
            support_x, support_y
        )

        logits = self.predict_query(
            query_x, prototypes
        )

        return logits
```

The `query_y` variable is unpacked to match the supplied interface, but it is not used to calculate the prototypes or logits. The labels are needed later to calculate the query loss and accuracy.

Important assumption: The implementation expects the support labels to correspond to the same class ordering as the query labels used by cross-entropy. In this notebook, task labels should be local class indices such as `0, 1, 2, 3, 4`. If the labels are arbitrary global dataset IDs, they must first be remapped to consecutive task-local indices.

## Q23. How does the entire ProtoNet classification process work?

Answer: The following diagram summarises the forward pass.

Support images and labels

A few labelled examples per class

Shared Conv4 backbone

Convert images into embeddings

Calculate class prototypes

Mean embedding for each class

Query images

Examples to classify

Shared Conv4 backbone

Produce query embeddings

Negative distances to prototypes

Produce class logits for each query

Cross-entropy loss

Compare query logits with true query labels during training

Notice that the backbone is shared by the support and query sets. During training, the query loss sends gradients through the query embeddings and through the prototypes to the backbone.

That is how ProtoNet learns a feature space suitable for few-shot classification.

# Part 5 — Task 2: Training ProtoNet

## Q24. How is ProtoNet trained if it has no learnable classification head?

Answer: ProtoNet learns through its backbone.

Although the prototype classifier has no trainable linear layer, the backbone has trainable convolutional parameters. These parameters determine the embeddings, which determine the prototypes and query-to-prototype distances.

During meta-training, the model learns to produce embeddings that help query images match the correct support-set prototypes.

The objective is:

$$
\boxed{
L_{\text{query}}
=\operatorname{CE}(z_{\text{query}},y_{\text{query}})
}
$$

Here:

- $z_{\text{query}}$ contains the negative distances between query embeddings and class prototypes.
- $y_{\text{query}}$ contains the correct query labels.
- Cross-entropy measures how well the query images are classified.

The optimizer updates the backbone so that correct query classes receive higher logits relative to incorrect classes.

## Q25. What happens during one ProtoNet meta-training episode?

Answer: Every episode follows these six steps:

1. Sample a new few-shot task from the meta-training classes.
2. Load the support images and labels.
3. Load the query images and labels.
4. Construct prototypes and calculate query logits.
5. Calculate query cross-entropy loss and backpropagate.
6. Update the backbone's parameters.

This is the central implementation task in the notebook.

## Q26. How do we complete the ProtoNet training loop?

Answer: Replace the `... ### YOUR CODE HERE` placeholder with the following code.

python
```
# Reset gradients from the previous episode.
proto_opt.zero_grad()

# Forward pass: construct prototypes and classify queries.
query_logits = proto_model(
    (support_x, support_y),
    (query_x, query_y)
)

# Calculate loss using the true query labels.
train_query_loss = F.cross_entropy(
    query_logits,
    query_y
)

# Calculate query-set accuracy.
train_query_acc = top1_acc(
    query_logits,
    query_y
)

# Backpropagate through the query loss.
train_query_loss.backward()

# Update the backbone parameters.
proto_opt.step()
```

This is the complete essential solution to Task 2.

Let's examine why each operation is necessary.

| Operation | Purpose |
| --- | --- |
| `proto_opt.zero_grad()` | Clears gradients from the previous episode. |
| `proto_model(...)` | Builds support prototypes and predicts query logits. |
| `F.cross_entropy(...)` | Measures query classification error. |
| `train_query_loss.backward()` | Computes gradients for the backbone. |
| `proto_opt.step()` | Updates the trainable parameters. |

The notebook already creates the optimizer:

python
```
proto_opt = torch.optim.AdamW(
    proto_model.parameters(),
    lr=meta_lr,
    weight_decay=meta_l2
)
```

The default hyperparameters are:

python
```
max_episodes = 5000
meta_evaluate_interval = 500

meta_lr = 1e-3
meta_l2 = 1e-3
```

The optimizer is AdamW, which uses adaptive gradient updates and decoupled weight decay.

### Q27. Why do we calculate loss on the query set rather than the support set?

Answer: Because the goal is to learn representations that generalise beyond the examples used to construct the prototypes.

The support examples define the classes. The query examples test whether the model can correctly classify additional images from those classes.

If the model were trained only to perform well on the support examples, it could learn representations that fit those examples without necessarily generalising to new examples.

The query loss therefore provides the training signal for learning an embedding space that works for few-shot classification.

This is a key idea in episodic meta-learning: the model repeatedly practises learning from support examples and being tested on query examples.

## Q28. Why does `train()` matter during ProtoNet training?

Answer: This line:

python
```
proto_model.train()
```

puts the backbone into training mode.

That matters if the backbone contains layers whose behaviour changes between training and evaluation, such as dropout or BatchNorm.

The model must also remain differentiable during training so that gradients can flow through the query predictions, prototypes, and backbone.

During meta-evaluation, the notebook switches to:

python
```
model.eval()

with torch.no_grad():
    query_logits = model(
        (support_x, support_y),
        (query_x, query_y)
    )
```

This disables training-specific behaviour and avoids building a gradient graph when only measuring performance.

## Q29. What does the learning-rate scheduler do?

Answer: The scheduler changes the learning rate during meta-training.

When `annealing=False`, the learning rate remains at its initial value of $10^{-3}$.

When `annealing=True`, the notebook uses:

python
```
proto_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    proto_opt,
    max_episodes
)
```

The scheduler gradually changes the learning rate according to a cosine schedule.

A decreasing learning rate can help stabilise training as the model approaches a useful solution. However, scheduling is optional, and its effect should be assessed from the actual training and validation curves.

# Part 6 — Meta-evaluation and comparison

## Q30. How is ProtoNet evaluated on unseen tasks?

Answer: The function `meta_evaluate_protonet()` evaluates the model on the meta-test tasks without updating its weights.

For each task, it:

1. Obtains the support examples and their labels.
2. Obtains the query examples and their labels.
3. Constructs prototypes from the support set.
4. Predicts the query labels.
5. Calculates query cross-entropy loss and top-1 accuracy.
6. Averages the results across the meta-test tasks.

The important code is:

python
```
model.eval()

with torch.no_grad():
    query_logits = model(
        (support_x, support_y),
        (query_x, query_y)
    )

    query_losses.append(
        F.cross_entropy(
            query_logits, query_y
        ).item()
    )

    query_accs.append(
        top1_acc(query_logits, query_y)
    )
```

Unlike the Reptile evaluation, ProtoNet does not fine-tune its backbone on each meta-test task. It uses the support set directly to construct the prototypes.

That difference is fundamental:

- Reptile: Adapt the model's parameters using gradient-based fine-tuning.
- ProtoNet: Adapt the classifier by computing new prototypes from the support embeddings.

Both approaches use the support set, but they adapt in different ways.

## Q31. What does meta-test query accuracy tell us?

Answer: It tells us how successfully the model classifies new examples in tasks containing held-out classes.

The notebook calculates the mean query accuracy across all meta-test tasks:

$$
\overline{A}_{\text{query}}
=\frac{1}{N}\sum_{i=1}^{N} A_i
$$

where $A_i$ is the query accuracy for task $i$.

If there are 15 meta-test tasks, each task contributes one accuracy value to the average.

This is the key metric used to compare the baseline, Reptile, and ProtoNet.

A high baseline training accuracy does not guarantee a high meta-test query accuracy. The latter measures the ability to classify examples from new classes using the few-shot task setup.

## Q32. How does the notebook select the best ProtoNet model?

Answer: It periodically evaluates the model and saves a checkpoint whenever the recorded meta-test query accuracy improves.

The relevant code is:

python
```
if test_query_acc > current_best_acc:
    torch.save(
        proto_model.state_dict(),
        "best_protonet.ckpt"
    )

    current_best_acc = test_query_acc
```

After training, the notebook reloads the best checkpoint:

python
```
proto_model.load_state_dict(
    torch.load(
        "best_protonet.ckpt",
        map_location=device,
        weights_only=True
    )
)
```

Using `map_location=device` helps ensure that the checkpoint is loaded onto the intended device. `weights_only=True` is appropriate for loading a state dictionary with modern PyTorch versions.

Assessment consideration: The supplied notebook uses meta-test tasks during periodic model selection. In a rigorous experimental design, repeated tuning against the meta-test set introduces information leakage. Ideally, use a separate meta-validation task set for checkpoint selection and reserve the meta-test set for final reporting.

## Q33. What are the most important differences between the three methods?

Answer: They differ in how they use the training classes and how they adapt to new tasks.

| Property | Baseline | Reptile | ProtoNet |
| --- | --- | --- | --- |
| Training approach | Supervised learning | Episodic meta-learning | Episodic meta-learning |
| Main trainable component | Backbone and original head | Meta-backbone | Embedding backbone |
| Task-specific classifier | New head | New head during adaptation | Computed prototypes |
| Adaptation on new tasks | Fine-tune the head | Fine-tune model parameters | Compute class prototypes |
| Gradient updates at test time | Yes, for the head | Yes, during fine-tuning | No |
| Main evaluation metric | Average query accuracy | Average query accuracy | Average query accuracy |

The comparison is informative because it investigates whether learning from sampled tasks is more useful for few-shot classification than ordinary supervised pre-training.

# Part 7 — Common problems and how to solve them

## Q34. What if ProtoNet produces the wrong output shape?

Answer: Check the shapes of the embeddings, prototypes, and logits.

For a five-way task with an embedding dimension of 64, an example set of expected shapes is:

| Tensor | Shape |
| --- | --- |
| Support images | `(25, C, H, W)` |
| Support labels | `(25,)` |
| Support embeddings | `(25, 64)` |
| Prototypes | `(5, 64)` |
| Query embeddings | `(75, 64)` |
| Query logits | `(75, 5)` |
| Query labels | `(75,)` |

These shapes assume 25 support examples and 75 query examples. Your notebook's actual dimensions depend on its task configuration.

The key requirement is that each query receives one logit per class, and that the query logits and query labels can be passed directly to cross-entropy.

## Q35. What if the loss does not decrease?

Answer: Check the implementation and experiment settings systematically.

- Check label encoding. The query labels must match the class ordering of the prototypes.
- Check the backbone output. It should return one embedding vector per image.
- Check gradient updates. Confirm that `zero_grad()`, `backward()`, and `step()` are all executed.
- Check the learning rate. A rate that is too high can make training unstable, while one that is too low may slow learning.
- Check the prototype calculation. Every class should have at least one support example.
- Check the evaluation procedure. Meta-training loss and meta-test accuracy are different measurements and need not improve at exactly the same rate.

A useful diagnostic is to print the shapes and loss for the first few episodes before launching thousands of training iterations.

## Q36. What if ProtoNet performs worse than Reptile or the baseline?

Answer: That is possible and does not necessarily indicate a coding error.

ProtoNet depends on the quality of the embedding space. If the embeddings do not place examples of the same class near each other, the prototypes may not be useful.

Try the following experiments one at a time:

| Parameter or choice | Possible experiment | Purpose |
| --- | --- | --- |
| Backbone initialisation | Train from scratch versus initialise from the pre-trained backbone | Tests whether pre-trained features help |
| Meta-learning rate | Compare a smaller and larger rate around $10^{-3}$ | Tests optimisation stability |
| Temperature | Try values such as 0.5, 1, and 2 | Changes the scale of the classification logits |
| Number of episodes | Increase training duration if the model is still improving | Tests whether more task exposure helps |
| Weight decay | Compare a smaller value with the default $10^{-3}$ | Tests regularisation strength |

Do not assume that the largest number of episodes or the lowest training loss will necessarily produce the best meta-test accuracy.

Also, the notebook's comment that the pre-trained backbone can optionally be loaded is significant. A pre-trained backbone may provide better initial features, but the result must be measured rather than assumed.

## Q37. What implementation issues should you check in the supplied notebook?

Answer: Several details are worth checking before interpreting the results.

- Backbone variable: The earlier example creates `ProtoNet(backbone=meta_backbone)`. Make sure `meta_backbone` has actually been defined and that it is the intended feature extractor. Otherwise, instantiate ProtoNet with an available `ConvBackbone()` or a properly loaded checkpoint.
- Reptile final evaluation: Check that the variable passed to `meta_evaluate()` is the intended Reptile model, as discussed in Q14.
- Checkpoint consistency: Save and load the same model architecture and state dictionary.
- Task-local labels: Ensure prototype order matches the labels used in cross-entropy.
- Model mode: Use `train()` during optimisation and `eval()` during evaluation.
- Meta-test usage: Ideally, reserve meta-test tasks for the final assessment rather than repeatedly using them to choose checkpoints.

These checks are particularly useful when the notebook runs without obvious Python errors but produces unexpectedly poor accuracy.

# Part 8 — Reporting your results

## Q38. Which plots should you include in the assessment?

Answer: Include plots that show both how the models train and how well they generalise to unseen tasks.

At minimum, aim to report:

1. Baseline plot: Training and validation loss and accuracy during supervised pre-training.
2. Reptile plot: Meta-training query performance and meta-test evaluation performance across episodes.
3. ProtoNet plot: Meta-training query performance and meta-test evaluation performance across episodes.
4. Final comparison: A clear comparison of the best baseline, Reptile, and ProtoNet meta-test query accuracies.

The supplied notebook already includes plotting functions for much of this. Save the resulting figures after running the experiment.

## Q39. How should you compare the final results?

Answer: Use the measured average query accuracy for each method.

Here is a table to fill in with the actual results from your notebook.

| Method | Average meta-test query accuracy |
| --- | --- |
| Baseline | `[your result]%` |
| Reptile | `[your result]%` |
| ProtoNet | `[your result]%` |

For example, suppose your measured results were 55% for the baseline, 57% for Reptile, and 64% for ProtoNet. These are illustrative values, not results from your experiment.

ProtoNet's improvement over the baseline would be:

$$
64\%-55\%=9
$$

That is an improvement of 9 percentage points.

Its relative improvement would be:

$$
\frac{64-55}{55}\times100
\approx16.4\%
$$

Distinguish percentage points from relative percentage improvement when writing your report.

## Q40. What should you write in the discussion section?

Answer: Summarise the methods, compare the measured results, explain any difficulties, and state what you learned.

You can adapt the following template after running the practical.

Writing

## Discussion — Practical 6: Few-Shot Learning and Meta-Learning

In this practical, I investigated few-shot image classification using a supervised baseline, Reptile, and Prototypical Networks. The baseline learned image features through ordinary supervised training, whereas Reptile learned an initialisation intended to support rapid adaptation to new tasks. ProtoNet classified query images by comparing their embeddings with class prototypes calculated from the support set.

For ProtoNet, I implemented prototype construction by averaging support embeddings within each class. I then used Euclidean distances between query embeddings and prototypes to produce classification logits. During meta-training, I optimised the backbone using cross-entropy loss on the query set.

The baseline achieved **\[X%\]**, Reptile achieved **\[Y%\]**, and ProtoNet achieved **\[Z%\]** average meta-test query accuracy. These results suggest that **\[explain which approach performed best and whether it improved on the baseline\]**.

One challenge I encountered was **\[describe an actual issue\]**. I addressed it by **\[describe the solution you implemented\]**. Overall, the practical helped me understand how episodic training, support/query separation, and embedding-based classification can be used to learn new categories from limited labelled data.

Replace the placeholders with your actual observations. If ProtoNet or Reptile does not outperform the baseline, report that honestly and discuss possible explanations.

# Part 9 — Advanced understanding and exam preparation

## Q41. What is the fundamental difference between Reptile and ProtoNet?

Answer: Reptile learns a useful parameter initialisation, while ProtoNet learns a useful embedding space for prototype-based classification.

Reptile uses task-specific gradient updates to adapt the backbone. Its meta-update moves the initial parameters toward parameters that perform well after adaptation.

ProtoNet instead uses the query classification loss to train its embedding backbone. At evaluation time, it computes class prototypes from the support set and classifies queries by distance, without gradient-based fine-tuning.

The essential distinction is:

$$
\text{Reptile: learn an adaptable initialisation}
$$

$$
\text{ProtoNet: learn an embedding space for comparison}
$$

Both are meta-learning approaches, but they use different mechanisms to transfer knowledge across tasks.

## Q42. Why can ProtoNet work without a learned linear classification head?

Answer: The class prototypes act as task-specific reference points in the embedding space.

A conventional linear classifier learns a weight vector for each class. ProtoNet instead computes one representative embedding for each class from the current support set.

Because these prototypes are recalculated for each task, the classifier automatically adapts to the classes that appear in that task.

This makes ProtoNet especially convenient when the evaluation classes were not present during training.

## Q43. Why is the query loss useful for learning the backbone?

Answer: The query loss measures whether the learned embedding space supports correct classification of additional examples, not just the support examples.

When a query is incorrectly classified, the cross-entropy gradient propagates through its embedding and through the prototypes that were computed from the support embeddings.

Consequently, the backbone receives a learning signal that can improve the relationships between support and query representations.

Repeated across different episodes, this encourages the model to learn representations useful for many classification tasks.

## Q44. What are the five most important things to remember for an oral exam?

1. Few-shot learning: Classify new categories using only a few labelled examples.
2. Support and query sets: Support examples define or adapt the task-specific classifier; query examples measure generalisation.
3. Reptile: Fine-tune on a task, then move the meta-initialisation toward the adapted parameters.
4. ProtoNet: Average support embeddings into prototypes and classify queries using negative distances.
5. Meta-test query accuracy: The key evaluation metric for comparing how well the models generalise to unseen tasks.

## Final takeaway

The most important lesson in Practical 6 is that learning to classify a fixed collection of classes is different from learning to classify new classes from a handful of examples.

The baseline investigates whether ordinary supervised pre-training produces useful features. Reptile investigates whether the initial parameters can be optimised for rapid task adaptation. ProtoNet investigates whether a learned embedding space allows new classes to be recognised through their support-set prototypes.

Your final results should establish which approach works best under the practical's task setup, rather than assuming that any one method must win.
