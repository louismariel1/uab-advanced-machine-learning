# Practical 3 – Pretext Tasks and Self-Supervised Learning

## Questions-and-Answers Focused Narrative Script

This narrative follows the same teaching-oriented structure as a practical walkthrough for LoRA and efficient learning: it explains the purpose of each stage, discusses the important concepts, anticipates questions that may arise during the practical, and provides answers that connect the implementation to the underlying machine-learning theory.

The script is designed to be used alongside the Practical 3 notebook, either as a teaching presentation or as a spoken explanation during the practical.

# 1. Introduction: What is Self-Supervised Learning?

### Q1. What is the main objective of Practical 3?

The objective of this practical is to investigate how we can train a neural network to learn useful visual representations without requiring labelled training data.

In previous practicals, we trained models using labelled datasets, where every training image was associated with a known class. For example, an image of a dog would have the label dog, and an image of a car would have the label car.

However, obtaining labelled datasets can be expensive and time-consuming. Human annotators must manually assign labels, and large-scale annotation is not always practical.

Self-supervised learning offers an alternative.

Instead of relying on human-provided labels, we create a learning task directly from the data itself. The model learns by solving an artificially constructed task, called a pretext task.

The representations learned through this process can subsequently be transferred to another task where labelled data is available.

Our ultimate objective is therefore:

- Learn useful feature representations from a large unlabelled dataset.
- Transfer these representations to a smaller labelled dataset.
- Improve classification performance compared with training a model from scratch.

### Q2. What is self-supervised learning?

Self-supervised learning is a machine-learning approach in which the training signal is generated automatically from the input data rather than provided through manually annotated labels.

For example, imagine that we have an image of a cat.

We could:

- Rotate the image and ask the model to predict its rotation.
- Remove part of the image and ask the model to reconstruct the missing information.
- Generate two different crops of the same image and ask the model whether they originate from the same image.

In all these cases, the original image provides the information needed to construct the training target.

The model therefore learns without needing an external human annotation.

Important distinction: Self-supervised learning is not simply learning without labels. It involves designing a task that generates its own supervisory signal from the available data.

### Q3. How does self-supervised learning differ from supervised and unsupervised learning?

There are three important learning paradigms to distinguish.

| Learning approach | Training information | Example |
| --- | --- | --- |
| Supervised learning | Human-provided labels | Classify images as cats or dogs |
| Unsupervised learning | No explicit labels or generated targets | Discover clusters in image data |
| Self-supervised learning | Automatically generated training targets | Predict image rotation |

Self-supervised learning is often considered part of the broader unsupervised learning family because it does not require manually labelled data.

However, unlike traditional unsupervised methods, it explicitly constructs a prediction task with a learning objective.

### Q4. What is a pretext task?

A pretext task is an artificially constructed task that encourages a model to learn useful representations from unlabelled data.

The task itself is not necessarily the final objective.

For example, predicting whether an image has been rotated by 0, 90, 180 or 270 degrees is not necessarily useful as an application in its own right.

However, to solve this task, the model may need to recognise objects, shapes, structures and spatial relationships.

These learned features could then help with a different task, such as classifying animals or vehicles.

This leads to an important distinction:

- Pretext task: The artificial task used to learn representations.
- Downstream task: The actual task we ultimately want the model to perform.

The quality of the pretext task is important because it determines what kind of representations the model learns.

# 2. Understanding the Learning Setup

### Q5. What datasets do we use in this practical?

The notebook provides two possible dataset configurations:

1. CIFAR-100 subset as the labelled target dataset and CIFAR-10 as the unlabelled source dataset.
2. STL-10, which provides separate labelled and unlabelled portions.

The default configuration in the notebook is:

python

```python
base_data = 'cifar'
```

This means that we use CIFAR-100 and CIFAR-10.

The datasets have different roles:

Source dataset – CIFAR-10

- Contains 50,000 training images.
- Originally has 10 object categories.
- All labels are deliberately removed.
- Used for self-supervised pre-training.

Target dataset – CIFAR-100 subset

- Contains a selected subset of CIFAR-100.
- Retains its original labels, remapped to five classes.
- Used for supervised training and evaluation.
- Represents our actual classification objective.

The important point is that the source dataset and target dataset do not have to contain the same classes.

They should, however, share useful visual characteristics.

For example, both datasets contain objects with shapes, textures, edges and visual structures that a convolutional neural network can learn to recognise.

### Q6. Why do we use two different datasets?

The purpose is to investigate whether representations learned from one dataset can improve performance on another dataset.

This is a transfer learning problem.

However, unlike traditional transfer learning, the source dataset is completely unlabelled.

We therefore cannot directly train a conventional classifier on the source dataset.

Instead, we create a pretext task to learn useful representations.

The workflow is:

Unlabelled source dataset

CIFAR-10 images without labels

Self-supervised pre-training

Pre-trained feature backbone

Learns general visual representations

Transfer and fine-tuning

Labelled target dataset

CIFAR-100 subset

Final downstream classifier

Evaluated using validation accuracy

### Q7. Why are the datasets different in terms of their classes?

The objective is not to learn the original class categories from the source dataset.

Instead, we want the network to learn more general visual features.

For example, a network trained on unlabelled images might learn:

- Edges and boundaries.
- Textures and patterns.
- Shapes and object structures.
- Spatial relationships.
- More complex visual representations.

These features may be useful even when the actual object categories differ.

This is why self-supervised learning can be effective for transfer learning across datasets with different classes.

### Q8. Why do we remove the CIFAR-10 labels?

The purpose is to simulate a realistic self-supervised learning scenario.

In the notebook, this is implemented as:

python

```python
cifarx.targets = [-1] * len(cifarx.data)
```

The labels are replaced with `-1`, effectively making them unavailable for learning.

The model can still access the images, but it cannot use the original class labels.

This ensures that the pre-training process genuinely relies on self-supervised learning.

# 3. Dataset Preparation and Preprocessing

### Q9. Why do we create custom dataset classes?

The notebook introduces two custom PyTorch dataset classes:

- `UnlabelledDataset`
- `LabelledDataset`

These provide a consistent interface for handling both types of data.

The unlabelled dataset stores only images:

python

```python
class UnlabelledDataset(Dataset):
```

The labelled dataset stores image-label pairs:

python

```python
class LabelledDataset(Dataset):
```

This distinction is important because the training procedures are different.

For supervised training, the model needs an input image and its corresponding target label.

For self-supervised training, the model needs an input image and a target generated through the pretext task.

### Q10. Why do we split the target dataset into training and validation sets?

We use:

python

```python
val_frac = 0.1
```

This reserves 10% of the labelled target dataset for validation.

The remaining 90% is used for training.

The training set is used to update the model parameters, while the validation set is used to evaluate how well the model generalises to unseen examples.

This is particularly important because the target dataset is relatively small.

Without validation, we might mistakenly believe that a model performs well simply because it memorises its training examples.

### Q11. What is data augmentation, and why is it important?

Data augmentation creates modified versions of existing images through transformations.

The notebook includes several examples:

python

```python
transforms.RandomResizedCrop(...)
transforms.RandomHorizontalFlip(...)
transforms.ColorJitter(...)
transforms.RandomErasing(...)
```

These transformations introduce variation into the training data.

For example:

- Random cropping changes which parts of an image are visible.
- Horizontal flipping changes the orientation of an image.
- Colour jitter changes brightness, contrast and colour.
- Random erasing removes parts of an image.

The objective is to prevent the model from relying too heavily on specific visual details.

Instead, it should learn more robust representations that generalise across different image conditions.

### Q12. Why is data augmentation particularly important in self-supervised learning?

Self-supervised learning often depends on the model learning relationships between different versions of the same image.

If the model always receives identical images, it may learn shortcuts that do not transfer well.

Augmentation introduces diversity and encourages the model to identify more meaningful visual information.

However, there is an important consideration:

The augmentations must not contradict the pretext task.

For example, if the pretext task is predicting image rotation, randomly rotating the image again after applying the rotation label could confuse the model.

The model would receive an image whose final orientation no longer corresponds to its assigned target.

Therefore, the pretext task and the augmentation strategy must be designed together.

# 4. Model Architecture: Backbone and Classification Head

### Q13. Why do we separate the backbone from the classification head?

This is one of the most important architectural decisions in the practical.

The model is divided into two components:

1. The backbone, which learns and extracts visual features.
2. The classification head, which uses those features to make predictions for a particular task.

The backbone is defined as:

python

```python
class ConvBackbone(nn.Module):
```

The classification head is defined as:

python

```python
class ClassifierHead(nn.Module):
```

This modular design makes transfer learning easier.

During self-supervised pre-training, the backbone learns representations that are useful for the pretext task.

When we move to the downstream classification task, we retain the backbone but replace the pretext-specific head with a new classification head.

### Q14. What does the ConvBackbone actually do?

The backbone is a convolutional neural network.

Its architecture includes:

- Four convolutional layers.
- ReLU activation functions.
- Max pooling.
- Adaptive average pooling.
- Fully connected layers.
- Dropout.
- A final feature projection layer.

The convolutional layers learn increasingly complex visual patterns.

The early layers typically learn relatively simple features such as edges and textures.

Deeper layers can learn more complex patterns and object-level structures.

The final output is a feature vector.

In this notebook:

python

```python
backbone_output_size = 64
```

Therefore, each image is represented by a 64-dimensional feature vector.

This feature representation is what we ultimately want to transfer to the downstream task.

### Q15. What is the role of the classification head?

The classification head converts the extracted feature vector into predictions for a particular task.

Its architecture contains:

python

```python
self.projection = nn.Linear(input_size, projection_size)
self.classifier = nn.Linear(projection_size, num_classes)
```

The projection layer transforms the feature representation into a smaller intermediate representation.

The final classifier maps this representation to the required number of classes.

For example, if the selected CIFAR-100 subset contains five classes, the final classifier produces five output values.

These values are called logits.

The predicted class is obtained by selecting the class with the highest logit.

### Q16. Why do we use a separate head for each task?

Because the backbone and the head serve different purposes.

The backbone learns general-purpose representations.

The head translates those representations into predictions for a particular objective.

During pre-training, the head predicts the artificial pretext targets.

During fine-tuning, the head predicts the actual target classes.

The pretext task and downstream task are different, so their classification heads must also be different.

This separation is fundamental to the transfer learning process.

# 5. Baseline Training: Establishing a Reference Performance

### Q17. Why do we train a baseline model before self-supervised learning?

Before investigating whether self-supervised learning improves performance, we need a reference point.

The baseline model is trained using only the labelled target dataset.

It has:

- A randomly initialised backbone.
- A classification head for the target classes.
- No self-supervised pre-training.

This model establishes how well we can perform using the limited labelled data alone.

The training process is implemented using:

python

```python
train_supervised(...)
```

The model is trained using the following hyperparameters:

python

```python
ft_num_epochs = 30
ft_lr = 1e-3
ft_l2_reg = 1e-1
```

These parameters control the training process.

### Q18. What is the purpose of the learning rate?

The learning rate determines the size of the parameter updates during optimisation.

A high learning rate can cause training to become unstable because the model may overshoot good parameter values.

A very low learning rate can make training unnecessarily slow.

Here, the learning rate is:

python

```python
ft_lr = 1e-3
```

This controls how quickly the model updates its weights when learning from the target dataset.

### Q19. What is L2 regularisation?

L2 regularisation penalises large model weights.

In this practical, it is implemented through the `weight_decay` parameter of the AdamW optimiser.

python

```python
opt = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=l2_reg
)
```

The objective is to discourage overly complex solutions and help reduce overfitting.

This is especially relevant because the target dataset is relatively small.

### Q20. Why might the baseline model overfit?

Overfitting occurs when a model learns the training examples too specifically and fails to generalise to unseen examples.

In this practical, we have a relatively small labelled dataset.

The neural network has enough parameters to memorise patterns in the training data, particularly if training continues for too long.

We may therefore observe:

- Training accuracy increasing.
- Training loss decreasing.
- Validation accuracy improving initially but then stagnating or decreasing.
- Validation loss increasing.

This gives us an important motivation for self-supervised learning.

If we can learn useful representations from a much larger unlabelled dataset, we may improve downstream performance using the limited labelled data.

### Q21. What do the training and validation curves tell us?

The notebook plots loss and accuracy during training.

There are two important metrics:

Training loss and accuracy

These measure how well the model performs on the examples used for updating its parameters.

Validation loss and accuracy

These measure how well the model performs on data it has not used for parameter updates.

The most important metric for this practical is validation accuracy.

We use:

python

```python
baseline_acc = baseline_metrics.best_val_acc
```

This stores the best validation accuracy achieved by the supervised baseline.

It becomes our reference for evaluating the self-supervised approach.

# 6. Task 1: Designing the Pretext Task

### Q22. What is the most important part of this practical?

The most important part is designing a suitable pretext task.

The quality of the pretext task determines whether the model learns useful representations or simply learns to solve an artificial problem that has little relevance to the downstream task.

We want the model to learn visual features that can transfer to other classification problems.

The notebook provides several possible approaches.

### Q23. What types of pretext tasks can we implement?

There are several possibilities.

1. Rotation prediction

Paper--Unsupervised Representation Learning by Predicting Image Rotations | Mia's Blog

Stanford CS231N | Spring 2025 | Lecture 12- Self-Supervised Learning | Tuan-Anh Bui

图像分类 - 斟知

We rotate an image by one of four possible angles:

- 0 degrees.
- 90 degrees.
- 180 degrees.
- 270 degrees.

The model must predict the applied rotation.

This encourages it to learn information about object structure and spatial orientation.

2. Colour transformation prediction

Efficient Image Segmentation Using PyTorch: Part 1 | by Dhruv Matani | TDS Archive | Medium

Boosting Deep Learning Performance with Data Augmentation Techniques | by VinLab | Medium

Self-Supervised Learning: Definition, Tutorial & Examples

We apply different colour transformations and ask the model to predict their intensity.

This is a regression task.

The model must learn relationships between the original image appearance and its transformed version.

3. Relative position prediction

Apprentissage autosupervisé et tâches de prétexte · Apprentissage Profond

We extract two patches from an image.

The model must predict their relative spatial positions.

This encourages the model to learn spatial relationships and object structure.

4. Jigsaw puzzle prediction

Self-Supervised Learning

Unsupervised visual representation learning overview: Toward Self-Supervision | PDF

Self-Supervised Learning - Pretext Tasks · Deep Learning

We divide an image into patches, shuffle them, and ask the model to predict their original arrangement.

This encourages the learning of spatial and contextual relationships.

5. Contrastive learning

SimCLR: A Simple Framework for Contrastive Learning of Visual Representations: A Brief Summary | by Kavishka Abeywardana | Medium

SimCLR Visually Explained - Vasudev Sharma

Self-Supervised Learning - Part 1: Simple and intuitive introduction for beginners - Willogy Insights

We create two different augmented versions of an image.

The model learns to recognise that both images originate from the same original image, while distinguishing them from unrelated images.

This encourages similar images to have similar feature representations.

### Q24. Which pretext task should we choose?

There is no universally best pretext task.

The appropriate choice depends on:

- The dataset.
- The available computational resources.
- The downstream task.
- The complexity of implementation.
- The type of representations we want the model to learn.

For this practical, rotation prediction is a particularly accessible starting point.

It is relatively straightforward to implement using a classification head with four output classes.

It also provides a clear way to understand how a model can learn visual representations without class labels.

However, the notebook deliberately leaves the choice open so that students can experiment with different approaches.

### Q25. Why should the pretext task not be too easy?

If the task is too easy, the model may achieve very high accuracy without learning meaningful visual representations.

For example, imagine that we create a task where the model must detect an extremely obvious visual transformation.

It might learn to identify superficial pixel-level differences rather than understand the content of the image.

The result could be excellent pretext accuracy but poor downstream performance.

This is an important distinction:

High pretext accuracy does not necessarily mean good feature learning.

We want a task that is sufficiently challenging to encourage useful representations, but not so difficult that the model cannot learn it.

### Q26. Why are denoising and deblurring not always ideal pretext tasks?

Denoising and deblurring are possible self-supervised tasks.

However, they can often be solved using relatively low-level visual information.

For example, a network might learn to reconstruct local pixel patterns without understanding the objects represented in the image.

If our downstream task is object classification, these low-level features may not be sufficient.

This illustrates why pretext task design matters.

We want to encourage learning of semantic information, not merely low-level image processing.

### Q27. What is contrastive learning?

Contrastive learning is a self-supervised learning approach in which the model learns to distinguish between related and unrelated examples.

The central idea is to create:

- Positive pairs: two different views of the same image.
- Negative pairs: views originating from different images.

The model learns to bring positive pairs closer together in the representation space while separating negative pairs.

A common example is SimCLR.

The notebook also suggests simpler alternatives, such as pairwise contrastive loss and triplet loss.

These approaches can provide useful representations without requiring the full complexity of SimCLR.

# 7. Implementing the Pretext Task

### Q28. How do we implement a pretext task in PyTorch?

The notebook provides a general implementation strategy.

There are five main steps:

1. Define the transformation that creates the pretext task.
2. Generate the corresponding training targets.
3. Create a dataset and data loader that return the transformed images and targets.
4. Initialise a backbone and a task-specific head.
5. Train the model on the pretext objective.

For example, with rotation prediction:

- Take an original image.
- Apply one of four possible rotations.
- Assign a numerical label corresponding to the rotation.
- Pass the rotated image to the model.
- Train the model to predict the rotation label.

The original CIFAR-10 labels are not required at any point.

### Q29. Why do we need a custom collate function?

The standard data loader returns batches of images.

However, the pretext task may require additional processing.

For rotation prediction, the collate function must:

- Receive a batch of unlabelled images.
- Apply a randomly selected rotation to each image.
- Generate the corresponding rotation labels.
- Convert the images into tensors.
- Return both the transformed images and the generated labels.

This is important because the training loop expects a pair of inputs and targets.

The generated labels are not human annotations. They are derived automatically from the transformation.

### Q30. Can we reuse the supervised training function?

Yes.

One of the useful design decisions in the notebook is that many pretext tasks can be treated as ordinary supervised learning problems once the training pairs have been generated.

For example, rotation prediction is simply a four-class classification problem.

We can therefore reuse:

python

```python
train_supervised(...)
```

We only need to provide:

- A pretext-specific data loader.
- A pretext-specific model.
- The correct number of output classes.

For regression-based pretext tasks or image reconstruction, we may need to modify the loss function and training loop.

### Q31. Which loss function should we use?

It depends on the pretext task.

For classification tasks, such as rotation prediction, we use:

python

```python
nn.CrossEntropyLoss()
```

This compares the predicted class logits with the correct integer targets.

For regression tasks, such as predicting colour transformation intensity, we could use:

python

```python
nn.MSELoss()
```

For image reconstruction, we might use a pixel-wise reconstruction loss.

For contrastive learning, we use a contrastive objective designed to compare feature representations.

The key principle is that the loss function must match the type of prediction required by the pretext task.

### Q32. How do we know whether pretext training is working?

We examine the training loss, validation loss and pretext accuracy.

Ideally, we want to see:

- Training loss gradually decreasing.
- Validation loss stabilising.
- Accuracy improving.
- No extreme instability during optimisation.

However, we should not interpret pretext accuracy in isolation.

A model achieving very high accuracy might simply be exploiting an easy shortcut.

The more important question is whether the learned backbone transfers successfully to the downstream task.

# 8. Task 2: Supervised Fine-Tuning

### Q33. What happens after self-supervised pre-training?

Once the model has learned useful representations from the unlabelled dataset, we transfer its backbone to the downstream classification task.

This is the second major stage of the practical.

We already have:

- A pre-trained backbone.
- A classification head designed for the pretext task.

We now want:

- The same pre-trained backbone.
- A new classification head designed for the target dataset.

We discard the pretext-specific head because it is no longer needed.

### Q34. Why do we replace the classification head?

The pretext task and downstream task have different prediction objectives.

For example:

| Pretext task | Downstream task |
| --- | --- |
| Predict image rotation | Predict object category |
| Four output classes | Five output classes |
| Learns orientation-related features | Learns category-related features |

The original head is designed to predict rotation.

It therefore cannot directly perform the target classification task.

We replace it with a new head whose output dimension matches the number of target classes.

### Q35. What does fine-tuning mean?

Fine-tuning means continuing to train a pre-trained model on a new task using the labelled data available for that task.

In our case:

1. We load the pre-trained backbone.
2. We attach a new classification head.
3. We train the resulting model using the labelled target dataset.
4. We evaluate the validation accuracy.

The backbone is no longer randomly initialised.

It already contains useful visual representations learned during self-supervised pre-training.

Fine-tuning adapts those representations to the target classification problem.

### Q36. Should we freeze the backbone or fine-tune all its parameters?

There are two common strategies.

Feature extraction / frozen backbone

We freeze the pre-trained backbone and train only the new classification head.

This preserves the learned representations.

It can be useful when the target dataset is very small or when we want to reduce computational costs.

Full fine-tuning

We update both the backbone and the classification head.

This allows the model to adapt its existing representations to the target dataset.

It can produce better performance when the source and target domains are sufficiently related.

The practical encourages fine-tuning and also suggests experimenting with layer freezing and different hyperparameters.

### Q37. Why might we use a lower learning rate during fine-tuning?

The backbone already contains useful learned representations.

If we use an excessively high learning rate, we might destroy these useful features through large parameter updates.

A lower learning rate allows the network to adapt more gradually.

This is particularly useful when the target dataset is small.

We may also experiment with stronger regularisation to reduce overfitting.

### Q38. How do we compare self-supervised learning with the baseline?

We compare the best validation accuracy of the two approaches.

The baseline is:

python

```python
baseline_acc = baseline_metrics.best_val_acc
```

The self-supervised result is:

python

```python
ft_acc = ft_metrics.best_val_acc
```

We then calculate:

python

```python
improvement = ft_acc - baseline_acc
```

If the improvement is positive, the self-supervised approach has outperformed the baseline.

For example, suppose we obtain:

- Baseline validation accuracy: 45%.
- Self-supervised fine-tuned validation accuracy: 55%.

The improvement is:

$$
55\%-45\%=10\text{ percentage points}
$$

This would suggest that self-supervised pre-training has helped the model learn representations that transfer successfully to the downstream task.

These figures are illustrative, not results reported by the notebook.

### Q39. What if self-supervised learning does not improve performance?

This is an important possible outcome.

Self-supervised learning does not automatically guarantee better downstream performance.

Several factors could explain a disappointing result:

- The pretext task may be too easy.
- The pretext task may not encourage useful semantic representations.
- The model may not have trained for long enough.
- The augmentations may be inappropriate.
- The source and target datasets may have insufficiently related visual distributions.
- The fine-tuning learning rate may be too high.
- The model may overfit the target training data.

The solution is to investigate the learning process rather than simply assume that self-supervised learning is ineffective.

We can experiment with different pretext tasks, augmentation strategies, training durations and fine-tuning hyperparameters.

# 9. Evaluation, Interpretation and Assessment

### Q40. What should we report in the final assessment?

The notebook asks us to address three main areas.

1. Description of the pretext task

Explain:

- Which pretext task was selected.
- Why it was selected.
- How it was implemented.
- What representations it was intended to encourage.

2. Performance comparison

Include plots comparing:

- Supervised baseline performance.
- Self-supervised pre-training followed by supervised fine-tuning.

Discuss the final validation accuracy and whether self-supervised learning improved performance.

3. Issues and insights

Explain:

- Any implementation difficulties.
- Problems encountered during training.
- How these problems were resolved.
- What was learned about pretext task design.
- Whether the learned representations transferred successfully.

### Q41. Why is validation accuracy more important than pretext accuracy?

The ultimate objective is to improve the downstream classification task.

The pretext task is only an intermediate learning objective.

A model could achieve excellent performance on the pretext task but learn representations that are not particularly useful for object classification.

For this reason, downstream validation accuracy is the most important indicator of whether the self-supervised approach has achieved its purpose.

### Q42. What is the main insight we should take away from this practical?

The main insight is that a neural network can learn useful representations without explicit human annotations.

By designing an appropriate pretext task, we can use large quantities of unlabelled data to initialise a model.

We can then transfer the learned backbone to a downstream task and fine-tune it using a relatively small labelled dataset.

This provides a way to reduce our dependence on labelled training data.

The practical also demonstrates that the quality of the learned representations depends strongly on the pretext task, the augmentations and the transfer learning strategy.

# 10. Overall Practical Workflow: From Unlabelled Data to Classification

The entire practical can be summarised in the following sequence.

1. Select the datasets
   Choose a large unlabelled source dataset and a smaller labelled target dataset.
2. Prepare the data
   Remove source labels, split target data into training and validation sets, and define preprocessing and augmentation.
3. Build the baseline
   Train a randomly initialised CNN on the labelled target data and record its best validation accuracy.
4. Design the pretext task
   Choose a task such as rotation prediction, jigsaw prediction or contrastive learning.
5. Pre-train the backbone
   Train on the unlabelled source data using automatically generated targets.
6. Transfer the backbone
   Keep the pre-trained feature extractor and replace its pretext-specific head.
7. Fine-tune on the target task
   Train using the labelled target dataset.
8. Evaluate and compare
   Compare validation accuracy against the supervised baseline and interpret the results.

# 11. Summary of Key Questions and Answers

The following table provides a concise revision guide covering the most important concepts from Practical 3.

| Key question | Short answer |
| --- | --- |
| What is self-supervised learning? | Learning representations using supervisory signals generated automatically from the data. |
| What is a pretext task? | An artificial learning task designed to encourage useful feature learning. |
| What is a downstream task? | The actual task we ultimately want the model to perform. |
| Why use unlabelled data? | It allows us to learn from large datasets without costly manual annotations. |
| Why use two datasets? | One provides unlabelled data for representation learning; the other defines the labelled target task. |
| Why remove CIFAR-10 labels? | To ensure pre-training is genuinely self-supervised. |
| What is the backbone? | The feature extractor that learns and produces visual representations. |
| What is the classification head? | The task-specific component that converts features into predictions. |
| Why separate backbone and head? | So that the backbone can be reused across different tasks. |
| What is data augmentation? | Applying transformations to create diverse versions of training images. |
| Why use augmentation? | To encourage robust representations and reduce reliance on superficial image details. |
| What makes a good pretext task? | It is learnable but challenging enough to encourage useful, transferable features. |
| Why can high pretext accuracy be misleading? | The model may solve an easy task without learning useful semantic representations. |
| What is contrastive learning? | Learning representations by bringing related examples closer and separating unrelated examples. |
| Why train a baseline? | To establish the reference performance of supervised learning without pre-training. |
| What is fine-tuning? | Adapting a pre-trained model to a new task using labelled data. |
| Why replace the pretext head? | Because the pretext and downstream tasks require different outputs. |
| Should we freeze the backbone? | We can freeze it for feature extraction or update it during full fine-tuning. |
| What is the most important evaluation metric? | Downstream validation accuracy. |
| How do we measure improvement? | Subtract baseline validation accuracy from fine-tuned validation accuracy. |
| Does self-supervised learning always improve performance? | No. Its effectiveness depends on task design, training and transferability. |

## Five particularly important questions to understand for an oral examination

1. Why is self-supervised learning useful when labelled data is limited?

Because we can exploit a large unlabelled dataset to learn general visual representations before using the smaller labelled dataset for the actual classification task.

2. Why is the pretext task important?

Because it determines what information the model learns to represent. A good pretext task encourages features that are useful for the downstream task rather than superficial patterns that only solve the artificial task.

3. What exactly do we transfer from pre-training to fine-tuning?

We transfer the backbone, including its learned parameters. We discard the pretext-specific classification head and attach a new head for the downstream classification task.

4. Why might self-supervised learning outperform supervised training from scratch?

The backbone has already learned useful visual representations from a larger collection of unlabelled images. Consequently, the downstream model has a better starting point and may require fewer labelled examples to learn the target task.

5. How do we determine whether self-supervised learning was successful?

By comparing the downstream validation accuracy of the self-supervised fine-tuned model against the supervised baseline, while also considering validation loss, training stability and generalisation.

## Final takeaway

The central message of Practical 3 is that we do not necessarily need labelled data to learn useful features.

Through self-supervised learning, we can create artificial prediction tasks from unlabelled images, train a feature extractor on those tasks, and transfer the resulting representations to a different classification problem.

The practical therefore demonstrates a powerful learning strategy:

Unlabelled data → Pretext task → Self-supervised pre-training → Feature transfer → Supervised fine-tuning → Improved downstream performance.

The most important lesson is not simply how to implement a particular pretext task, but how to design a learning objective that encourages representations that remain useful beyond the task on which they were originally learned.
