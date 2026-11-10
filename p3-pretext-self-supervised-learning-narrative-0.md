# Practical 3 – Pretext Tasks and Self-Supervised Learning

Questions-and-Answers Focused Narrative Script

Format: Audio-only instructional narration Audience: Advanced Machine Learning students Assumption: Students can hear the narrator but cannot see him or any visual presentation. Therefore, all explanations, questions, code concepts, and transitions must be communicated verbally. Students should be able to follow the practical using audio alone, without needing to see the instructor demonstrate anything.

## 1. Introduction and learning objectives

Narrator:

Welcome to Practical 3 of Advanced Machine Learning.

In this practical, we are going to explore an important machine learning approach called self-supervised learning, with a particular focus on something known as pretext learning.

Before we start working through the notebook, let's establish the main question that will guide this practical.

Question 1: What is the main objective of this practical?

The objective is to improve the performance of a machine learning model on a classification task when we have only a relatively small amount of labelled training data.

We will achieve this by first training the model on a much larger dataset that does not have any labels.

The idea is to teach the model to recognise useful patterns and visual representations without explicitly telling it what each image represents.

We will then transfer the knowledge acquired during this initial training stage to a classification task where labelled examples are available.

This is called self-supervised pre-training followed by supervised fine-tuning.

Question 2: Why is this approach useful?

In many real-world applications, collecting and manually labelling large datasets is expensive, time-consuming, or simply impractical.

However, collecting images without labels is often much easier.

For example, we might have thousands of photographs but no information about the objects contained in them.

Self-supervised learning allows us to exploit these unlabelled images to learn useful representations before introducing the smaller labelled dataset.

The hope is that the model will perform better than a model trained only on the limited labelled data.

Question 3: How is self-supervised learning different from supervised learning?

In supervised learning, the model receives an input and its corresponding correct label.

For example, an image of a dog is provided together with the label 'dog'.

The model learns to associate visual patterns with the correct class.

In self-supervised learning, we do not provide manually assigned labels.

Instead, we create a learning objective directly from the data itself.

For example, we might rotate an image and ask the model to predict how much it was rotated.

The original image already contains all the information needed to construct the training target.

This is why the approach is called self-supervised: the data provides its own supervision.

Question 4: What is a pretext task?

A pretext task is an artificial learning problem that we deliberately design to teach the model useful representations.

It is not necessarily the task we ultimately want the model to perform.

Instead, it is a preliminary task that encourages the model to learn visual features that can later be reused.

For example, predicting image rotation is not normally the final objective of an image classification application.

However, to predict rotation correctly, the model may need to recognise object shapes, structures, and spatial relationships.

These learned representations may subsequently help it distinguish between different object categories.

That is the central idea behind this practical.

## 2. Understanding the learning scenario

Narrator:

Let's now examine the datasets and the learning problem we are going to implement.

We will work with two datasets.

The first is a small labelled dataset, which defines our downstream classification task.

The second is a larger unlabelled dataset, which we will use for self-supervised pre-training.

Question 5: What is the difference between the source dataset and the target dataset?

The source dataset is the dataset used during the initial self-supervised training stage.

It contains images, but we deliberately remove or ignore their original labels.

The target dataset is the dataset used for our final classification task.

It contains labelled images, and its labels define what we ultimately want the model to predict.

Think of the source dataset as the material we use to teach the model general visual representations.

The target dataset is where we teach the model how to use those representations for a specific classification problem.

Question 6: Which datasets are used in this practical?

The notebook offers two possible configurations.

The first configuration uses CIFAR-100 as the labelled target dataset and CIFAR-10 as the unlabelled source dataset.

The second configuration uses STL-10, which provides both labelled and unlabelled image collections.

For the default configuration in the notebook, we use CIFAR.

The CIFAR-100 dataset provides the labelled target examples.

We select one superclass category containing five related object classes.

For example, we might select vehicles, which includes bicycles, buses, motorcycles, pickup trucks, and trains.

The notebook then extracts the images belonging to those five classes.

Meanwhile, CIFAR-10 provides the unlabelled source images.

Its original labels are deliberately destroyed or ignored.

This gives us a large collection of images that can be used for self-supervised learning.

Question 7: Why do we use datasets from different classes?

This is an important design choice.

The source and target datasets do not need to contain exactly the same object categories.

What matters is that they share useful visual characteristics.

For example, both datasets contain images with edges, shapes, textures, colours, backgrounds, and spatial structures.

A model trained to recognise these general characteristics may learn features that remain useful when it encounters new object categories.

This is the advantage we want to exploit.

Question 8: What is the difference between this practical and the previous semi-supervised learning practical?

In semi-supervised learning, we used labelled and unlabelled examples to improve performance on essentially the same classification problem.

The unlabelled examples helped the model learn additional information about the target data distribution.

In this practical, the source and target datasets can belong to different domains or contain different object categories.

The initial training task is also different from the final classification task.

We first learn representations using an artificial pretext task.

We then transfer those representations to the actual classification problem.

Therefore, the main objective is not to predict the original classes of the source dataset.

It is to learn features that can be reused for another task.

## 3. Notebook setup and dataset preparation

Narrator:

Let's move to the first technical part of the notebook.

We begin by importing the necessary libraries.

The notebook uses PyTorch for model construction and training, torchvision for image datasets and transformations, NumPy for numerical operations, and Matplotlib for visualising training results.

We also import TorchInfo to inspect the architecture and parameter counts of our neural networks.

There is no need to memorise every import statement.

The important point is to understand the purpose of the different components.

Question 9: Why do we need custom dataset classes?

The notebook defines two custom dataset classes.

The first is called `UnlabelledDataset`.

It stores images without any associated targets.

Its purpose is to provide images for self-supervised training.

The second is called `LabelledDataset`.

It stores images together with their corresponding labels.

This is the dataset we use for supervised classification and validation.

Separating these two dataset types helps us manage the different stages of training.

Question 10: What is the purpose of the TrainingMetrics class?

The TrainingMetrics class records the training and validation performance.

It stores information such as training loss, validation loss, training accuracy, and validation accuracy.

It also keeps track of the unsupervised loss when we implement a suitable pretext task.

Finally, it provides plotting functionality so that we can examine how the model learns over time.

This is important because we do not want to evaluate a model only through its final accuracy.

We also want to understand whether its training is stable, whether it is overfitting, and whether its loss is converging.

Question 11: Why do we split the labelled target dataset into training and validation subsets?

We need two separate subsets to evaluate whether our model generalises to images it has not seen during training.

The training subset is used to update the model's parameters.

The validation subset is used to measure performance during training.

In the notebook, ten percent of the labelled target dataset is reserved for validation.

The random seed is fixed to make the split reproducible.

This is important because we want to compare the baseline model and the self-supervised model using exactly the same validation examples.

Question 12: Why do we preprocess and normalise the images?

Neural networks generally work better when their input data has a consistent numerical representation.

The notebook converts images into tensors and normalises their pixel values.

This ensures that images are represented in a suitable format for PyTorch.

The notebook also defines stronger random data augmentations.

These include random resized cropping, horizontal flipping, colour jitter, and random erasing.

Data augmentation creates variations of the original images.

This helps the model become less dependent on specific image characteristics and improves its ability to generalise.

However, when designing a pretext task, we must ensure that the augmentations do not destroy the information needed to solve that task.

We will return to this point later.

## 4. Understanding the model architecture

Narrator:

Before we train the model, we need to understand how it is structured.

One of the most important ideas in this practical is the separation between the feature extraction component and the classification component.

The notebook implements this using a backbone and a classification head.

Question 13: What is a backbone in a neural network?

The backbone is the part of the model responsible for extracting useful features from the input images.

In this practical, the backbone is a convolutional neural network.

It contains several convolutional layers, activation functions, pooling operations, and fully connected layers.

The convolutional layers learn visual patterns.

Early layers may learn relatively simple features such as edges and textures.

Deeper layers can combine these patterns into more complex representations.

The final output of the backbone is a feature vector.

This vector represents the information extracted from the original image.

Question 14: What is a classification head?

The classification head takes the feature vector produced by the backbone and converts it into predictions for a particular classification task.

In this notebook, the classification head contains a projection layer followed by a final classification layer.

The projection layer transforms the backbone features into a smaller intermediate representation.

The final layer produces one output score for each target class.

These scores are often called logits.

The model uses these logits to calculate the classification loss and make predictions.

Question 15: Why do we separate the backbone from the classification head?

Because we want to reuse the backbone for different learning tasks.

During self-supervised pre-training, the backbone is trained to extract features that help solve the pretext task.

However, the output layer used for the pretext task is specific to that task.

When we move to the downstream classification problem, we need a different classification head.

We therefore keep the trained backbone and replace the original task-specific head.

This modular architecture is what makes transfer learning possible.

Question 16: What is the purpose of the ConvBackbone class in the notebook?

The ConvBackbone class implements the convolutional feature extractor.

It receives an image tensor and passes it through several convolutional operations.

After the convolutional layers, adaptive average pooling reduces the spatial dimensions.

The resulting representation is flattened and passed through fully connected layers.

Finally, the model produces a feature vector.

The notebook sets the backbone output size to 64.

This means that each image is represented by a vector containing 64 learned features.

These features are then passed to whichever task-specific head is attached to the backbone.

## 5. Establishing the supervised baseline

Narrator:

We now reach the first major experiment.

Before introducing self-supervised learning, we need to establish how well a model performs when trained only on the labelled target dataset.

This is our baseline.

Question 17: What is a baseline model, and why do we need one?

A baseline model provides a reference performance against which we can compare our improved approach.

In this practical, the baseline model consists of our convolutional backbone and a classification head.

Both components are randomly initialised.

The model is trained directly on the labelled target training data.

It does not use the unlabelled source dataset.

After training, we evaluate it on the target validation dataset.

The resulting validation accuracy becomes our reference point.

Question 18: Which loss function is used for baseline training?

The notebook uses cross-entropy loss.

Cross-entropy loss is commonly used for multi-class classification.

It measures how well the model's predicted class probabilities correspond to the correct labels.

A high loss indicates that the model is making poor predictions.

A lower loss generally indicates better predictions.

During training, the model attempts to minimise this loss by adjusting its parameters.

Question 19: Which optimiser is used?

The notebook uses AdamW.

AdamW is an adaptive optimisation algorithm that updates the model parameters using information from the gradients.

It also supports weight decay, which helps regularise the model.

The notebook defines a learning rate of 0.001 and a weight decay value of 0.1 for the fine-tuning configuration.

The baseline is trained for 30 epochs.

Question 20: What does an epoch represent?

An epoch represents one complete pass through the training dataset.

During an epoch, the model processes all the training batches.

For each batch, it calculates predictions, computes the loss, performs backpropagation, and updates its parameters.

After completing the epoch, the notebook evaluates the model on the validation dataset.

This process is repeated until the specified number of epochs has been reached or early stopping is triggered.

Question 21: Why might the baseline model overfit?

Overfitting happens when a model learns the training examples too specifically and fails to generalise well to new examples.

This is particularly relevant in our practical because the target dataset is relatively small.

The model may learn to memorise patterns in the training images rather than learning sufficiently general visual representations.

We may observe training accuracy continuing to improve while validation accuracy stops improving or begins to decrease.

This is precisely the limitation we want to address through self-supervised pre-training.

Narrator:

Once the baseline training is complete, the notebook records its best validation accuracy.

Let's call this value our baseline accuracy.

We will compare it with the accuracy obtained after self-supervised pre-training and supervised fine-tuning.

Remember that our objective is to improve performance on the target classification task, not simply to achieve high accuracy on the pretext task.

## 6. Task 1 – Designing and implementing a pretext task

Narrator:

We now arrive at the most important and creative part of this practical.

We need to design our own pretext task.

Remember, the source dataset contains images but no labels.

We therefore need to create a learning problem that provides useful supervision without requiring manually annotated classes.

Question 22: What makes a good pretext task?

A good pretext task should encourage the model to learn meaningful visual representations.

Ideally, solving the task should require the model to understand more than simple pixel-level information.

For example, a task that requires the model to recognise object structures, spatial relationships, or visual similarities may encourage it to learn features that are useful for downstream classification.

However, a task that is too easy may not teach the model anything particularly useful.

We therefore need to find a balance.

The pretext task should be sufficiently challenging to encourage useful learning, but not so difficult that the model cannot converge.

Question 23: What pretext tasks can we implement?

The notebook suggests several possibilities.

Let's examine them one by one.

The first possibility is rotation prediction.

We randomly rotate an image by one of four possible angles: zero degrees, ninety degrees, one hundred and eighty degrees, or two hundred and seventy degrees.

The model must predict which rotation was applied.

To solve this task, the model needs to learn information about object orientation and spatial structure.

The second possibility is colour transformation prediction.

We apply a colour transformation to an image and ask the model to estimate the intensity of that transformation.

This can be formulated as a regression task.

The third possibility is crop prediction.

We crop an image and ask the model to predict information about the crop parameters.

This encourages the model to understand spatial information.

The fourth possibility is relative patch position prediction.

We extract two image patches and ask the model to predict their relative positions in the original image.

This encourages the model to learn spatial relationships.

The fifth possibility is the jigsaw puzzle task.

We divide an image into multiple patches, rearrange them, and ask the model to predict their original arrangement.

Finally, we can use a contrastive approach.

We provide two images or image crops and ask the model to determine whether they originate from the same original image.

This encourages the model to learn representations that capture visual similarity.

Question 24: Which pretext task should we choose?

There is no single universally correct answer.

The choice depends on the amount of time available, computational resources, and the complexity of the implementation.

For a first implementation, rotation prediction is a reasonable starting point.

It is relatively straightforward because it converts an unlabelled image into an image-label pair.

For example, an image rotated by ninety degrees receives a corresponding rotation label.

We can therefore reuse much of the supervised training infrastructure already implemented in the notebook.

However, the task must be designed carefully.

If the model learns to predict rotation using superficial image characteristics, the resulting features may not transfer particularly well.

A more sophisticated task may encourage richer representations.

Question 25: Why should we avoid making the pretext task too easy?

Suppose we create a task where the model simply predicts whether an image is brighter or darker.

It might solve this problem by learning basic pixel intensity statistics.

This does not necessarily encourage it to understand object structure or semantic information.

Consequently, the learned features may be of limited value for our final classification task.

The objective is not simply to make the model good at the artificial task.

The objective is to make the model learn representations that are useful for other tasks.

Question 26: What role does data augmentation play in pretext learning?

Data augmentation creates different versions of the same image.

This encourages the model to learn representations that are robust to changes in the input.

For example, we might randomly crop an image, flip it horizontally, modify its colours, or convert it to grayscale.

These transformations help prevent the model from relying too heavily on specific visual details.

However, augmentations must be compatible with the pretext task.

For example, if we ask the model to predict whether an image has been rotated by ninety degrees, applying another random rotation afterwards could change the correct answer.

The transformation used to generate the pretext label must remain distinguishable from the additional augmentation.

This is an important practical consideration.

Question 27: How do we implement the pretext task in the notebook?

There are several steps.

First, we define the transformation that generates the pretext input.

Second, we determine the corresponding target label or regression value.

Third, we create a data-loading procedure that returns the transformed images and their automatically generated targets.

Fourth, we initialise a backbone and attach a task-specific head.

Finally, we train the model using an appropriate loss function.

For a classification-based pretext task, we can often reuse the supervised training loop.

For regression-based or image-reconstruction tasks, we may need to modify the loss function and training loop.

Narrator:

Notice something important about the notebook at this stage.

The actual implementation of the pretext task is deliberately left to you.

The notebook provides the infrastructure, possible approaches, and suggested transformations.

You are expected to design and implement the task yourself.

Therefore, the code section marked as the implementation of your pretext task is where your own solution must be inserted.

There is no single pretext-task implementation already completed in the supplied notebook.

## 7. Training the self-supervised model

Narrator:

Once we have designed the pretext task, we can begin self-supervised pre-training.

The objective is to train the backbone to extract features that make the pretext task easier to solve.

Question 28: What happens during self-supervised pre-training?

The model receives images from the unlabelled source dataset.

The pretext transformation is applied to each image.

The model then predicts the automatically generated target.

We calculate the pretext loss and use backpropagation to update the model parameters.

This process is repeated over multiple batches and epochs.

The backbone gradually learns representations that help it solve the artificial task.

Question 29: Should we expect very high accuracy on the pretext task?

Not necessarily.

A high pretext accuracy is not automatically evidence of good representation learning.

In fact, if the model achieves extremely high accuracy very quickly, the task might be too easy.

For example, it might be exploiting simple visual shortcuts instead of learning meaningful features.

What we want is a task that converges stably while encouraging the model to learn useful representations.

The most important evaluation will ultimately happen on the downstream classification task.

Question 30: How do we know whether pre-training is working?

We monitor the training loss and, where appropriate, the pretext validation loss and accuracy.

We want to see the loss decreasing over time.

We also want training to remain stable rather than producing highly fluctuating results.

If the model is not learning, we may need to reconsider the task design, the learning rate, the transformations, or the training duration.

It is also useful to inspect transformed image batches manually.

This allows us to verify that the transformations and generated labels are correct.

For example, if we use rotation prediction, we should check that the image rotation and its associated label agree.

An error in this relationship would prevent the model from learning the intended task.

Question 31: Why might self-supervised pre-training require more training than supervised fine-tuning?

The pretext task may be more difficult than the final classification task.

Furthermore, the model is learning general-purpose representations rather than directly learning the target classes.

It may therefore require more iterations before its features become useful.

The notebook explicitly encourages us to allow the pretext loss to converge sufficiently before moving to fine-tuning.

## 8. Task 2 – Supervised fine-tuning

Narrator:

We have now reached the second major stage of the practical.

We have trained a backbone using our self-supervised pretext task.

The next step is to transfer the learned representations to the original classification problem.

This is called supervised fine-tuning.

Question 32: What is supervised fine-tuning?

Fine-tuning means taking a model that has already learned useful parameters and adapting it to a new task.

In our case, we take the backbone trained on the unlabelled source dataset.

We then attach a new classification head designed for the labelled target dataset.

The new head predicts the actual target classes.

We subsequently train the combined model using the labelled target training data.

Question 33: Why do we replace the original pretext classification head?

Because the two tasks have different objectives.

The pretext head predicts something such as image rotation or relative patch position.

The target classification head predicts the actual object categories.

For example, if our target dataset contains five classes, the final classification layer must produce five output scores.

The pretext head is therefore no longer required.

However, the backbone contains the representations learned during pre-training.

Those are the parameters we want to preserve and reuse.

Question 34: What exactly do we transfer from the self-supervised model?

We transfer the trained backbone parameters.

In PyTorch, this is commonly done by saving the backbone's state dictionary and loading it into the backbone of the new target model.

The new target classification head is initialised separately.

We then combine the transferred backbone and the new head.

This gives us a model that starts the downstream classification task with previously learned visual representations rather than completely random initialisation.

Question 35: Should we freeze the backbone or fine-tune all its parameters?

There are two main possibilities.

The first is to freeze the backbone.

In this case, the backbone parameters remain unchanged, and only the new classification head is trained.

This is useful when we want to preserve the learned representations or when we have extremely limited labelled data.

The second is to fine-tune the entire model.

In this case, both the backbone and classification head are updated using the target labels.

This allows the model to adapt its previously learned features to the new classification task.

The notebook focuses on fine-tuning, although layer freezing can also be explored.

Question 36: Should we use the same hyperparameters as the baseline model?

For a fair comparison, it is important to keep the experimental conditions as consistent as possible.

The notebook suggests using the same fine-tuning configuration as the baseline.

This includes the number of epochs, learning rate, and regularisation.

However, self-supervised models may benefit from different hyperparameters.

For example, we might use a lower learning rate to avoid changing the pre-trained backbone too aggressively.

The important thing is to document any differences and ensure that the comparison remains meaningful.

Question 37: How do we evaluate whether self-supervised learning has helped?

We evaluate the fine-tuned model on exactly the same target validation dataset used for the baseline.

We then compare the best validation accuracy of the self-supervised model with the best validation accuracy of the baseline.

If the fine-tuned model performs better, this suggests that the pre-trained representations have helped the downstream classification task.

If performance is similar or worse, we need to investigate why.

Possible explanations include an unsuitable pretext task, insufficient pre-training, incompatible source and target distributions, or inappropriate fine-tuning settings.

## 9. Comparing the results and interpreting the learning curves

Narrator:

We now reach the experimental evaluation.

This is where we determine whether our approach has achieved its objective.

The notebook plots training and validation loss and accuracy.

It also allows us to compare the fine-tuning curves with the baseline curves.

Question 38: What should we look for when comparing the baseline and fine-tuned models?

First, compare their best validation accuracies.

The self-supervised model should ideally achieve higher validation accuracy than the supervised baseline.

Second, examine the validation loss.

A lower validation loss may indicate that the model is making more reliable predictions.

Third, examine the training curves.

We want to understand whether the self-supervised model converges more effectively or shows less overfitting.

However, we should not interpret a single metric in isolation.

Validation accuracy is the primary performance measure for our practical objective.

The training curves provide additional information about how the model learned.

Question 39: What does it mean if the self-supervised model outperforms the baseline?

It means that the representations learned during pre-training have provided a useful starting point for the downstream classification task.

The model has benefited from exposure to a larger collection of unlabelled images.

This suggests that it learned visual features that were transferable to the target classes.

This is the desired outcome of self-supervised learning.

Question 40: What if the self-supervised model performs worse than the baseline?

This does not necessarily mean that self-supervised learning is ineffective.

It may indicate that our particular implementation was unsuccessful.

We should investigate several possibilities.

Perhaps the pretext task was too easy.

Perhaps it was too difficult to optimise.

Perhaps the model learned low-level features that were not useful for the target classification task.

Perhaps the pre-training duration was insufficient.

We should also check whether the data transformations and automatically generated labels were implemented correctly.

Finally, we should examine the similarity between the source and target datasets.

A good self-supervised representation is not guaranteed to transfer equally well between every pair of datasets.

Question 41: Why is it important to compare models under fair experimental conditions?

Because otherwise, we cannot confidently attribute any performance improvement to self-supervised learning.

For example, imagine that the fine-tuned model is trained for twice as many epochs as the baseline.

An improvement might simply result from additional training rather than the pre-training strategy.

Similarly, using different validation samples would make the comparison less reliable.

We therefore need to keep the target training and validation splits consistent and carefully document the experimental settings.

## 10. Final assessment and reflection

Narrator:

Before concluding the practical, let's review what you are expected to demonstrate in your assessment.

There are three main components.

First, describe your pretext task.

Explain what the model is asked to predict, why you selected that task, and how you implemented it.

Second, report your experimental results.

You should include plots comparing the supervised baseline with the self-supervised model after fine-tuning.

Your discussion should focus particularly on validation accuracy.

Third, reflect on the implementation.

Describe any difficulties you encountered, how you addressed them, and what you learned about designing useful pretext tasks.

You should also explain whether the learned representations transferred successfully to the downstream classification problem.

Question 42: What is the most important lesson from this practical?

The most important lesson is that useful representations can be learned without manually labelled data.

By constructing an artificial learning objective, we can encourage a neural network to discover patterns and structures in unlabelled images.

We can then transfer this knowledge to a different task where labelled data is available.

The effectiveness of this approach depends heavily on the quality of the pretext task.

A successful pretext task should encourage the model to learn representations that are relevant beyond the artificial problem used during pre-training.

That is the central principle of self-supervised learning.

Narrator – Closing:

To conclude, remember the overall workflow.

We start with a small labelled target dataset and establish a supervised baseline.

We then use a larger unlabelled source dataset to design and train a self-supervised pretext task.

Next, we transfer the learned backbone to the target classification problem, replace the pretext head with a new classification head, and fine-tune the model.

Finally, we compare its validation performance with the original supervised baseline.

The key question throughout the practical is not simply whether the model can solve the pretext task.

It is whether solving that task helps the model learn better representations and ultimately improve its performance on the real classification problem.

Good luck with the practical\!

# Summary: Key questions and answers

The following is a final audio-friendly revision section covering the most important questions from the practical.

| Key question | Short answer |
| --- | --- |
| 1. What is self-supervised learning? | Learning useful representations from unlabelled data by generating supervision from the data itself. |
| 2. What is a pretext task? | An artificial learning task designed to encourage the model to learn transferable features. |
| 3. What is the objective of Practical 3? | Improve downstream classification accuracy using self-supervised pre-training. |
| 4. What is the source dataset? | The large unlabelled dataset used for pre-training. |
| 5. What is the target dataset? | The smaller labelled dataset used for the final classification task. |
| 6. Why can source and target classes be different? | They can share useful visual features even when their object categories differ. |
| 7. What is a backbone? | The neural network component that extracts features from images. |
| 8. What is a classification head? | The task-specific component that converts features into class predictions. |
| 9. Why establish a baseline? | To measure whether self-supervised learning improves performance. |
| 10. What is a good pretext task? | One that is learnable but sufficiently challenging to encourage meaningful representations. |
| 11. Give examples of pretext tasks. | Rotation prediction, colour prediction, crop prediction, jigsaw puzzles, and contrastive learning. |
| 12. Why use data augmentation? | To encourage robust and generalisable representations. |
| 13. Why must augmentations be chosen carefully? | They must not destroy or contradict the information needed to solve the pretext task. |
| 14. What happens during pre-training? | The backbone learns to solve the pretext task using unlabelled images. |
| 15. What is fine-tuning? | Adapting the pre-trained model to the actual target classification task. |
| 16. What do we transfer? | The trained backbone parameters. |
| 17. Why replace the pretext head? | Because the pretext and downstream tasks require different predictions. |
| 18. What is the main evaluation metric? | Validation accuracy on the target classification task. |
| 19. What if self-supervised learning does not improve accuracy? | Investigate the pretext task, training duration, augmentations, data compatibility, and fine-tuning settings. |
| 20. What is the central lesson? | Unlabelled data can help a model learn transferable visual representations that improve supervised learning. |

## The complete practical workflow to remember

1. Choose the target dataset
   Select a small labelled dataset defining the final classification problem.
2. Train the supervised baseline
   Train a randomly initialised backbone and classification head using labelled target data.
3. Choose the source dataset
   Select a larger dataset and remove or ignore its labels.
4. Design the pretext task
   Generate artificial training targets from the unlabelled images.
5. Perform self-supervised pre-training
   Train the backbone to solve the pretext task.
6. Transfer the backbone
   Keep the pre-trained feature extractor and replace the pretext head with a new target classification head.
7. Fine-tune the model
   Train on the labelled target dataset.
8. Compare the results
   Evaluate validation accuracy and compare against the supervised baseline.

Final takeaway for students:

Self-supervised learning changes how we use data. Instead of requiring every training image to have a manually assigned label, we create learning objectives from the images themselves. By choosing an appropriate pretext task, we can train a backbone to discover useful visual representations and subsequently reuse those representations for a different supervised learning problem.

Note for the practical: The supplied notebook intentionally leaves the pretext-task implementation and fine-tuning implementation as student exercises. Consequently, no specific pretext-task results or accuracy improvements can be assumed in this narrative.
