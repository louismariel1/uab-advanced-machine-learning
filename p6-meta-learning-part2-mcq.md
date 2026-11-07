Absolutely\! We'll follow exactly the same two-step workflow for every part of Practical 6.

For Part 2, we are now completing Step 1 only: generating the MCQ questions with answers and explanations. Once you're happy with this, we'll move to Step 2 and convert them into an interactive practice exam.

# Advanced Machine Learning — Practical 6

## Part 2: The Baseline Supervised Model

Step 1 — MCQ with answers and explanationsTopics covered (Q6–Q9 of the summary):

1. Why we need a supervised baseline.
2. The backbone and classification head.
3. Cross-entropy, accuracy and weight decay.
4. Fine-tuning and adaptation to new few-shot tasks.

20 multiple-choice questions, progressing from fundamental concepts to more challenging exam-style questions.

## Section A — Understanding the baseline

### Question 1. Why does the assignment train a baseline supervised model before implementing meta-learning?

A. To guarantee that ProtoNet achieves 100% accuracy.

B. To establish a reference point for comparing meta-learning approaches.

C. To eliminate the need for a feature extractor.

D. To avoid evaluating unseen classes.

Correct answer: BExplanation:

The baseline establishes a benchmark against which Reptile and ProtoNet can be compared.

It helps answer an important research question:

Does meta-learning improve few-shot classification compared with ordinary supervised pre-training?

Without a baseline, we would not know whether the meta-learning methods actually provide an improvement.

### Question 2. What are the two main components of the baseline model?

A. Support set and query set.

B. Optimizer and scheduler.

C. Backbone and classification head.

D. Prototype and distance function.

Correct answer: CExplanation:

The baseline model consists of:

- Backbone: Extracts useful visual features from images.
- Classification head: Converts those features into class scores (logits).

The two components work together to perform classification.

### Question 3. What is the primary function of the backbone?

A. Calculate cross-entropy loss.

B. Extract useful visual features from images.

C. Assign final class labels directly.

D. Generate support and query datasets.

Correct answer: BExplanation:

The backbone learns useful visual representations, such as:

- Edges.
- Textures.
- Shapes.
- Object parts.
- Visual patterns.

These representations are passed to the classification head.

Mathematically:

$$
x\longrightarrow f_\theta(x)
$$

Here, $f_\theta(x)$ represents the extracted feature vector.

### Question 4. What is the primary function of the classification head?

A. Convert extracted features into class logits.

B. Extract image pixels.

C. Create new training images.

D. Calculate Euclidean distances between support images.

Correct answer: AExplanation:

The classification head receives the feature representation produced by the backbone and generates one score per class.

For example, if the dataset contains 20 meta-training classes, the classification head produces 20 logits.

### Question 5. Why can the backbone be reused for few-shot learning?

A. It already knows every possible future class.

B. It has learned general visual features that may transfer to unfamiliar categories.

C. It does not contain trainable parameters.

D. It automatically creates a new dataset.

Correct answer: BExplanation:

The backbone learns general visual representations that can be useful for previously unseen categories.

Rather than learning everything from scratch, few-shot learning can reuse these representations and train a new task-specific classifier using a small support set.

## Section B — Backbone and classification head

### Question 6. Consider the following code:

python

```python
baseline_model = nn.Sequential(
    baseline_backbone,
    baseline_classifier
)
```

What does `nn.Sequential` do here?

A. Trains both models independently.

B. Combines the backbone and classifier into a sequential model.

C. Creates two separate datasets.

D. Freezes all model parameters.

Correct answer: BExplanation:

`nn.Sequential` combines the two components into one model.

The data flows as follows:

$$
\text{Image}
\rightarrow
\text{Backbone}
\rightarrow
\text{Features}
\rightarrow
\text{Classifier}
\rightarrow
\text{Logits}
$$

The output of the backbone automatically becomes the input of the classification head.

### Question 7. The backbone produces a feature vector of dimension 64. There are 20 meta-training classes. What should the classification head output?

A. 5 values.

B. 20 class logits.

C. 64 class logits.

D. 1 class logit.

Correct answer: BExplanation:

The classification head needs one output logit for each class.

Therefore:

- Input dimension: 64.
- Output dimension: 20.

The resulting tensor for a batch of 32 images would have shape:

$$
(32,20)
$$

### Question 8. Why does the baseline classification head use the number of meta-training classes?

A. Because it must predict the classes used during supervised pre-training.

B. Because every future task must contain the same classes.

C. Because the number of classes determines image dimensions.

D. Because it determines the learning rate.

Correct answer: AExplanation:

The original classification head is designed specifically for the meta-training categories.

However, meta-test tasks may contain completely different categories.

This explains why the baseline evaluation creates a new task-specific classification head.

### Question 9. What does the mathematical expression $f_\theta(x)$ represent?

A. The final cross-entropy loss.

B. The feature embedding extracted from image $x$ by the backbone.

C. The predicted class label.

D. The learning rate.

Correct answer: BExplanation:

The expression represents the feature vector produced by the backbone.

- $x$: Input image.
- $f$: Feature extraction function.
- $\theta$: Backbone parameters.

The output is a numerical representation of the input image.

### Question 10. Why is it useful to separate the backbone from the classification head?

A. It allows the backbone to be reused with a different task-specific classifier.

B. It removes the need for training.

C. It guarantees perfect generalisation.

D. It prevents the model from learning features.

Correct answer: AExplanation:

This separation provides flexibility.

The backbone can retain useful representations learned from the original classes, while the classification head can be replaced to recognise new classes.

This is particularly useful in few-shot learning.

## Section C — Loss functions and evaluation metrics

### Question 11. What is the purpose of cross-entropy loss in baseline training?

A. Measure how well predicted class probabilities match the correct labels.

B. Measure the size of the images.

C. Calculate the number of model parameters.

D. Determine the number of training episodes.

Correct answer: AExplanation:

Cross-entropy measures classification error.

It compares the predicted class distribution with the true class labels.

The training objective is to minimise cross-entropy loss.

### Question 12. What does accuracy measure?

A. The magnitude of model weights.

B. The proportion of correctly classified examples.

C. The number of convolutional layers.

D. The learning rate of the optimizer.

Correct answer: BExplanation:

Accuracy measures the proportion of correct predictions.

$$
\text{Accuracy}
=
\frac{\text{Correct predictions}}
{\text{Total predictions}}
$$

For example, if a model correctly classifies 80 out of 100 images:

$$
\text{Accuracy}=80\%
$$

### Question 13. What is the purpose of top-5 accuracy?

A. A prediction is correct only if the model produces exactly five classes.

B. A prediction is correct if the true class is among the five highest-scoring predictions.

C. The model must classify five images simultaneously.

D. The model must achieve at least 5% accuracy.

Correct answer: BExplanation:

Top-5 accuracy considers the five classes with the highest predicted scores.

A prediction counts as correct if the true class appears anywhere in those five predictions.

This is useful when the model classifies many categories.

### Question 14. Why does the notebook use top-1 accuracy for five-way meta-test tasks?

A. Because there are no class labels.

B. Because the highest-scoring class must be the correct prediction.

C. Because top-5 accuracy cannot be calculated mathematically.

D. Because the model has five backbones.

Correct answer: BExplanation:

In a five-way task, the model predicts one of five possible classes.

Top-1 accuracy checks whether the class with the highest score is correct.

This is the standard accuracy measure for the few-shot evaluation tasks.

### Question 15. What is weight decay (L2 regularisation) intended to do?

A. Increase the number of training classes.

B. Discourage unnecessarily large parameter values.

C. Remove the classification head.

D. Increase image resolution.

Correct answer: BExplanation:

Weight decay discourages excessively large model weights.

It helps regularise the model and can reduce overfitting.

In the practical, the baseline uses:

python

```python
l2_reg = 1e-2
```

This corresponds to a weight decay value of 0.01.

## Section D — Optimisation and hyperparameters

### Question 16. In the notebook, what does `lr = 1e-2` mean?

A. The learning rate is 0.1.

B. The learning rate is 0.01.

C. The learning rate is 2.

D. The learning rate is 10.

Correct answer: BExplanation:

Scientific notation:

$$
1e^{-2}=1\times10^{-2}=0.01
$$

The learning rate controls the size of parameter updates during optimisation.

### Question 17. What is the purpose of momentum in SGD?

A. To help updates build on consistent gradient directions.

B. To calculate class prototypes.

C. To freeze the backbone.

D. To calculate query accuracy.

Correct answer: AExplanation:

Momentum incorporates information from previous gradients.

This can accelerate optimisation when gradients consistently point in similar directions and help reduce oscillations.

### Question 18. What does cosine annealing do in the baseline training process?

A. Gradually adjusts the learning rate according to a cosine schedule.

B. Changes the number of classes every epoch.

C. Calculates cosine similarity between images.

D. Replaces cross-entropy with Euclidean distance.

Correct answer: AExplanation:

Cosine annealing changes the learning rate according to a cosine-shaped schedule.

The learning rate generally decreases during training, helping the model make smaller and more stable updates towards the end.

## Section E — Fine-tuning and few-shot adaptation

### Question 19. During few-shot evaluation, the baseline uses `freeze_backbone=True`. What does this mean?

A. The backbone is randomly reinitialised.

B. The backbone parameters are not updated during task-specific fine-tuning.

C. The classification head is permanently removed.

D. The entire model is frozen.

Correct answer: BExplanation:

The pre-trained backbone is kept fixed.

Only the newly created task-specific classification head is trained using the support examples.

This tests whether the visual features learned during supervised pre-training are already useful for recognising unfamiliar categories.

### Question 20. What does `full_state=False` and `freeze_backbone=True` mean in the baseline evaluation?

A. The complete original model and head are reused and updated.

B. The backbone is copied into a task model and frozen while a new task-specific head is trained.

C. The backbone and head are both randomly initialised.

D. The model is evaluated without support examples.

Correct answer: BExplanation:

The baseline evaluation:

1. Reuses the pre-trained backbone.
2. Creates a new classification head for the current task.
3. Freezes the backbone.
4. Trains the new head using the support set.
5. Evaluates the resulting model on the query set.

This allows us to test whether ordinary supervised pre-training produces useful features for few-shot classification.

## Final revision sheet — Part 2

These are the most important concepts to remember for an oral or written assessment.

| Concept | Essential takeaway |
| --- | --- |
| Baseline | Reference model for comparing meta-learning methods |
| Backbone | Extracts image features |
| Classification head | Converts features into class logits |
| Cross-entropy | Measures classification error |
| Accuracy | Measures correct predictions |
| Top-5 accuracy | True class appears among five highest scores |
| Top-1 accuracy | Highest-scoring class is correct |
| Weight decay | Regularises model parameters |
| Momentum | Uses previous gradients to improve updates |
| Cosine annealing | Adjusts the learning rate over training |
| Frozen backbone | Feature extractor is not updated |
| New classifier head | Adapts to the classes in a new task |

### Five particularly important exam questions

1. Why do we need a baseline before implementing meta-learning?
2. What is the difference between a backbone and a classification head?
3. Why do we use top-5 accuracy for baseline training but top-1 accuracy for five-way tasks?
4. What is the purpose of weight decay and cosine annealing?
5. Why does the baseline freeze the backbone and train a new classification head during few-shot adaptation?

Step 1 for Part 2 is complete. Next, we can convert these exact 20 questions into an interactive practice exam, with answer selection, navigation, scoring and explanations revealed after submission.
