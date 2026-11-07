# Advanced Machine Learning — Practical 6

## Part 7: Common Problems and How to Solve Them

MCQ Practice Questions with Answers and Explanations

This section focuses on Q34–Q37 of Practical 6. It tests your understanding of common implementation problems in Prototypical Networks (ProtoNet), debugging tensor shapes, diagnosing training issues, improving performance, and identifying notebook implementation errors.

25 MCQsAnswers includedExplanations included

## Section A — Tensor shapes and dimensions

### Question 1. What is the expected shape of the support images in a five-way, five-shot classification task, assuming RGB images of size 32 × 32?

A. `(5, 3, 32, 32)`

B. `(25, 3, 32, 32)`

C. `(25, 64)`

D. `(5, 64)`

Correct answer: BExplanation: Five-way means five classes, and five-shot means five support images per class. Therefore, the total number of support images is:

$$
5\times5=25
$$

Each RGB image has three channels and dimensions 32 × 32. The resulting tensor shape is `(25, 3, 32, 32)`.

### Question 2. If the backbone produces 64-dimensional embeddings, what is the expected shape of the support embeddings for 25 support images?

A. `(25, 64)`

B. `(5, 64)`

C. `(25, 5)`

D. `(64, 25, 32)`

Correct answer: AExplanation: The backbone produces one 64-dimensional embedding per support image. Therefore:

$$
\text{Shape}=(25,64)
$$

The first dimension represents the number of support images, and the second represents the embedding dimension.

### Question 3. What is the expected shape of the prototypes for a five-way classification task with a 64-dimensional embedding space?

A. `(25, 64)`

B. `(5, 25)`

C. `(5, 64)`

D. `(64, 75)`

Correct answer: CExplanation: ProtoNet calculates one prototype per class. Each prototype is a 64-dimensional mean embedding.

Therefore:

$$
\text{Prototype shape}=(5,64)
$$

There are five prototypes, each representing one class.

### Question 4. If there are 75 query images and five class prototypes, what is the expected shape of the distance matrix?

A. `(5, 75)`

B. `(75, 5)`

C. `(75, 64)`

D. `(5, 64)`

Correct answer: BExplanation: Each query image must be compared with all five prototypes.

Using:

python

```python
distances = torch.cdist(
    query_embeddings,
    prototypes,
    p=2
)
```

The output shape is:

$$
(75,5)
$$

Each row represents one query image, and each column represents its distance to one class prototype.

### Question 5. What is the expected shape of the query logits for 75 query images in a five-way classification task?

A. `(75, 5)`

B. `(5, 75)`

C. `(75, 64)`

D. `(25, 5)`

Correct answer: AExplanation: Every query image receives one logit for each of the five classes.

Therefore:

$$
\text{Logits shape}=(75,5)
$$

This is the appropriate format for passing the predictions to PyTorch's cross-entropy loss.

## Section B — Debugging ProtoNet implementation

### Question 6. Why must the query logits and query labels have compatible dimensions when calculating cross-entropy loss?

A. Because cross-entropy requires the backbone to have 64 dimensions.

B. Because each query prediction must correspond to one true class label.

C. Because query labels must contain the Euclidean distances.

D. Because the support and query sets must have identical sizes.

Correct answer: BExplanation: For multiclass classification, PyTorch expects logits with shape `(number_of_queries, number_of_classes)` and target labels with shape `(number_of_queries,)`.

Each query must have exactly one corresponding ground-truth class label.

### Question 7. What problem can occur if support labels and query labels use inconsistent class indices?

A. The model automatically corrects the labels.

B. The model may classify queries against the wrong prototype classes or produce incorrect loss calculations.

C. The embedding dimension automatically increases.

D. The optimiser stops calculating gradients permanently.

Correct answer: BExplanation: ProtoNet assumes that prototype ordering corresponds to the class indices used by the query labels.

For example, if prototype index 0 represents a dog but query label 0 represents a ship, the loss calculation becomes incorrect.

Task-local labels should therefore be consistently mapped to consecutive indices such as `0, 1, 2, 3, 4`.

### Question 8. What happens if one class has no examples in the support set?

A. ProtoNet automatically creates a prototype using query images.

B. The model calculates a zero prototype automatically.

C. The model cannot calculate a valid mean prototype for that class.

D. The model automatically removes all other classes.

Correct answer: CExplanation: A prototype is calculated by averaging support embeddings belonging to a class.

If no support examples exist for that class, there is nothing to average. Every class in the task must have at least one labelled support example.

### Question 9. What should you check first if ProtoNet produces an unexpected output shape?

A. Only the learning-rate scheduler.

B. The support embeddings, prototypes, query embeddings, and query logits.

C. Only the number of training episodes.

D. The filename of the checkpoint.

Correct answer: BExplanation: Shape errors often originate from an incorrect backbone output, prototype construction, or distance calculation.

Checking the dimensions at each stage helps identify exactly where the problem occurs.

### Question 10. What is the purpose of the following operation?

python

```python
support_embeddings[support_y == cls]
```

A. Selects all query embeddings.

B. Selects support embeddings belonging to a particular class.

C. Calculates Euclidean distances.

D. Converts embeddings into probabilities.

Correct answer: BExplanation: The Boolean condition `support_y == cls` selects the support examples whose labels match the specified class.

These selected embeddings are then averaged to calculate the class prototype.

## Section C — Diagnosing training problems

### Question 11. What should you investigate if the ProtoNet training loss does not decrease?

A. Whether `zero_grad()`, `backward()`, and `step()` are correctly executed.

B. Whether the model has a fixed classification head.

C. Whether the query set is removed.

D. Whether the checkpoint filename is sufficiently long.

Correct answer: AExplanation: These three operations are essential for gradient-based optimisation:

- `zero_grad()` clears previous gradients.
- `backward()` calculates gradients.
- `step()` updates the model parameters.

If any operation is missing or incorrectly used, learning may fail.

### Question 12. What can happen if the learning rate is too high?

A. Training always becomes faster and more accurate.

B. The model may make excessively large updates, causing unstable training.

C. The model automatically increases its number of classes.

D. The query set becomes identical to the support set.

Correct answer: BExplanation: An excessively high learning rate can cause optimisation to overshoot useful parameter values.

This may result in fluctuating loss, unstable training, or failure to converge.

### Question 13. What can happen if the learning rate is too low?

A. Training may progress very slowly.

B. The model automatically overfits immediately.

C. The embedding dimension becomes zero.

D. The model stops recognising support labels.

Correct answer: AExplanation: A very small learning rate produces small parameter updates.

Although training may remain stable, it may take considerably longer to reach a useful solution.

### Question 14. Why should you print tensor shapes during the first few training episodes?

A. To increase model accuracy directly.

B. To verify that the tensors have the expected dimensions before running extensive training.

C. To change the optimiser automatically.

D. To avoid calculating cross-entropy loss.

Correct answer: BExplanation: Printing shapes helps detect implementation problems early.

For example, an unexpected prototype shape or incorrectly formatted query logits can be identified before launching thousands of episodes.

### Question 15. What should you check if the ProtoNet backbone does not receive gradients?

A. Whether the query loss is connected to the model's computational graph.

B. Whether the checkpoint filename is correct.

C. Whether the number of classes is five.

D. Whether the query set contains RGB images.

Correct answer: AExplanation: The query loss must be differentiable with respect to the backbone parameters.

Operations such as detaching embeddings or calculating predictions inside `torch.no_grad()` during training can prevent gradients from reaching the backbone.

### Question 16. Why might meta-training loss decrease while meta-test accuracy remains poor?

A. Training loss and meta-test accuracy measure exactly the same thing.

B. The model may be fitting the meta-training task distribution without generalising effectively to unseen tasks.

C. Meta-test accuracy is unrelated to few-shot learning.

D. Cross-entropy cannot be used with ProtoNet.

Correct answer: BExplanation: Training loss measures performance on meta-training query examples. Meta-test accuracy measures generalisation to tasks involving held-out classes.

A decreasing training loss does not guarantee good performance on unseen tasks.

## Section D — Improving ProtoNet performance

### Question 17. Why might initialising ProtoNet with a pre-trained backbone improve performance?

A. It guarantees 100% meta-test accuracy.

B. It provides useful visual features learned from previous supervised training.

C. It eliminates the need for support examples.

D. It removes the need for prototype calculation.

Correct answer: BExplanation: A pre-trained backbone may already recognise useful visual patterns such as edges, textures, shapes, and object parts.

This can provide a better starting point for learning an embedding space suitable for few-shot classification.

However, improvement is not guaranteed and should be measured experimentally.

### Question 18. What is the purpose of experimenting with different temperature values?

A. To change the number of support images.

B. To control the scale and sharpness of the classification logits.

C. To change the embedding dimension.

D. To change the number of meta-test classes.

Correct answer: BExplanation: ProtoNet calculates logits using:

$$
z_k=-\frac{d_k}{T}
$$

A larger temperature generally produces softer probability distributions, whereas a smaller positive temperature makes the logits more extreme.

The nearest prototype remains unchanged for positive temperatures.

### Question 19. What is the purpose of increasing the number of meta-training episodes?

A. To expose the model to more sampled few-shot tasks.

B. To increase the number of classes in the dataset automatically.

C. To eliminate the query set.

D. To change ProtoNet into Reptile.

Correct answer: AExplanation: Each episode provides another sampled task.

Increasing the number of episodes can give the backbone more opportunities to learn representations that generalise across tasks, although additional episodes do not guarantee improved performance.

### Question 20. Why might experimenting with weight decay improve ProtoNet performance?

A. It increases the number of prototypes.

B. It changes the task from five-way to ten-way classification.

C. It controls regularisation and can help reduce overfitting.

D. It automatically fixes incorrect labels.

Correct answer: CExplanation: Weight decay discourages excessively large parameter values.

Testing different values can help determine whether the model benefits from stronger or weaker regularisation.

### Question 21. Which experimental approach is most appropriate when ProtoNet performs poorly?

A. Change all hyperparameters simultaneously.

B. Increase the learning rate without checking the loss.

C. Experiment with one parameter or design choice at a time and compare meta-test performance.

D. Remove the support set.

Correct answer: CExplanation: Changing one variable at a time makes it easier to understand which change affects performance.

Useful experiments include different initialisations, learning rates, temperatures, episode counts, and weight-decay values.

## Section E — Notebook implementation issues

### Question 22. What is the potential problem with the following code?

python

```python
proto_model = ProtoNet(
    backbone=meta_backbone
)
```

A. ProtoNet cannot accept a backbone.

B. `meta_backbone` may not have been defined or may not be the intended feature extractor.

C. ProtoNet requires a classification head.

D. The temperature must always be zero.

Correct answer: BExplanation: The variable `meta_backbone` must exist before it is passed to the constructor.

If it has not been defined, Python raises a `NameError`. If it exists but contains the wrong model, the resulting implementation may not work as intended.

### Question 23. In the Reptile section, what potential error occurs if the final evaluation uses `meta_model` when the actual model is named `reptile_backbone`?

A. The model automatically changes its name.

B. Python may raise a `NameError` because `meta_model` is undefined.

C. ProtoNet calculates incorrect distances.

D. The query labels become invalid.

Correct answer: BExplanation: Python requires variables to be defined before they are used.

If `meta_model` has not been assigned, the evaluation call fails. The intended model variable should be passed instead:

python

```python
final_meta_results = meta_evaluate(
    reptile_backbone,
    ...
)
```

### Question 24. Why is checkpoint consistency important?

A. The saved checkpoint and loaded model must have compatible architectures and state dictionaries.

B. Every checkpoint must contain the entire image dataset.

C. Checkpoints must always be loaded onto a CPU.

D. The model must use a different backbone every time it is loaded.

Correct answer: AExplanation: A state dictionary contains model parameters and buffers associated with a particular architecture.

If the model architecture changes, the checkpoint may not load correctly because parameter names or tensor dimensions may differ.

### Question 25. Why should meta-test tasks ideally be reserved for final evaluation?

A. Because meta-test tasks cannot contain query examples.

B. Because repeatedly using them to select hyperparameters or checkpoints can cause information leakage and overly optimistic results.

C. Because meta-validation tasks are always easier.

D. Because ProtoNet cannot evaluate unseen classes.

Correct answer: BExplanation: Meta-test data should provide an independent measurement of generalisation.

If we repeatedly inspect meta-test results and use them to make development decisions, we indirectly tune the model to the test set.

A better experimental design is:

- Meta-training: Learn the model.
- Meta-validation: Select hyperparameters and checkpoints.
- Meta-testing: Perform the final evaluation.

# Final revision summary

These are the key concepts to remember for Part 7.

| Topic | Key point |
| --- | --- |
| Tensor shapes | Check support, embeddings, prototypes and query logits. |
| Label encoding | Prototype ordering must match query label indices. |
| Missing prototypes | Every class needs support examples. |
| Training loss | Check gradients, optimiser updates and learning rate. |
| Poor generalisation | Training loss and meta-test accuracy are different metrics. |
| Pre-trained backbone | Can provide useful initial visual representations. |
| Temperature | Controls the scale of ProtoNet logits. |
| Weight decay | Provides regularisation. |
| Checkpoints | Save and load compatible model state dictionaries. |
| Data leakage | Reserve meta-test tasks for final evaluation. |

## Most important exam takeaway

Part 7 is about debugging and improving few-shot learning implementations. You should be able to identify incorrect tensor shapes, inconsistent labels, missing gradients, unstable learning rates, poor generalisation, and notebook variable or checkpoint errors.

The most important practical debugging habit is to verify tensor dimensions and label consistency before running thousands of meta-training episodes.

When you're ready, we can follow the same workflow as Parts 1–6:

Step 2: Convert these 25 MCQs into a fully interactive practice exam, with one question at a time, instant feedback, explanations, automatic scoring, answer review, and a retake option.
