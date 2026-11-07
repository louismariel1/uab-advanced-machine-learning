# Advanced Machine Learning — Practical 6

## Part 6: Meta-Evaluation and Comparison

25 MCQsAnswers includedExplanations included

This question-focused revision covers Part 6 — Meta-Evaluation and Comparison (Q30–Q33) of Practical 6.

The questions are designed to prepare you for a lab assessment, written examination, or oral exam. They test conceptual understanding, mathematical interpretation, code comprehension, and critical evaluation of experimental results.

## Section A — Evaluating ProtoNet on unseen tasks

### Q1. What is the main purpose of `meta_evaluate_protonet()`?

A. To train the ProtoNet backbone using the meta-test tasks.

B. To evaluate ProtoNet's ability to classify query examples from unseen tasks without updating its parameters.

C. To fine-tune a new classification head for every meta-test task.

D. To calculate training accuracy on the original meta-training dataset.

Correct answer: BExplanation: `meta_evaluate_protonet()` measures how well ProtoNet generalises to unseen tasks. It constructs prototypes from support examples and predicts query examples without performing gradient updates.

### Q2. What is the first operation performed when evaluating a ProtoNet task?

A. Update the backbone using query loss.

B. Load the best checkpoint.

C. Obtain the support examples and their labels.

D. Calculate the final average accuracy.

Correct answer: CExplanation: The support examples are needed to construct class prototypes. These prototypes are subsequently used to classify the query examples.

### Q3. During ProtoNet meta-evaluation, what is the purpose of the support set?

A. To calculate the final average query accuracy.

B. To construct class prototypes.

C. To update the AdamW optimiser.

D. To calculate gradients for the backbone.

Correct answer: BExplanation: ProtoNet calculates the mean embedding of the support examples belonging to each class. These mean embeddings become the class prototypes.

### Q4. What is the purpose of the query set during meta-evaluation?

A. To construct the prototypes.

B. To train a new classification head.

C. To evaluate classification performance using additional examples.

D. To initialise the backbone.

Correct answer: CExplanation: The query set contains examples that the model must classify using the prototypes constructed from the support set. Its labels are used to calculate query loss and accuracy.

### Q5. Which sequence correctly describes ProtoNet meta-evaluation?

A. Fine-tune backbone → calculate prototypes → update weights → calculate accuracy.

B. Construct prototypes → predict query logits → calculate query loss and accuracy.

C. Calculate query loss → update weights → construct prototypes.

D. Train a classification head → calculate training accuracy → save checkpoint.

Correct answer: BExplanation: ProtoNet evaluation involves constructing prototypes, predicting query logits and measuring performance. It does not update the backbone during evaluation.

## Section B — Understanding evaluation mode and gradients

### Q6. Why does the evaluation function use `model.eval()`?

A. To enable gradient updates.

B. To switch the model into evaluation mode.

C. To reset the model parameters.

D. To create new class prototypes.

Correct answer: BExplanation: `model.eval()` changes the behaviour of layers such as BatchNorm and Dropout to their evaluation settings. It does not automatically disable gradient calculations.

### Q7. What is the purpose of `torch.no_grad()` during meta-evaluation?

A. To calculate gradients more accurately.

B. To update the backbone without an optimiser.

C. To prevent gradient computation and reduce memory consumption.

D. To increase the number of query examples.

Correct answer: CExplanation: `torch.no_grad()` disables gradient tracking for operations inside its context. Since evaluation does not require backpropagation, this saves memory and computational resources.

### Q8. Which code correctly performs ProtoNet inference during meta-evaluation?

A.

python

```python
model.train()

query_logits = model(
    (support_x, support_y),
    (query_x, query_y)
)

query_loss.backward()
```

B.

python

```python
model.eval()

with torch.no_grad():
    query_logits = model(
        (support_x, support_y),
        (query_x, query_y)
    )
```

C.

python

```python
model.eval()

query_logits.backward()
```

D.

python

```python
model.zero_grad()
optimizer.step()
```

Correct answer: BExplanation: The model is placed in evaluation mode, and `torch.no_grad()` prevents unnecessary gradient tracking. The forward pass produces query logits without updating model parameters.

### Q9. Does `model.eval()` automatically disable gradient calculations?

A. Yes, always.

B. No, `model.eval()` changes the behaviour of certain layers, whereas `torch.no_grad()` disables gradient tracking.

C. Yes, but only for convolutional layers.

D. No, because evaluation mode is only used for calculating accuracy.

Correct answer: BExplanation: This is an important distinction in PyTorch. `eval()` and `no_grad()` serve different purposes. They are commonly used together during model evaluation.

### Q10. Why does ProtoNet not need to fine-tune its backbone on each meta-test task?

A. Because ProtoNet ignores the support set.

B. Because the backbone has no trainable parameters.

C. Because the support examples directly construct task-specific prototypes in the learned embedding space.

D. Because query labels are used to update the model.

Correct answer: CExplanation: ProtoNet adapts its classifier by calculating new prototypes from the support examples. It does not need gradient-based fine-tuning of the backbone during meta-testing.

## Section C — Understanding query loss and accuracy

### Q11. Which loss function is used to evaluate ProtoNet's query predictions?

A. Mean squared error.

B. Binary cross-entropy.

C. Cross-entropy loss.

D. Hinge loss.

Correct answer: CExplanation: The practical uses `F.cross_entropy(query_logits, query_y)` to measure how well the model classifies the query examples.

### Q12. What does the following code calculate?

python

```python
F.cross_entropy(query_logits, query_y).item()
```

A. The number of correctly classified images.

B. The query-set classification loss as a Python numerical value.

C. The distance between class prototypes.

D. The learning rate of the model.

Correct answer: BExplanation: Cross-entropy measures the discrepancy between predicted query logits and true query labels. `.item()` extracts the scalar value from the tensor.

### Q13. What does `top1_acc(query_logits, query_y)` measure?

A. The percentage of query images classified correctly using the highest-scoring predicted class.

B. The percentage of images classified correctly among the five highest-scoring classes.

C. The average Euclidean distance between prototypes.

D. The percentage of support images used during training.

Correct answer: AExplanation: Top-1 accuracy checks whether the class with the highest predicted logit matches the true query label.

### Q14. Why does the practical average query accuracy across multiple meta-test tasks?

A. To measure performance across different sampled tasks rather than relying on one task.

B. To increase the number of training episodes.

C. To calculate the learning rate.

D. To eliminate the need for query labels.

Correct answer: AExplanation: Individual tasks may vary in difficulty. Averaging their query accuracies provides a more representative estimate of generalisation performance.

### Q15. Suppose ProtoNet is evaluated on 10 meta-test tasks with the following query accuracies:

50%, 60%, 70%, 80%, 40%, 50%, 60%, 70%, 80%, 40%.

What is the average query accuracy?

A. 50%.

B. 55%.

C. 60%.

D. 65%.

Correct answer: CExplanation:

$$
\overline A=\frac{1}{N}\sum_{i=1}^{N}A_i
$$

$$
=\frac{50+60+70+80+40+50+60+70+80+40}{10}
$$

$$
=\frac{600}{10}=60\%
$$

The average query accuracy is 60%.

## Section D — Checkpoint selection and model saving

### Q16. What is the purpose of saving a ProtoNet checkpoint when its meta-test query accuracy improves?

A. To save the model that has achieved the best recorded evaluation performance.

B. To save the model with the highest training loss.

C. To reset the optimiser.

D. To increase the number of meta-training classes.

Correct answer: AExplanation: Checkpointing preserves the model parameters associated with the best recorded evaluation accuracy, allowing the experiment to recover that model later.

### Q17. What does this condition mean?

python

```python
if test_query_acc > current_best_acc:
```

A. Save the model if the current accuracy is lower than the previous best.

B. Save the model if the current accuracy is higher than the previous best.

C. Save the model after every training episode.

D. Stop training when the accuracy improves.

Correct answer: BExplanation: The condition checks whether the current evaluation accuracy exceeds the best previously recorded accuracy. If it does, the model checkpoint is saved.

### Q18. What is the purpose of `torch.save()` in the following code?

python

```python
torch.save(
    proto_model.state_dict(),
    "best_protonet.ckpt"
)
```

A. To save the entire dataset.

B. To save the model's parameter state dictionary to a checkpoint file.

C. To calculate model accuracy.

D. To initialise a new backbone.

Correct answer: BExplanation: `state_dict()` contains the model's parameters and registered buffers. `torch.save()` writes this state dictionary to the specified checkpoint file.

### Q19. What does the following code do?

python

```python
proto_model.load_state_dict(
    torch.load(
        "best_protonet.ckpt",
        map_location=device,
        weights_only=True
    )
)
```

A. It trains ProtoNet from scratch.

B. It loads the saved checkpoint parameters into the ProtoNet model.

C. It creates new prototypes.

D. It evaluates the model on the support set.

Correct answer: BExplanation: `torch.load()` reads the checkpoint, and `load_state_dict()` restores the saved parameters into the existing model architecture.

### Q20. What is the purpose of `map_location=device`?

A. To calculate the model's location in the embedding space.

B. To ensure loaded checkpoint tensors are mapped to the specified device.

C. To select the best-performing query examples.

D. To change the model architecture.

Correct answer: BExplanation: A checkpoint may have been saved on a different device. `map_location=device` allows its tensors to be loaded onto the intended device, such as a CPU or GPU.

### Q21. Why is `weights_only=True` useful when loading a state dictionary?

A. It loads only the query examples.

B. It restricts loading to the supported weights-only format rather than unrestricted object deserialisation.

C. It removes all model parameters.

D. It automatically calculates the best accuracy.

Correct answer: BExplanation: In modern PyTorch, `weights_only=True` is a safer loading option for state dictionaries because it restricts the deserialisation process compared with unrestricted pickle-based loading.

## Section E — Critical evaluation and experimental design

### Q22. What is a potential problem with repeatedly using meta-test tasks to select the best checkpoint?

A. The model will lose its backbone.

B. It introduces information leakage because the meta-test results influence model selection.

C. It prevents prototype construction.

D. It makes cross-entropy mathematically invalid.

Correct answer: BExplanation: Repeatedly choosing checkpoints based on meta-test performance means information from the test set influences model development. This can produce an overly optimistic estimate of generalisation.

### Q23. What is the recommended experimental practice for checkpoint selection?

A. Use the training set for final reporting.

B. Use meta-validation tasks for checkpoint selection and reserve meta-test tasks for final evaluation.

C. Use the meta-test tasks for every hyperparameter experiment.

D. Avoid evaluating the model altogether.

Correct answer: BExplanation: Meta-validation supports model selection and hyperparameter tuning. Meta-testing should provide an independent assessment of performance on unseen tasks.

### Q24. Which statement correctly describes the difference between Reptile and ProtoNet during meta-testing?

A. Both methods always fine-tune the backbone using the support set.

B. Reptile adapts model parameters through gradient-based fine-tuning, whereas ProtoNet constructs prototypes from support embeddings without gradient-based backbone adaptation.

C. ProtoNet fine-tunes its backbone, whereas Reptile never uses gradients.

D. Neither method uses support examples.

Correct answer: BExplanation: Reptile uses the support set to fine-tune model parameters. ProtoNet uses the support set to calculate class prototypes. Their adaptation mechanisms are fundamentally different.

### Q25. Why is average meta-test query accuracy the most important comparison metric in this practical?

A. It measures how quickly the model processes images.

B. It measures how well each method classifies new examples from held-out classes across few-shot tasks.

C. It measures the number of trainable parameters.

D. It measures the accuracy of the original meta-training classifier only.

Correct answer: BExplanation: The main goal of few-shot learning is to recognise unfamiliar classes from limited labelled examples. Average meta-test query accuracy directly measures this ability across multiple unseen tasks.

# Final revision summary

The following concepts are particularly important for your examination.

| Concept | Essential knowledge |
| --- | --- |
| `meta_evaluate_protonet()` | Evaluates ProtoNet on unseen tasks. |
| Support set | Constructs class prototypes. |
| Query set | Measures classification performance. |
| `model.eval()` | Switches to evaluation behaviour. |
| `torch.no_grad()` | Disables gradient tracking. |
| Cross-entropy | Measures query classification loss. |
| Top-1 accuracy | Measures correctly classified query examples. |
| Average query accuracy | Averages performance across meta-test tasks. |
| `torch.save()` | Saves model checkpoint parameters. |
| `load_state_dict()` | Restores saved parameters. |
| Meta-validation | Used for model selection and hyperparameter tuning. |
| Meta-testing | Used for final generalisation evaluation. |

## Five particularly important oral-exam questions

1. Why does ProtoNet not need gradient-based fine-tuning during meta-testing?
2. What is the difference between `model.eval()` and `torch.no_grad()`?
3. Why do we average query accuracy across multiple meta-test tasks?
4. Why should we avoid repeatedly using meta-test results to select checkpoints?
5. What is the fundamental difference between Reptile and ProtoNet when adapting to a new task?

Key takeaway: Part 6 is about evaluating whether a meta-learning model can generalise to genuinely unseen classes. ProtoNet constructs prototypes from support examples, evaluates predictions on query examples, and reports average query accuracy. Fair experimental evaluation also requires keeping meta-validation and meta-testing separate.

When you're ready, we can follow the same workflow as Parts 1–5:

Step 2: Convert these 25 MCQs into a fully interactive practice exam, with question navigation, answer selection, automatic marking, explanations, scoring and the ability to retake the entire exam.
