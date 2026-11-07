# Advanced Machine Learning — Practical 6

## Part 5: Training Prototypical Networks (ProtoNet)

MCQ revision set 25 questions

This MCQ set focuses on Part 5 — Task 2: Training ProtoNet, covering the meta-training process, episodic learning, query-set loss, backpropagation, AdamW, learning-rate scheduling, training mode and evaluation mode.

Each question has four possible answers, followed by the correct answer and an explanation.

# Section A — Understanding ProtoNet Training

### Q1. How is ProtoNet trained if it does not have a learnable classification head?

A. By training a fixed linear classification head for every class.

B. By learning the backbone so that its embeddings support accurate prototype-based classification.

C. By updating the class labels after every episode.

D. By removing the backbone and using raw image pixels.

Correct answer: BExplanation: ProtoNet does not require a conventional learned classification head. Instead, it learns a useful embedding space through its trainable backbone. The backbone produces embeddings that allow query images to be classified using support-set prototypes.

### Q2. Which component contains the trainable parameters in ProtoNet?

A. The query labels.

B. The support-set labels.

C. The backbone.

D. The class prototypes themselves as independent parameters.

Correct answer: CExplanation: The backbone contains trainable convolutional parameters. Prototypes are calculated dynamically from support embeddings rather than being stored as independent learned classifier weights.

### Q3. What is the main objective of ProtoNet meta-training?

A. To minimise the classification error on query examples across many sampled tasks.

B. To memorise every support image.

C. To maximise the number of classes in the original dataset.

D. To minimise the number of backbone parameters.

Correct answer: AExplanation: ProtoNet learns representations that allow query images to be classified correctly using prototypes constructed from support examples. It achieves this by minimising query-set cross-entropy loss across episodes.

Here is the corrected version, formatted for GitHub Markdown (GFM) using GitHub-supported LaTeX math syntax. I have replaced `\operatorname{CE}` with `\mathrm{CE}` to avoid the unsupported macro.

Writing

### Q4. Which mathematical expression represents the ProtoNet training objective?

**A.**

$$
L = \mathrm{MSE}(x,y)
$$

**B.**

$$
L = \mathrm{CE}(z_{\text{query}}, y_{\text{query}})
$$

**C.**

$$
L = \mathrm{CE}(z_{\text{support}}, y_{\text{support}})
$$

**D.**

$$
L = \mathrm{Accuracy}(z,y)
$$

**Correct answer: B**

**Explanation:**

The training objective is the cross-entropy loss between the query logits and the corresponding ground-truth query labels:

$$
L_{\text{query}} =
\mathrm{CE}(z_{\text{query}}, y_{\text{query}})
$$

This measures how well the model classifies query examples using the prototypes computed from the support set.

# Section B — Understanding Meta-Training Episodes

### Q5. What happens during one ProtoNet meta-training episode?

A. The model trains on the entire dataset using one fixed classifier.

B. The model samples a task, constructs prototypes, predicts query labels, calculates loss and updates the backbone.

C. The model only evaluates its accuracy without updating parameters.

D. The model replaces the backbone with a new network.

Correct answer: BExplanation: An episode represents one few-shot classification task. ProtoNet uses the support set to construct prototypes, predicts the query examples, calculates query loss and updates its backbone.

### Q6. What is the first step in a typical ProtoNet training episode?

A. Update the optimizer.

B. Calculate query accuracy.

C. Sample a new few-shot task.

D. Load the best checkpoint.

Correct answer: CExplanation: The model first samples a task from the meta-training classes. This task provides the support and query examples needed for the episode.

### Q7. What is the role of the support set during ProtoNet training?

A. It provides labelled examples used to construct class prototypes.

B. It provides the final test accuracy.

C. It updates the optimizer directly.

D. It replaces the query set during training.

Correct answer: AExplanation: The support set defines the classes for the current episode. The backbone embeds the support images, and their mean embeddings become the class prototypes.

### Q8. What is the role of the query set during ProtoNet training?

A. It constructs the class prototypes.

B. It provides additional examples used to calculate classification loss and train the backbone.

C. It determines the number of convolutional layers.

D. It replaces the support labels.

Correct answer: BExplanation: Query examples are classified using the support-set prototypes. Their true labels are used to calculate cross-entropy loss, which provides the learning signal for updating the backbone.

### Q9. Why is the query set important for meta-learning?

A. It allows the model to practise classifying examples beyond those used to construct prototypes.

B. It guarantees 100% training accuracy.

C. It removes the need for labelled support examples.

D. It prevents the model from calculating embeddings.

Correct answer: AExplanation: The query set tests whether the learned embedding space generalises beyond the support examples. Training on query loss encourages representations that work for additional examples of the same classes.

### Q10. Which statement correctly describes the six main operations in one ProtoNet training episode?

A. Sample task → load support/query data → construct prototypes and logits → calculate loss → backpropagate → update parameters.

B. Sample task → calculate accuracy → delete prototypes → update labels → stop training.

C. Load query data → update parameters → construct prototypes → calculate loss → sample task.

D. Load checkpoint → calculate test accuracy → freeze all parameters → stop.

Correct answer: AExplanation: This is the standard episodic training workflow. The model uses support examples to define the task, query examples to calculate the objective, and gradient descent to improve the backbone.

# Section C — Implementing the ProtoNet Training Loop

### Q11. What is the purpose of `proto_opt.zero_grad()`?

A. It resets the model's weights to zero.

B. It clears gradients accumulated from the previous episode.

C. It calculates the query loss.

D. It freezes the backbone.

Correct answer: BExplanation: PyTorch accumulates gradients by default. `zero_grad()` clears the previous gradients before calculating gradients for the current episode.

### Q12. What does the following code perform?

python

```python
query_logits = proto_model(
    (support_x, support_y),
    (query_x, query_y)
)
```

A. It calculates the final test accuracy.

B. It constructs prototypes from the support set and predicts query logits.

C. It updates the optimizer.

D. It saves the model checkpoint.

Correct answer: BExplanation: The forward pass processes the support set to calculate prototypes, processes the query images to obtain embeddings, calculates distances and returns negative-distance logits.

### Q13. Why do we use `F.cross_entropy()` in ProtoNet training?

A. To calculate the distance between support images.

B. To measure the classification error between query logits and true query labels.

C. To calculate the number of prototypes.

D. To update the learning rate.

Correct answer: BExplanation: Cross-entropy measures how well the query predictions match the true labels. It is the training objective used to improve the embedding backbone.

### Q14. What does `train_query_loss.backward()` do?

A. Calculates the final accuracy.

B. Computes gradients of the query loss with respect to trainable model parameters.

C. Updates the optimizer's learning rate.

D. Removes the support-set prototypes.

Correct answer: BExplanation: Backpropagation calculates gradients through the query predictions, query embeddings and support-derived prototypes. These gradients flow back to the backbone parameters.

### Q15. What is the purpose of `proto_opt.step()`?

A. It updates the trainable model parameters using the calculated gradients.

B. It clears all gradients.

C. It calculates class prototypes.

D. It switches the model into evaluation mode.

Correct answer: AExplanation: The optimizer uses the gradients calculated by `backward()` to update the trainable backbone parameters.

### Q16. Which sequence correctly implements the essential ProtoNet optimisation step?

A.

python

```python
proto_opt.step()
train_query_loss.backward()
proto_opt.zero_grad()
```

B.

python

```python
proto_opt.zero_grad()
query_logits = proto_model(...)
train_query_loss = F.cross_entropy(query_logits, query_y)
train_query_loss.backward()
proto_opt.step()
```

C.

python

```python
proto_opt.zero_grad()
proto_opt.step()
query_logits = proto_model(...)
```

D.

python

```python
train_query_loss.backward()
query_logits = proto_model(...)
proto_opt.step()
```

Correct answer: BExplanation: The correct order is to clear old gradients, perform the forward pass, calculate loss, backpropagate and finally update the parameters. This ensures that each episode uses its own gradients.

### Q17. What is the purpose of `top1_acc(query_logits, query_y)`?

A. To calculate the percentage of correctly classified query examples using the highest-scoring class.

B. To calculate the top-five training accuracy.

C. To calculate the distance between prototypes.

D. To update the backbone parameters.

Correct answer: AExplanation: Top-1 accuracy measures the proportion of query examples for which the class with the highest logit matches the true label. It is a performance metric, not the loss function.

# Section D — Optimisation and Hyperparameters

### Q18. Which optimiser does the practical use for ProtoNet?

A. SGD.

B. RMSprop.

C. AdamW.

D. Adagrad.

Correct answer: CExplanation: The notebook uses AdamW, an adaptive gradient-based optimiser with decoupled weight decay.

### Q19. Which code correctly creates the ProtoNet optimiser?

A.

python

```python
proto_opt = torch.optim.AdamW(
    proto_model.parameters(),
    lr=meta_lr,
    weight_decay=meta_l2
)
```

B.

python

```python
proto_opt = torch.optim.SGD(
    query_y,
    lr=meta_lr
)
```

C.

python

```python
proto_opt = torch.optim.AdamW(
    support_y
)
```

D.

python

```python
proto_opt = torch.optim.CrossEntropyLoss()
```

Correct answer: AExplanation: AdamW receives the trainable model parameters, the meta-learning rate and the weight-decay coefficient. This allows it to optimise the ProtoNet backbone.

### Q20. What is the default value of `meta_lr` in the practical?

A. $10^{-1}$

B. $10^{-2}$

C. $10^{-3}$

D. $10^{-5}$

Correct answer: CExplanation: The practical sets:

python

```python
meta_lr = 1e-3
```

This is the initial learning rate used by AdamW.

### Q21. What is the default value of `meta_l2`?

A. $10^{-1}$

B. $10^{-2}$

C. $10^{-3}$

D. $10^{-6}$

Correct answer: CExplanation: The practical uses `meta_l2 = 1e-3`, which is passed to AdamW as its weight-decay parameter.

### Q22. What is the purpose of weight decay in AdamW?

A. To encourage unnecessarily large model weights.

B. To regularise the model by discouraging excessively large parameter values.

C. To increase the number of classes.

D. To replace cross-entropy loss.

Correct answer: BExplanation: Weight decay helps regularise the model by discouraging excessively large parameter values. AdamW applies weight decay separately from its adaptive gradient update.

# Section E — Training Mode, Evaluation Mode and Scheduling

### Q23. What is the purpose of `proto_model.train()`?

A. It switches the model into training mode.

B. It disables all gradients.

C. It saves the best checkpoint.

D. It calculates the final meta-test accuracy.

Correct answer: AExplanation: `train()` activates training-specific behaviour in layers such as BatchNorm and Dropout. It is used when optimising the model.

### Q24. What is the purpose of `model.eval()` together with `torch.no_grad()` during meta-evaluation?

A. To update model parameters faster.

B. To disable training-specific behaviour and avoid building a gradient computation graph during evaluation.

C. To calculate gradients on the query set.

D. To create new trainable prototypes.

Correct answer: BExplanation: `eval()` switches the model to evaluation mode, while `torch.no_grad()` prevents gradient tracking. This makes evaluation more efficient because no parameter updates are required.

### Q25. What is the purpose of `CosineAnnealingLR` in the ProtoNet training loop?

A. To gradually adjust the learning rate according to a cosine schedule.

B. To calculate Euclidean distances.

C. To create new class prototypes.

D. To increase the number of episodes automatically.

Correct answer: AExplanation: Cosine annealing gradually changes the learning rate according to a cosine curve. This can help stabilise optimisation as training progresses.

The notebook implements it as:

python

```python
proto_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    proto_opt,
    max_episodes
)
```

The scheduler is optional and is used when annealing is enabled.

# Final Exam Revision — Part 5

These are the key concepts to remember for an oral or written assessment.

| Concept | What you must remember |
| --- | --- |
| Meta-training | Learns across many sampled few-shot tasks. |
| Episode | One sampled classification task. |
| Support set | Constructs class prototypes. |
| Query set | Provides classification loss and accuracy. |
| Cross-entropy | Measures query classification error. |
| `zero_grad()` | Clears previous gradients. |
| `backward()` | Calculates gradients. |
| `step()` | Updates backbone parameters. |
| AdamW | Optimiser used in the practical. |
| `train()` | Activates training mode. |
| `eval()` | Activates evaluation mode. |
| `no_grad()` | Disables gradient tracking. |
| CosineAnnealingLR | Adjusts the learning rate during training. |

## The most important training workflow

$$
\boxed{
\begin{aligned}
&\text{Sample episode}\\
&\downarrow\\
&\text{Load support and query sets}\\
&\downarrow\\
&\text{Construct prototypes}\\
&\downarrow\\
&\text{Predict query logits}\\
&\downarrow\\
&\text{Calculate query cross-entropy loss}\\
&\downarrow\\
&\text{Backpropagation}\\
&\downarrow\\
&\text{Update backbone using AdamW}
\end{aligned}
}
$$

Exam tip: The most important distinction in Part 5 is that the support set constructs the classifier, while the query set supplies the loss used to train the backbone. This is the central principle of episodic ProtoNet meta-training.
