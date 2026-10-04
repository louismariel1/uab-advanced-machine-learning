 # Comprehensive MCQ — Semi-Supervised Learning with UDA

 ## Part I — Big Picture: What Problem Are We Solving?

 ### 1\. What is the main problem addressed by this practical?

 A. Training a model when there are no images available\
 B. Improving classification when only a small portion of the training data is labelled\
 C. Replacing CNNs with transformers\
 D. Reducing the number of classes in the dataset

 **Answer: B**

 **Explanation:**\
 The practical addresses **semi-supervised learning (SSL)**. We have:

 - a relatively small labelled dataset $D_L$
- a much larger unlabelled dataset $D_U$

 Instead of ignoring the unlabelled examples, we use them to improve the classifier.

---

 ### 2\. Why is labelled data often the bottleneck in machine learning?

 A. Neural networks cannot process unlabelled images\
 B. Labelling data can require expensive human annotation\
 C. Labelled images always have lower resolution\
 D. Labels consume GPU memory

 **Answer: B**

 **Explanation:**\
 Obtaining images can be relatively easy, while assigning correct labels often requires human experts, time, and money. Semi-supervised learning attempts to exploit large amounts of **cheap unlabelled data**.

---

 ### 3\. Which setup best describes semi-supervised learning?

 A. 100% labelled training data\
 B. 100% unlabelled training data\
 C. A mixture of labelled and unlabelled training data\
 D. Training exclusively on validation data

 **Answer: C**

---

 ### 4\. What is the key intuition behind consistency regularisation?

 A. Different augmentations of the same image should receive different predictions\
 B. The model should make similar predictions for different versions of the same input\
 C. The model should ignore labelled examples\
 D. Augmentation should change the class of an image

 **Answer: B**

 **Explanation:**\
 If an image is, for example, a cat, then a slightly rotated, cropped, translated, or otherwise augmented version should still be classified as a cat.

 The objective is therefore:

 $$
f(x) \approx f(\text{augmentation}(x))
$$

---

 ### 5\. What does UDA stand for in this practical?

 A. Unlabelled Data Analysis\
 B. Universal Deep Architecture\
 C. Unsupervised Data Augmentation\
 D. Unified Dataset Algorithm

 **Answer: C**

 **Explanation:**\
 UDA stands for **Unsupervised Data Augmentation**. The central idea is to use predictions on clean/unlabelled examples as a consistency target for predictions on augmented versions.

---

 # Part II — Supervised vs Semi-Supervised Learning

 ### 6\. In ordinary supervised classification, what information is available?

 A. Inputs only\
 B. Labels only\
 C. Input-label pairs $(x,y)$\
 D. Predictions only

 **Answer: C**

---

 ### 7\. What loss is normally used for a multi-class classification problem in this notebook?

 A. Mean squared error\
 B. Cross entropy\
 C. Binary cross entropy only\
 D. Hinge loss

 **Answer: B**

 The code is:

```
loss_func = nn.CrossEntropyLoss()
```

---

 ### 8\. What does the supervised loss encourage?

 A. Similar predictions for augmented and clean images\
 B. Predictions that match the known labels\
 C. Larger model weights\
 D. More data augmentation

 **Answer: B**

 Mathematically:

 $$
L_s = CE(f(x),y)
$$

 where $y$ is the known ground-truth label.

---

 ### 9\. What is special about the unlabelled dataset?

 A. It contains images but no ground-truth class labels\
 B. It contains only validation examples\
 C. It contains only incorrectly labelled images\
 D. It contains numerical labels instead of categorical labels

 **Answer: A**

---

 ### 10\. Why can't we simply calculate ordinary cross-entropy on the unlabelled examples?

 A. Cross entropy requires a target label\
 B. CNNs cannot process unlabelled images\
 C. Cross entropy only works for regression\
 D. PyTorch does not support unlabelled data

 **Answer: A**

 **Explanation:**\
 For an unlabelled image $x$, we don't know its true $y$. UDA gets around this by constructing a **consistency target** from the model's own prediction.

---

 # Part III — Consistency Regularisation

 ### 11\. Suppose $x$ is an unlabelled image. What are $x_{\text{clean}}$ and $x_{\text{aug}}$?

 A. Two unrelated images\
 B. The same image in clean and augmented forms\
 C. Two labelled images\
 D. Training and validation versions of different images

 **Answer: B**

---

 ### 12\. What should ideally happen when the model processes these two versions?

 A. Their predictions should be similar\
 B. Their predictions should be completely different\
 C. Only the augmented version should produce a prediction\
 D. Both should produce random predictions

 **Answer: A**

---

 ### 13\. What does the consistency loss attempt to minimise?

 A. Difference between model parameters\
 B. Difference between predictions for clean and augmented inputs\
 C. Difference between training and validation datasets\
 D. Difference between CNN layers

 **Answer: B**

---

 ### 14\. Which expression best represents the overall UDA objective?

 A.

 $$
L=L_s-L_u
$$

 B.

 $$
L=L_s+\lambda_uL_u
$$

 C.

 $$
L=L_s/\lambda_u
$$

 D.

 $$
L=L_u-L_s
$$

 **Answer: B**

 This is one of the **most important equations in the practical**.

---

 ### 15\. What does $\lambda_u$ control?

 A. Learning rate\
 B. Number of classes\
 C. Weight of the unsupervised loss\
 D. Batch size

 **Answer: C**

---

 ### 16\. If $\lambda_u=0$, what happens?

 A. Only the unsupervised loss is used\
 B. Training becomes supervised-only\
 C. Training stops\
 D. The model has no parameters

 **Answer: B**

 Because:

 $$
L=L_s+0L_u=L_s
$$

---

 ### 17\. What is a possible problem if $\lambda_u$ is excessively large?

 A. The model may overemphasise a noisy unsupervised signal\
 B. The supervised loss becomes mathematically impossible\
 C. The dataset disappears\
 D. The optimizer automatically stops

 **Answer: A**

---

 # Part IV — KL Divergence

 ### 18\. Why is KL divergence appropriate for consistency regularisation?

 A. We want to compare probability distributions produced by the model\
 B. We are solving a regression problem\
 C. KL divergence increases disagreement\
 D. KL divergence removes labels

 **Answer: A**

---

 ### 19\. If $p$ is the clean prediction and $q$ is the augmented prediction, what do we want?

 $$
KL(p\|q)
$$

 to be:

 A. Large\
 B. Negative\
 C. Small\
 D. Infinite

 **Answer: C**

---

 ### 20\. Why does the direction of KL divergence matter?

 A. KL divergence is symmetric\
 B. KL divergence is generally not symmetric\
 C. KL divergence cannot compare probabilities\
 D. KL divergence is always zero

 **Answer: B**

 In general:

 $$
KL(P\|Q)\neq KL(Q\|P)
$$

 So the implementation must use the intended prediction as the target/reference distribution.

---

 ### 21\. What is a crucial implementation detail for the clean prediction?

 A. It should usually be detached from the computation graph when acting as the target\
 B. It should be converted to an integer label\
 C. It should be deleted\
 D. It should be randomly shuffled

 **Answer: A**

---

 ### 22\. Why detach the clean target?

 A. To prevent gradients from flowing through the target distribution\
 B. To make the image smaller\
 C. To increase the number of classes\
 D. To disable the optimizer

 **Answer: A**

 Conceptually:

```
clean_logits.detach()
```

 means the clean prediction provides a target without allowing the consistency loss to directly update the network through that target branch.

---

 ### 23\. What could happen if you forget to detach the target?

 A. The consistency objective may update both sides instead of treating one as a fixed target\
 B. The model cannot run\
 C. The labels become one-hot automatically\
 D. The CNN becomes a regression network

 **Answer: A**

---

 # Part V — Understanding the UDA Code

 ### 24\. Why is `itertools.cycle(trainU_loader)` used?

 A. To shuffle the labelled dataset\
 B. To make the unlabelled loader effectively infinite\
 C. To reduce GPU memory\
 D. To remove augmentation

 **Answer: B**

 The loaders may contain different numbers of batches.

---

 ### 25\. Why is cycling particularly useful here?

 A. The labelled and unlabelled loaders can have different lengths\
 B. The model requires exactly one unlabelled batch per epoch\
 C. It disables backpropagation\
 D. It increases the number of classes

 **Answer: A**

 The code:

```
trainU_iter = iter(itertools.cycle(trainU_loader))
```

 allows:

```
ux_clean, ux_aug = next(trainU_iter)
```

 to continue indefinitely.

---

 ### 26\. What does this code do?

```
x, y = batch
x, y = x.to(device), y.to(device)
```

 A. Moves labelled inputs and labels to the selected device\
 B. Creates an optimizer\
 C. Performs augmentation\
 D. Calculates KL divergence

 **Answer: A**

---

 ### 27\. Why must `x` and `y` be moved to the same device as the model?

 A. Otherwise tensor/device mismatches can occur\
 B. Otherwise CrossEntropyLoss becomes regression\
 C. Otherwise the model has no labels\
 D. It is only for visualization

 **Answer: A**

---

 ### 28\. What does the following calculate?

```
pred = model(x)
```

 A. Predictions for labelled images\
 B. Predictions for validation images\
 C. The optimizer state\
 D. The KL divergence

 **Answer: A**

---

 ### 29\. What does this calculate?

```
supervised_loss = loss_func(pred, y)
```

 A. Consistency loss\
 B. Supervised classification loss\
 C. Validation accuracy\
 D. Confidence

 **Answer: B**

---

 ### 30\. What do these lines calculate?

```
clean_logits = model(ux_clean)
aug_logits = model(ux_aug)
```

 A. Predictions for clean and augmented unlabelled inputs\
 B. Predictions for validation labels\
 C. Two independent supervised losses\
 D. Two optimizers

 **Answer: A**

---

 # Part VI — Logits, Probabilities, and Softmax

 ### 31\. What are logits?

 A. Raw outputs of the final classification layer before softmax\
 B. Ground-truth labels\
 C. Images after augmentation\
 D. Validation scores

 **Answer: A**

---

 ### 32\. Why are logits useful?

 A. They can be passed to losses such as `CrossEntropyLoss` without manually applying softmax\
 B. They are always plus a consistency objective that encourages predictions on clean and augmented versions of unlabelled examples to agree, while experimenting with the weighting, warmup, and confidence probabilities\
 C. They are ground-truth labels\
 D. They contain the input image

 **Answer: A**

---

 ### 33\. What does softmax do?

 A. Converts logits into a probability distribution\
 B. Converts images into labels\
 C. Removes all classes\
 D. Calculates accuracy

 **Answer: A**

 For logits $z_i$:

 $$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

---

 ### 34\. Which statement about logits is correct?

 A. They must sum to 1\
 B. They are always between 0 and 1\
 C. They do not necessarily represent probabilities\
 D. They are always integers

 **Answer: C**

---

 ### 35\. Which line converts clean logits to probabilities?

```
clean_probs = F.softmax(clean_logits.detach(), dim=1)
```

 What does `dim=1` typically mean here?

 A. Batch dimension\
 B. Class dimension\
 C. Image height\
 D. Image width

 **Answer: B**

 For a tensor shaped approximately:

```
[batch_size, number_of_classes]
```

 `dim=1` is the class dimension.

---

 # Part VII — Confidence Filtering

 ### 36\. Why introduce confidence filtering?

 A. Model predictions early in training can be unreliable\
 B. It increases the number of labelled examples\
 C. It removes the supervised loss\
 D. It changes the number of classes

 **Answer: A**

---

 ### 37\. What is the confidence score in this implementation?

```
confidence = clean_probs.max(dim=1).values
```

 A. Average probability\
 B. Minimum class probability\
 C. Probability of the most likely class\
 D. Training accuracy

 **Answer: C**

---

 ### 38\. What does this mask mean?

```
mask = confidence >= confidence_threshold
```

 A. Keep examples whose predicted class has sufficiently high confidence\
 B. Keep examples with low confidence\
 C. Delete labelled examples\
 D. Select validation examples

 **Answer: A**

---

 ### 39\. If the threshold is 0.8, what does confidence filtering mean?

 A. Only predictions with maximum probability at least 0.8 contribute to the unsupervised loss\
 B. Exactly 80% of images are used\
 C. The model must achieve 80% validation accuracy\
 D. 80 classes are selected

 **Answer: A**

---

 ### 40\. What happens if no examples satisfy the threshold?

```
if mask.any():
    ...
else:
    unsupervised_loss = torch.tensor(0.0, device=device)
```

 A. The batch crashes\
 B. The unsupervised loss is set to zero for that batch\
 C. The supervised loss is removed\
 D. The model is reinitialised

 **Answer: B**

---

 ### 41\. Why can confidence filtering be useful?

 A. It can prevent unreliable pseudo-targets from dominating training\
 B. It guarantees perfect predictions\
 C. It eliminates the need for labels\
 D. It always improves accuracy

 **Answer: A**

 Important: it **can** help; it is not guaranteed to improve every experiment.

---

 ### 42\. What is a potential downside of using a very high confidence threshold?

 A. Too few examples may contribute to the unsupervised loss\
 B. Every example becomes labelled\
 C. The learning rate increases\
 D. The model loses its CNN layers

 **Answer: A**

---

 # Part VIII — Warmup

 ### 43\. Why might warmup epochs be useful?

 A. The model's predictions are unreliable at the beginning\
 B. The GPU cannot perform augmentation initially\
 C. Labels do not exist after epoch 1\
 D. Cross entropy only works after five epochs

 **Answer: A**

---

 ### 44\. What does this code do?

```
if e < warmup_epochs:
    current_lambda_u = 0.0
else:
    current_lambda_u = lambda_u
```

 A. Disables the unsupervised loss during warmup\
 B. Disables supervised training\
 C. Freezes the model\
 D. Disables validation

 **Answer: A**

---

 ### 45\. Suppose:

```
warmup_epochs = 3
```

 and epochs are numbered:

```
0, 1, 2, 3, 4, ...
```

 During which epochs is the unsupervised loss disabled?

 A. 0 only\
 B. 0, 1\
 C. 0, 1, 2\
 D. 1, 2, 3

 **Answer: C**

 Because:

```
e < 3
```

 is true for 0, 1, and 2.

---

 ### 46\. What is another possible strategy besides a hard warmup?

 A. Gradually ramp $\lambda_u$ upward\
 B. Delete the validation set\
 C. Remove augmentation\
 D. Randomly change labels

 **Answer: A**

 For example:

 $$
\lambda_u(t)
$$

 could gradually increase over training.

---

 # Part IX — Full Training Loop

 ### 47\. What happens first in a typical training iteration?

 A. Backpropagation\
 B. Load labelled and unlabelled data\
 C. Validation\
 D. Early stopping

 **Answer: B**

---

 ### 48\. Why is `opt.zero_grad()` called?

 A. To clear gradients from the previous iteration\
 B. To delete model parameters\
 C. To zero the loss\
 D. To reset the dataset

 **Answer: A**

---

 ### 49\. What happens during:

```
total_loss.backward()
```

 A. Gradients are computed\
 B. Parameters are immediately deleted\
 C. Validation accuracy is calculated\
 D. The dataset is shuffled

 **Answer: A**

---

 ### 50\. What happens during:

```
opt.step()
```

 A. Model parameters are updated using their gradients\
 B. The validation set is evaluated\
 C. The model is put into evaluation mode\
 D. Labels are generated

 **Answer: A**

---

 ### 51\. What is the complete loss in UDA?

```
total_loss = supervised_loss + current_lambda_u * unsupervised_loss
```

 Equivalent mathematical expression:

 $$
L =
L_s+\lambda_uL_u
$$

---

 ### 52\. Why is the supervised loss still important?

 A. It anchors learning to actual ground-truth labels\
 B. It is only used for plotting\
 C. It replaces the CNN\
 D. It is needed only during validation

 **Answer: A**

---

 ### 53\. What could happen if training used only consistency loss from a randomly initialised model?

 A. The model could reinforce arbitrary or incorrect predictions\
 B. It would automatically become perfect\
 C. It would have access to ground truth\
 D. Cross entropy would be calculated automatically

 **Answer: A**

 This is one reason supervised warmup and/or confidence filtering can help.

---

 # Part X — Training Mode and Evaluation

 ### 54\. What does:

```
model.train()
```

 do?

 A. Puts the model in training mode\
 B. Trains the model automatically\
 C. Calculates validation accuracy\
 D. Resets the weights

 **Answer: A**

---

 ### 55\. Why is training mode important for layers such as dropout and batch normalization?

 A. Their behaviour can differ between training and evaluation\
 B. They only work on CPUs\
 C. They require labels\
 D. They replace the optimizer

 **Answer: A**

---

 ### 56\. Why should validation normally be performed without training updates?

 A. Validation measures performance without changing the model\
 B. Validation should modify the weights\
 C. Validation requires random labels\
 D. Validation is another optimizer step

 **Answer: A**

---

 # Part XI — Early Stopping

 ### 57\. What is the purpose of early stopping?

 A. Stop training when validation performance has stopped improving\
 B. Stop after one batch regardless of performance\
 C. Increase the learning rate\
 D. Increase the dataset size

 **Answer: A**

---

 ### 58\. What does `stopping_patience` represent?

 A. Number of classes\
 B. Number of epochs allowed without improvement before stopping\
 C. Batch size\
 D. Learning rate

 **Answer: B**

---

 ### 59\. Why is validation accuracy preferable to training accuracy for early stopping?

 A. We care about generalisation to unseen data\
 B. Training accuracy is always zero\
 C. Validation accuracy is the optimization objective\
 D. Training accuracy cannot be calculated

 **Answer: A**

---

 # Part XII — Optimizer and Regularisation

 ### 60\. Which optimizer is used?

```
torch.optim.AdamW(...)
```

 A. SGD\
 B. AdamW\
 C. RMSProp\
 D. Adagrad

 **Answer: B**

---

 ### 61\. What does `lr` mean?

 A. Label ratio\
 B. Learning rate\
 C. Loss regularizer\
 D. Logarithmic reward

 **Answer: B**

---

 ### 62\. What does `l2_reg` control in this implementation?

 A. Weight decay / L2-style regularisation\
 B. Number of labels\
 C. Number of epochs\
 D. Confidence threshold

 **Answer: A**

 The code uses:

```
weight_decay=l2_reg
```

---

 ### 63\. Why can regularisation be helpful?

 A. It can reduce overfitting\
 B. It guarantees 100% accuracy\
 C. It removes the need for training\
 D. It creates labels

 **Answer: A**

---

 # Part XIII — DataLoader Design

 ### 64\. Why might the unlabelled batch size be larger than the labelled batch size?

 A. The unlabelled consistency signal can be noisy, so more examples can provide a smoother estimate\
 B. Unlabelled images require less GPU memory\
 C. Labels cannot fit in large batches\
 D. Cross entropy requires small batches

 **Answer: A**

---

 ### 65\. Why don't we simply write:

```
for labelled_batch, unlabelled_batch in zip(trainL_loader, trainU_loader):
```

 A. The two loaders may contain different numbers of batches\
 B. PyTorch doesn't support tuples\
 C. DataLoader cannot be iterated\
 D. The model requires three loaders

 **Answer: A**

 `zip` stops when the **shorter** iterator ends.

---

 ### 66\. What problem does `cycle(trainU_loader)` solve?

 A. It allows the unlabelled iterator to continue when the labelled loader has more batches\
 B. It removes augmentation\
 C. It balances class frequencies automatically\
 D. It increases validation accuracy automatically

 **Answer: A**

---

 # Part XIV — Metrics

 ### 67\. Why do we track both supervised loss and unsupervised loss?

 A. To understand how each component contributes to training\
 B. Because the model needs two optimizers\
 C. Because accuracy cannot be measured\
 D. Because validation is impossible otherwise

 **Answer: A**

---

 ### 68\. What does:

```
metrics.log_train(
    loss=supervised_loss.item(),
    acc=batch_acc,
    unsup_loss=unsupervised_loss.item()
)
```

 allow us to track?

 A. Training loss, accuracy, and unsupervised loss\
 B. Only validation accuracy\
 C. Only learning rate\
 D. Dataset size

 **Answer: A**

---

 ### 69\. Why is `.item()` commonly used when logging a scalar tensor?

 A. To extract its Python numerical value\
 B. To calculate gradients\
 C. To move the whole model to CPU\
 D. To perform softmax

 **Answer: A**

---

 # Part XV — Experimental Design

 ### 70\. Why should each experimental configuration use a fresh model?

 A. To make comparisons fair and independent\
 B. Because PyTorch cannot train the same model twice\
 C. Because models cannot have optimizers\
 D. To increase the number of classes

 **Answer: A**

 If experiment 2 continued training experiment 1's model, you could not fairly attribute differences to the hyperparameters.

---

 ### 71\. Which hyperparameters were explicitly explored in the provided experiment?

 A. Number of classes and image size\
 B. $\lambda_u$ and warmup epochs\
 C. CNN depth and optimizer type only\
 D. Validation batch size only

 **Answer: B**

---

 ### 72\. What is the purpose of the experiment:

```
(1, 0)
(2, 0)
(3, 0)
(4, 0)
```

 A. Compare different unsupervised-loss weights with no warmup\
 B. Compare different CNN architectures\
 C. Compare different datasets\
 D. Compare four validation sets

 **Answer: A**

---

 ### 73\. What does:

```
(3, 3)
```

 represent?

 A. $\lambda_u=3$, warmup $=3$ epochs\
 B. Learning rate $=3$, batch size $=3$\
 C. 3 labelled and 3 unlabelled examples\
 D. 3 classes and 3 epochs

 **Answer: A**

---

 ### 74\. Why is it useful to compare standard UDA against UDA with confidence filtering?

 A. To experimentally test whether noisy pseudo-targets are hurting training\
 B. To change the dataset\
 C. To eliminate supervised learning\
 D. To guarantee improvement

 **Answer: A**

---

 ### 75\. What should you NOT conclude from one successful confidence-threshold experiment?

 A. That confidence filtering is guaranteed to improve all datasets\
 B. That it may have helped this experiment\
 C. That the threshold is a tunable hyperparameter\
 D. That model confidence can be useful for filtering

 **Answer: A**

---

 # Part XVI — Baseline Comparisons

 ### 76\. Why compare UDA against a partially labelled supervised baseline?

 A. To determine whether exploiting unlabelled data actually helped\
 B. To make training slower\
 C. Because the baseline is required by PyTorch\
 D. Because UDA cannot calculate accuracy

 **Answer: A**

---

 ### 77\. Suppose:

```
Partial-label baseline = 70%
UDA = 76%
```

 What is the improvement?

 A. 6% points\
 B. 8% points\
 C. 46% points\
 D. 70% points

 **Answer: A**

 More precisely:

 $$
76\%-70\%=6\text{ percentage points}
$$

---

 ### 78\. If the practical asks for at least a 5% improvement, what should you compare?

 A. UDA validation accuracy against the partial-label baseline\
 B. Training loss against zero\
 C. Number of epochs against 40\
 D. Batch size against learning rate

 **Answer: A**

---

 # Part XVII — Debugging and Common Errors

 ### 79\. Your UDA model trains, but accuracy is worse than the baseline. Which is a reasonable first thing to investigate?

 A. $\lambda_u$, warmup, and the consistency-loss implementation\
 B. Rename the model\
 C. Delete the validation set\
 D. Increase the number of classes

 **Answer: A**

---

 ### 80\. Which issue was specifically highlighted in the practical?

 A. Incorrect direction of KL divergence\
 B. Using a CNN instead of an RNN\
 C. Using too many classes\
 D. Not using reinforcement learning

 **Answer: A**

---

 ### 81\. Another specifically highlighted issue is:

 A. Failing to detach the unsupervised target logits\
 B. Detaching the input images\
 C. Removing all labels\
 D. Using Python instead of C++

 **Answer: A**

---

 ### 82\. If the consistency loss is enormous and dominates training, what could you investigate?

 A. Reduce $\lambda_u$\
 B. Increase it to 1000 immediately\
 C. Remove supervised loss\
 D. Stop validation

 **Answer: A**

---

 ### 83\. If the consistency loss seems ineffective, which parameter could you increase?

 A. $\lambda_u$\
 B. Number of classes\
 C. Validation size only\
 D. Filename length

 **Answer: A**

---

 ### 84\. Why can unsupervised training be noisy early in training?

 A. The model's predictions are poor, so its consistency targets are unreliable\
 B. The labels are always wrong\
 C. CNNs cannot process augmented images\
 D. AdamW cannot calculate gradients

 **Answer: A**

---

 # Part XVIII — Code Reading Challenge

 ### 85\. What is the purpose of:

```
if e < warmup_epochs:
    current_lambda_u = 0.0
else:
    current_lambda_u = lambda_u
```

 A. A hard transition from supervised-only to supervised \+ UDA\
 B. A learning-rate scheduler\
 C. Early stopping\
 D. Confidence filtering

 **Answer: A**

---

 ### 86\. What does this do?

```
batch_acc = get_batch_acc(pred, y)
```

 A. Calculates accuracy on the labelled batch\
 B. Calculates KL divergence\
 C. Generates pseudo-labels\
 D. Calculates validation loss

 **Answer: A**

---

 ### 87\. What does this do?

```
val_loss, val_acc = evaluate_model(
    model,
    val_loader,
    loss_func
)
```

 A. Evaluates the current model on validation data\
 B. Updates model weights\
 C. Creates unlabelled data\
 D. Performs augmentation

 **Answer: A**

---

 ### 88\. What is the purpose of:

```
model = BasicCNN(num_classes=10)
```

 A. Create a classifier with ten output classes\
 B. Create ten CNNs\
 C. Train the model automatically\
 D. Create ten datasets

 **Answer: A**

---

 ### 89\. Why is the model recreated before each experiment?

 A. To start each experiment from the same type of fresh initialization\
 B. To reset the dataset labels\
 C. To increase confidence\
 D. To disable the optimizer

 **Answer: A**

---

 # Part XIX — Identify the Bug

 ### 90\. Which implementation is most problematic?

 **A**

```
clean_logits = model(ux_clean)
aug_logits = model(ux_aug)

loss_u = unsup_loss_fn(
    clean_logits.detach(),
    aug_logits
)
```

 **B**

```
clean_logits = model(ux_clean)
aug_logits = model(ux_aug)

loss_u = unsup_loss_fn(
    clean_logits,
    aug_logits
)
```

 **C**

```
clean_logits = model(ux_clean)
aug_logits = model(ux_aug)

clean_target = clean_logits.detach()

loss_u = unsup_loss_fn(
    clean_target,
    aug_logits
)
```

 **D**

```
clean_logits = model(ux_clean)
aug_logits = model(ux_aug)

clean_target = clean_logits.detach()
loss_u = unsup_loss_fn(clean_target, aug_logits)
```

 **Answer: B**, assuming `unsup_loss_fn` expects the first argument to be a fixed target.

 **Explanation:**\
 The practical specifically recommends detaching the target logits. The exact implementation depends on how `unsup_loss_fn` was defined, but conceptually the target branch should not receive gradients through the consistency objective.

---

 ### 91\. Which implementation correctly applies confidence filtering?

 A.

```
confidence = clean_probs.max(dim=1).values
mask = confidence >= 0.8

unsup_loss = unsup_loss_fn(
    clean_logits[mask],
    aug_logits[mask]
)
```

 B.

```
confidence = clean_probs.min(dim=1).values
```

 C.

```
mask = confidence <= 0.8
```

 D.

```
mask = y == 0
```

 **Answer: A**

---

 ### 92\. Why should confidence be calculated from the clean prediction?

 A. The clean prediction is treated as the more reliable consistency target\
 B. Augmented images have labels\
 C. Clean images are validation data\
 D. It makes the optimizer faster

 **Answer: A**

---

 # Part XX — Deep Conceptual Questions

 ### 93\. What assumption does consistency regularisation make about the data?

 A. Small, label-preserving perturbations should not change the semantic class\
 B. Every augmentation changes the class\
 C. All images belong to one class\
 D. Unlabelled examples are useless

 **Answer: A**

 This is a fundamental assumption behind the technique.

---

 ### 94\. Why can data augmentation help semi-supervised learning?

 A. It provides multiple views of the same underlying example\
 B. It creates perfect ground-truth labels\
 C. It eliminates the need for a classifier\
 D. It guarantees calibration

 **Answer: A**

---

 ### 95\. What is the danger of an augmentation that changes the semantic class?

 A. The consistency objective may force the model to give the same label to semantically different images\
 B. It always improves performance\
 C. It increases the number of labels\
 D. It disables backpropagation

 **Answer: A**

 This is why augmentation must be **label-preserving**.

---

 ### 96\. What is pseudo-labelling?

 A. Using a model's prediction as a temporary target for an unlabelled example\
 B. Manually labelling every example\
 C. Removing labels\
 D. Randomly assigning labels

 **Answer: A**

 UDA can be interpreted as a form of **soft pseudo-target consistency**.

---

 ### 97\. Why is a soft probability distribution potentially preferable to a hard pseudo-label?

 A. It retains information about uncertainty across classes\
 B. It always has exactly one class\
 C. It requires no model\
 D. It removes confidence information

 **Answer: A**

 For example, instead of:

```
cat = 1
dog = 0
bird = 0
```

 the model might provide:

```
cat = 0.80
dog = 0.15
bird = 0.05
```

 The latter contains uncertainty information.

---

 # Part XXI — Practical Report

 ### 98\. What must the practical report include?

 A. Only the final accuracy\
 B. Results, plots, explanation of the solution, issues, and insights\
 C. Only the source code\
 D. Only screenshots of the notebook

 **Answer: B**

---

 ### 99\. Which comparison is explicitly requested?

 A. Baseline supervised performance with partial labels\
 B. Augmentation applied to unlabelled data\
 C. Semi-supervised performance using unlabelled data\
 D. All of the above

 **Answer: D**

---

 ### 100\. How long does the report need to be?

 A. At least 20 pages\
 B. Exactly 10 pages\
 C. Around one page is sufficient\
 D. No report is required

 **Answer: C**

---

 ### 101\. What must be submitted for credit?

 A. Only the PDF\
 B. Only the notebook\
 C. Both the text report and completed notebook\
 D. Only the final accuracy

 **Answer: C**

---

 # Part XXII — Mastery-Level Scenario Questions

 ### 102\. You get:

```
Partial-label baseline: 72%
UDA: 74%
```

 What should you conclude?

 A. UDA improved performance by 2 percentage points, but did not reach the suggested 5-point improvement\
 B. UDA failed completely\
 C. UDA improved by 74%\
 D. The model is unusable

 **Answer: A**

---

 ### 103\. You get:

```
Baseline = 72%
UDA = 79%
```

 What is the improvement?

 A. 7 percentage points\
 B. 9 percentage points\
 C. 79 percentage points\
 D. 51 percentage points

 **Answer: A**

 This exceeds the suggested 5-point improvement.

---

 ### 104\. Your confidence threshold is 0.99 and almost no unlabelled examples pass the mask. What is likely?

 A. The unsupervised learning signal becomes very weak\
 B. The supervised signal becomes stronger automatically\
 C. All predictions become correct\
 D. The model receives more labels

 **Answer: A**

---

 ### 105\. Your threshold is 0.1 and almost every example passes. What happens conceptually?

 A. Confidence filtering provides very little filtering\
 B. All examples become labelled\
 C. The model stops training\
 D. KL divergence becomes zero

 **Answer: A**

---

 ### 106\. You increase $\lambda_u$ from 1 to 10 and performance gets substantially worse. What is a reasonable interpretation?

 A. The unsupervised objective may be overpowering the reliable supervised signal\
 B. Higher $\lambda_u$ is always better\
 C. The validation set is broken\
 D. Cross entropy has stopped working

 **Answer: A**

---

 ### 107\. You train with zero warmup and poor early predictions. Why might performance suffer?

 A. The model immediately receives a noisy consistency signal from unreliable predictions\
 B. The model receives too many ground-truth labels\
 C. The learning rate becomes zero\
 D. Validation happens too late

 **Answer: A**

---

 ### 108\. You introduce three warmup epochs and performance improves. What is the most plausible explanation?

 A. The model gets time to learn useful representations from reliable supervised labels before relying heavily on its own predictions\
 B. The number of classes decreases\
 C. The validation set becomes labelled\
 D. The optimizer changes to SGD

 **Answer: A**

---

 # Part XXIII — "Explain This Line" Challenge

 ### 109\. Explain:

```
trainU_iter = iter(itertools.cycle(trainU_loader))
```

 **Answer:**

 It creates an iterator over an **infinite repetition of the unlabelled DataLoader**, allowing every labelled batch to obtain an unlabelled batch even when the two loaders have different lengths.

---

 ### 110\. Explain:

```
clean_probs = F.softmax(clean_logits.detach(), dim=1)
```

 **Answer:**

 It:

 1. detaches the clean logits from gradient computation,
2. converts them to class probabilities using softmax,
3. applies softmax across the class dimension.

---

 ### 111\. Explain:

```
confidence = clean_probs.max(dim=1).values
```

 **Answer:**

 For every unlabelled example, it finds the probability assigned to the model's most likely class. This becomes the model's confidence score.

---

 ### 112\. Explain:

```
mask = confidence >= confidence_threshold
```

 **Answer:**

 It creates a Boolean mask selecting only examples whose predicted class has confidence at least the specified threshold.

---

 ### 113\. Explain:

```
clean_logits[mask]
```

 **Answer:**

 It selects only the high-confidence examples from the clean prediction batch.

---

 ### 114\. Explain:

```
total_loss = supervised_loss + lambda_u * unsupervised_loss
```

 **Answer:**

 It combines:

 - reliable ground-truth supervision
- consistency-based supervision from unlabelled data

 with $\lambda_u$ controlling the relative importance of the latter.

---

 # Part XXIV — Ultimate Integration Questions

 ### 115\. Which sequence best describes one UDA training iteration?

 A.

```
Load labelled data
→ calculate supervised prediction/loss
→ load unlabelled clean/augmented data
→ calculate clean/augmented predictions
→ calculate consistency loss
→ combine losses
→ backward
→ optimizer step
```

 B.

```
Validation
→ delete labels
→ optimizer step
→ load data
```

 C.

```
Generate labels manually
→ validation
→ augmentation
```

 D.

```
Optimizer step
→ calculate loss
→ load data
```

 **Answer: A**

---

 ### 116\. What are the three essential ingredients of this practical?

 A. Supervised loss, unlabelled consistency loss, and labelled/unlabelled data\
 B. Reinforcement learning, GANs, and transformers\
 C. Regression, clustering, and PCA\
 D. Validation labels, test labels, and manually created labels

 **Answer: A**

---

 ### 117\. What is the conceptual difference between the two losses?

 | Loss | Information |
| --- | --- |
| $L_s$ | Ground-truth labels |
| $L_u$ | Agreement between predictions |

**Question:** Why is this distinction important?

 A. $L_s$ provides an external source of correctness, while $L_u$ encourages smooth/consistent predictions\
 B. Both require ground-truth labels\
 C. Neither requires model predictions\
 D. $L_u$ replaces the CNN

 **Answer: A**

---

 ### 118\. What is the fundamental reason UDA can exploit unlabelled data?

 A. The model can learn useful information from the structure of inputs and consistency under perturbations even without explicit labels\
 B. Unlabelled images secretly contain labels\
 C. PyTorch automatically creates labels\
 D. Validation data are converted into training data

 **Answer: A**

---

 ### 119\. What is the biggest conceptual danger of consistency regularisation?

 A. The model can reinforce its own incorrect predictions\
 B. It cannot process images\
 C. It always overfits the validation set\
 D. It requires millions of labels

 **Answer: A**

 This is the **confirmation-bias/self-training problem**.

---

 ### 120\. Why do warmup and confidence filtering address this danger?

 A. They reduce reliance on unreliable model predictions\
 B. They eliminate the supervised objective\
 C. They increase label noise\
 D. They make augmentation class-changing

 **Answer: A**

---

 # Final Mastery Question

 ### 121\. Which statement best summarises the entire practical?

 A. Train a CNN exclusively on a small labelled dataset and ignore the unlabelled data.

 B. Train a CNN using labelled cross-entropy plus a consistency objective that encourages predictions on clean and augmented versions of unlabelled examples to agree, while experimenting with the weighting, warmup, and confidence filtering to improve generalisation.

 C. Replace supervised learning completely with pseudo-labels.

 D. Use the validation set as additional unlabelled training data.

 **Answer: B — this is the core idea of the entire practical.**

---

 # The Practical in One Mental Model

 If you remember only one diagram, remember this:

```
                 LABELLED DATA
                  (x, y)
                     │
                     ▼
                  CNN
                     │
                     ▼
              supervised loss
                 CE(pred, y)
                     │
                     │
                     ├──────────────┐
                     │              │
                     ▼              │
                 total loss         │
                     ▲              │
                     │              │
                     │              │
              λu × consistency      │
                     ▲              │
                     │              │
        ┌────────────┴────────────┐ │
        │                         │ │
        ▼                         ▼ │
   clean image              augmented image
        │                         │
        ▼                         ▼
       CNN                       CNN
        │                         │
        ▼                         ▼
 clean prediction            aug prediction
        │                         │
        └──────────┬──────────────┘
                   │
                   ▼
              KL divergence
                   │
                   ▼
             consistency loss
```

 The fundamental equation is:

 $$
\boxed{
L_{\text{total}}
=
L_{\text{supervised}}
+
\lambda_u L_{\text{unsupervised}}
}
$$

 with:

 $$
L_{\text{supervised}}
=
CE(f(x_L),y_L)
$$

 and conceptually:

 $$
L_{\text{unsupervised}}
=
KL\left(
p_{\text{clean}}
\;\|\;
p_{\text{augmented}}
\right)
$$

 where the clean prediction acts as the consistency target.

 ## The five ideas you absolutely need to master

 1. **Semi-supervised learning:** use a small labelled set plus a larger unlabelled set.
2. **Consistency regularisation:** the same image under a label-preserving augmentation should produce a similar prediction.
3. **UDA:** uses clean/unlabelled predictions as soft targets for augmented versions.
4. **Confidence/warmup:** reduce the damage caused by unreliable early predictions.
5. **Loss balancing:**\

   $$
   L=L_s+\lambda_uL_u
   $$
   \
    and $\lambda_u$, warmup, and confidence threshold are important experimental hyperparameters.

 If you can explain **why each of those five ideas exists, what tensor goes into each line of the code, why the clean prediction is detached, why KL is used, and how `lambda_u`, warmup, and confidence filtering affect training**, you understand the practical rather than merely memorising the code.
