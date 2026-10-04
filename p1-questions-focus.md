Absolutely. Since students can **only hear you**, the script should be written as a **spoken narrative**, not as something that depends on slides, figures, or students seeing code. It should also move naturally through **questions → explanation → code concept → practical result**, so it feels like a guided viva/exam preparation rather than a lecture.

 Below is the regenerated version, incorporating your **actual results: 71.2% baseline → 80.7% transfer learning**.

 # Practical 1 — Question-Focused Narrative Speech

 ## Opening

 Let me walk you through Practical 1 in a way that focuses on the kinds of questions you should be able to answer in an exam or viva.

 I don't want you to memorize individual lines of code.

 Instead, I want you to understand **what the code is doing, why it is doing it, and how the different parts connect together**.

 The entire practical is built around one central question:

 **Can we improve the performance of a CNN on a small target dataset by first training it on a larger source dataset?**

 Everything in the notebook is designed to answer that question.

 So, as we go through the practical, keep asking yourself three things:

 **What is the code doing? Why are we doing it? And what would happen if we changed or removed it?**

---

 # Question 1: What is the main objective of the practical?

 The answer is:

 **To maximize validation accuracy on a small target dataset and determine whether transfer learning improves performance compared with training from scratch.**

 We actually have two approaches.

 First, we train a CNN directly on the target data.

 That gives us our **baseline**.

 Then we train another CNN on a larger source dataset, save what it has learned, and use that knowledge for the target task.

 That is our **transfer-learning experiment**.

 So the baseline gives us something to compare against.

 Without a baseline, we wouldn't know whether transfer learning actually helped.

---

 # Question 2: What are the source and target domains?

 This is one of the most important questions.

 The **source domain is CIFAR-10**.

 The **target domain is a selected superclass from CIFAR-100**.

 In our experiment, we selected the **Vehicles superclass**.

 That gives us five target classes:

 bicycle, bus, motorcycle, pickup truck, and train.

 So remember this very clearly:

 **CIFAR-10 is the source.**

 **CIFAR-100 Vehicles is the target.**

 The source is where we learn general visual representations.

 The target is where we ultimately care about performance.

---

 # Question 3: How much target data do we actually have?

 CIFAR-100 contains 100 fine-grained classes.

 Those are grouped into 20 superclasses.

 Each superclass contains five classes.

 Each fine-grained class contains 500 images.

 So for Vehicles, we have:

 five classes multiplied by 500 images.

 That gives us **2,500 images**.

 We then use an 80/20 split.

 That means approximately:

 **2,000 training images**

 and

 **500 validation images**.

 Now think about why this matters.

 Two thousand images may sound like a lot, but for training a CNN from scratch, it is relatively small.

 The model has many parameters and therefore has enough capacity to memorize aspects of the training data.

 This creates the possibility of **overfitting**.

 And that is exactly where transfer learning becomes useful.

---

 # Question 4: What is overfitting?

 Suppose our training accuracy keeps increasing.

 But our validation accuracy increases for a while and then stops improving, or even starts decreasing.

 What does that tell us?

 It suggests that the model is learning the training examples very well but is not generalizing equally well to unseen examples.

 That is **overfitting**.

 So if I ask you:

 **What is the difference between training accuracy and validation accuracy?**

 You should say:

 Training accuracy measures performance on examples used to update the model.

 Validation accuracy measures performance on held-out examples that are not used to update the model.

 The validation result is therefore a better indication of how well the model generalizes.

---

 # Question 5: Why do we remap the target labels?

 Now let's think about the code.

 CIFAR-100 has 100 classes, and each class has an original numerical label.

 But we are only selecting five classes from Vehicles.

 Those five original labels might not be:

 zero, one, two, three, and four.

 They could be arbitrary values.

 For example, imagine the original labels were:

 four, eight, nineteen, thirty-one, and fifty-five.

 Our target classifier only has five outputs.

 Therefore, it is much cleaner to remap those labels to:

 zero, one, two, three, and four.

 This is called **label remapping**.

 So if you are asked:

 **Why do we remap the labels?**

 The answer is:

 Because our target problem has five classes, so we need consecutive class indices from zero to four for the five-class classifier and cross-entropy loss.

---

 # Question 6: What does the CNN look like?

 Now let's think about the model.

 The input image is:

 **3 by 32 by 32.**

 Why three?

 Because the image is RGB.

 Why 32 by 32?

 Because CIFAR images are 32 by 32 pixels.

 The first convolution changes the number of channels from three to 32.

 Then we apply ReLU.

 Then max pooling.

 Then another convolution.

 Then ReLU.

 Then another max pooling operation.

 Then another convolution.

 Eventually, we flatten the feature maps and pass them through fully connected layers.

 Finally, we have the classifier.

 The important thing is to understand what each component does.

---

 # Question 7: What does convolution do?

 A convolutional layer learns visual patterns.

 Early convolutional layers can learn relatively simple patterns such as:

 edges,

 textures,

 and local shapes.

 Later layers can combine these into increasingly complex visual representations.

 This is actually very important for transfer learning.

 Why?

 Because those early and intermediate visual representations can often be useful for a different image classification problem.

 The model doesn't necessarily need to relearn everything from zero.

---

 # Question 8: Why do we use ReLU?

 ReLU is the function:

 maximum of zero and x.

 In other words:

 negative values become zero,

 and positive values remain positive.

 Its main purpose here is to introduce **non-linearity** into the network.

 Without non-linear activation functions, stacking linear layers would still effectively produce a linear transformation.

 So if I ask:

 **Why is ReLU used?**

 Think:

 **non-linearity.**

---

 # Question 9: What does max pooling do?

 Max pooling reduces the spatial dimensions of the feature maps.

 We start with 32 by 32.

 After the first pooling operation, we get:

 16 by 16.

 After the second pooling operation, we get:

 8 by 8.

 At that point we have 64 channels.

 So what happens when we flatten?

 We calculate:

 64 multiplied by 8 multiplied by 8.

 That equals:

 **4,096.**

 That is why the first fully connected layer has:

 **4,096 inputs.**

 This is a very common code-tracing question.

 If I ask:

 **Why does the linear layer have 4096 inputs?**

 You should be able to calculate it yourself:

 64 channels times 8 times 8 spatial dimensions equals 4096.

---

 # Question 10: What does the final classifier produce?

 The final layer produces **logits**.

 For the target task, we have five classes.

 Therefore the final classifier produces five logits.

 So the target model ends with something conceptually like:

 128 inputs going to 5 outputs.

 For CIFAR-10 pre-training, however, we need:

 128 inputs going to 10 outputs.

 Why?

 Because CIFAR-10 has ten classes.

 This difference becomes extremely important later.

---

 # Question 11: What are logits?

 Logits are the raw numerical outputs produced by the final classifier before converting them into probabilities.

 Suppose the model produces:

 one point two,

 minus zero point four,

 three point eight,

 zero point seven,

 and zero point two.

 The largest value is three point eight.

 That corresponds to class two if the classes are indexed from zero.

 Therefore:

 `argmax`

 gives us the predicted class.

 So remember:

 **logits → argmax → predicted class.**

---

 # Question 12: Why do we use cross-entropy loss?

 We use **cross-entropy loss** because this is a multi-class classification problem.

 The model produces logits.

 The true label tells us which class is correct.

 Cross-entropy measures how well the predicted distribution corresponds to the correct class.

 An important implementation detail is that we pass the raw logits directly to `CrossEntropyLoss`.

 We normally don't manually apply softmax first because PyTorch's cross-entropy implementation handles the appropriate log-softmax calculation internally.

 So if I ask:

 **Should we apply softmax before CrossEntropyLoss in this implementation?**

 The answer is:

 **No.**

---

 # Question 13: What happens during one training step?

 This is one of the most important code sequences in the notebook.

 Imagine we have one batch of images.

 First:

 we perform a **forward pass**.

 The images go through the CNN and produce logits.

 Second:

 we calculate the loss using the logits and the true labels.

 Third:

 we call `backward()`.

 This calculates the gradients.

 Fourth:

 we call `optimizer.step()`.

 That actually updates the model parameters.

 And before doing all of this again for the next batch, we call:

 `optimizer.zero_grad()`.

 Why?

 Because PyTorch accumulates gradients.

 So the basic sequence is:

 **forward, loss, backward, update.**

 And before the next iteration:

 **clear the previous gradients.**

---

 # Question 14: What does Adam do?

 Adam is the optimizer.

 Its job is to use the calculated gradients to update the model parameters.

 The learning rate controls the approximate size of those updates.

 In our baseline, we use a learning rate of:

 **1 times 10 to the minus 3.**

 We also use weight decay.

 Weight decay provides an L2-style regularization effect.

 The purpose is to discourage unnecessarily large weights and can help with generalization.

---

 # Question 15: What is the difference between an epoch and a training step?

 A **training step** refers to processing one batch and performing an optimization update.

 An **epoch** is one complete pass through the training dataset.

 We have approximately 2,000 training images.

 Our batch size is 128.

 So:

 2,000 divided by 128 is approximately 15.6.

 Therefore, we process roughly 16 batches per epoch, depending on the DataLoader settings.

 This distinction is important.

 A batch is not an epoch.

 One epoch contains multiple training steps.

---

 # Question 16: Why do we use model.train()?

 During training, we use:

 `model.train()`.

 This tells PyTorch that the model is operating in training mode.

 When we evaluate the validation data, we use:

 `model.eval()`.

 This switches the model into evaluation mode.

 So if I ask:

 **Which mode is used for training?**

 Answer:

 `train()`.

 **Which mode is used for validation?**

 Answer:

 `eval()`.

---

 # Question 17: Why do we use torch.no\_grad() during validation?

 During validation, we aren't updating the model.

 We only want to measure performance.

 Therefore, we don't need to construct the gradient computation graph.

 So we use:

 `torch.no_grad()`.

 This reduces unnecessary computation and memory usage.

 The typical validation combination is:

 **model.eval() plus torch.no\_grad().**

 Remember that combination.

---

 # Question 18: What was our baseline result?

 Now we come to the first major experimental result.

 We trained the BasicCNN **from scratch** on the 2,000 target training examples.

 The best validation accuracy we obtained was:

 **71.2 percent.**

 This is our baseline.

 It gives us the number that transfer learning has to beat.

 And this is extremely important scientifically.

 We don't simply say:

 "Our fine-tuned model achieved 80.7 percent."

 We ask:

 "Did it improve compared with training from scratch?"

 The baseline allows us to answer that question.

---

 # Question 19: Why did we pre-train on CIFAR-10?

 Now we move to the second stage.

 Instead of starting with a randomly initialized model, we train a BasicCNN on CIFAR-10.

 CIFAR-10 provides a much larger source dataset.

 The model can therefore learn useful visual representations from many more examples.

 Think about what the CNN may learn:

 edges,

 textures,

 color patterns,

 simple shapes,

 and increasingly complex visual structures.

 These representations aren't necessarily specific to one class.

 For example, recognizing edges can be useful whether the final task involves a cat, a vehicle, or something else.

 That is the fundamental reason transfer learning can work.

---

 # Question 20: Why can transfer learning work if the source and target classes are different?

 This is probably the most important conceptual question in the entire practical.

 The answer is:

 **Because we are not transferring the source labels. We are transferring learned visual representations.**

 CIFAR-10 does not need to contain the exact same five target classes.

 The CNN learns useful representations from the source data.

 Those representations can then be adapted to the target problem.

 So remember this distinction:

 **We transfer knowledge, not necessarily class labels.**

---

 # Question 21: What is pre-training?

 Pre-training means:

 **training the model on the source dataset before using it for the target task.**

 In our case:

 we create a CNN with ten outputs,

 because CIFAR-10 has ten classes.

 Then we train it on CIFAR-10.

 Once training is complete, we save the learned parameters.

 We save the model's `state_dict`.

---

 # Question 22: What is a state\_dict?

 A `state_dict` contains the model's learned parameter values.

 It allows us to save the trained weights and load them later.

 So when we save:

 `pretrained_model.ckpt`

 we are preserving the learned parameters from the source-domain training.

 We can then create another model and load those parameters into it.

 This is the mechanism that allows us to reuse the pretrained knowledge.

---

 # Question 23: What is the biggest implementation problem during transfer learning?

 Now we reach one of the most important coding issues.

 Our pretrained model was trained for CIFAR-10.

 Therefore its final classifier is:

 **128 to 10.**

 But our target task has five classes.

 Therefore the target classifier needs to be:

 **128 to 5.**

 We cannot simply use the original ten-output classifier for a five-class problem.

 The output space is wrong.

---

 # Question 24: How do we solve the classifier mismatch?

 We first create the model with the same architecture as the pretrained model.

 Then we load the pretrained checkpoint.

 After that, we replace the classifier.

 Conceptually, we do:

 the pretrained feature extractor,

 then remove the old ten-class classifier,

 then attach a new five-class classifier.

 The new classifier is newly initialized.

 The earlier layers contain the transferred knowledge.

 This is one of the most important implementation details in the practical.

---

 # Question 25: Why don't we replace the entire network?

 Because the earlier layers have already learned useful visual representations.

 We want to reuse those representations.

 The final classifier is more task-specific because it maps those representations to particular class labels.

 So we reuse the feature-learning part and replace the task-specific output layer.

 This is the basic logic behind transfer learning.

---

 # Question 26: What is fine-tuning?

 Fine-tuning means adapting the pretrained model to the target task.

 In our experiment, we load the pretrained CIFAR-10 model.

 Then we replace its ten-class classifier with a five-class classifier.

 Then we train the model on the CIFAR-100 Vehicles target data.

 Importantly, in our implementation, we do **not explicitly freeze the convolutional layers**.

 Therefore, the pretrained layers can continue to update during fine-tuning.

 That is why we call this **fine-tuning**.

---

 # Question 27: Why do we use a smaller learning rate during fine-tuning?

 During baseline training, we use:

 **1 times 10 to the minus 3.**

 During fine-tuning, we use:

 **1 times 10 to the minus 4.**

 Why?

 Because the pretrained model already contains useful information.

 We don't want to make extremely large updates that could destroy those useful representations.

 A smaller learning rate allows the pretrained features to be adjusted more gently.

 So remember:

 **pretrained knowledge → smaller updates → lower learning rate.**

---

 # Question 28: What does freezing a layer mean?

 Freezing a layer means preventing its parameters from being updated during training.

 If we froze the convolutional layers, they would retain their pretrained values.

 Only the unfrozen parts, such as the classifier, would be updated.

 Our experiment does not explicitly freeze the convolutional layers.

 So the pretrained features are allowed to adapt to the target domain.

---

 # Question 29: What is domain shift?

 Domain shift refers to differences between the source and target data distributions or tasks.

 CIFAR-10 and CIFAR-100 Vehicles aren't identical.

 They contain different classes and therefore represent different tasks.

 However, they are both small RGB natural-image datasets with the same image dimensions.

 So there is some related visual structure.

 This makes transfer learning reasonable.

 If the source and target were completely unrelated, transfer learning might provide little benefit.

 In extreme cases, it could even hurt performance.

 That is called **negative transfer**.

---

 # Question 30: What was our final result?

 Now we compare the two experiments.

 The baseline, trained from scratch on the target data, achieved:

 **71.2 percent best validation accuracy.**

 The transfer-learning model achieved:

 **80.7 percent best validation accuracy.**

 So transfer learning improved validation accuracy by:

 80.7 minus 71.2.

 That gives:

 **9.5 percentage points.**

 That is a substantial improvement.

 And it is exactly the type of result this practical is designed to demonstrate.

---

 # Question 31: What does the 9.5 percentage-point improvement actually tell us?

 It tells us that the features learned from the larger CIFAR-10 source dataset were useful for the CIFAR-100 Vehicles target task.

 Even though the source and target classes were different, the pretrained model started from a more useful representation than a randomly initialized model.

 Instead of learning everything from only 2,000 target training examples, the model first learned general visual features from the much larger source dataset.

 That gave the target task a better starting point.

---

 # Question 32: Does this mean transfer learning always improves accuracy?

 No.

 This is another important conceptual question.

 Transfer learning is not guaranteed to help.

 It depends on how useful the source knowledge is for the target task.

 If the source and target are sufficiently related, transfer can help.

 If they are very different, transfer may provide little benefit.

 In some cases, it can actually hurt performance.

 That is negative transfer.

 So never say:

 "Transfer learning always improves accuracy."

 Instead say:

 **Transfer learning can improve target performance when the source-learned representations are useful for the target task.**

---

 # Question 33: Which features are usually most transferable?

 Generally, early CNN layers learn relatively general features.

 Things like:

 edges,

 corners,

 textures,

 and simple patterns.

 These can transfer relatively well between image tasks.

 Later layers become more specialized to the source task.

 The final classifier is usually the most task-specific part.

 That is why replacing the final classifier is so common in transfer learning.

---

 # Question 34: What happens if we don't replace the classifier?

 Suppose we load the pretrained CIFAR-10 model and leave the ten-output classifier unchanged.

 The model would still produce ten outputs.

 But our target problem has only five classes.

 Therefore the output space doesn't match the target labels.

 This can produce a size mismatch when loading parameters if the architecture doesn't match, or simply produce an inappropriate classifier for the target task.

 So the safe sequence is:

 **create the source-compatible architecture, load the checkpoint, then replace the classifier with the target-specific classifier.**

---

 # Question 35: Why must the architecture match when loading a state\_dict?

 Because the checkpoint contains parameters with particular names and shapes.

 For example, the source classifier contains parameters corresponding to ten outputs.

 If we try to load those parameters directly into a five-output classifier, the shapes don't match.

 That's why the order matters.

 First:

 create the architecture matching the checkpoint.

 Second:

 load the pretrained weights.

 Third:

 replace the final classifier.

 Fourth:

 fine-tune on the target task.

---

 # Question 36: What is the difference between training from scratch and fine-tuning?

 Training from scratch means the parameters begin with random initialization.

 The model must learn useful visual representations from the target data.

 Fine-tuning starts with parameters that were already learned from the source dataset.

 Therefore the model doesn't begin from zero.

 This is especially useful when the target dataset is small.

---

 # Question 37: Why is validation accuracy more important than training accuracy for our conclusion?

 Because our goal isn't simply to memorize the target training examples.

 Our goal is to generalize to unseen target examples.

 Therefore the key comparison is:

 **baseline validation accuracy versus fine-tuned validation accuracy.**

 In our experiment:

 baseline:

 **71.2 percent.**

 fine-tuned:

 **80.7 percent.**

 That is the evidence that transfer learning helped.

---

 # Question 38: What would indicate overfitting in the training curves?

 Imagine training accuracy continues increasing toward a very high value.

 But validation accuracy stops increasing and starts falling.

 That would indicate overfitting.

 The model is becoming increasingly specialized to the training examples.

 This is particularly relevant because our target dataset is small.

 Transfer learning helps by giving the model useful representations before it sees the small target dataset.

---

 # Question 39: What would underfitting look like?

 If both training and validation accuracy remain low, the model may not have enough capacity, may not be trained sufficiently, or the optimization may not be working properly.

 That is more suggestive of **underfitting or an optimization problem** rather than classic overfitting.

 So:

 high training, low validation → think overfitting.

 low training, low validation → think underfitting or optimization problems.

---

 # Question 40: Why do we compare curves rather than only the final accuracy?

 Because the training curves tell us how the model behaves throughout training.

 We can see whether:

 training accuracy is increasing,

 validation accuracy is increasing,

 the model begins to overfit,

 or transfer learning gives faster or better convergence.

 The final number is important, but the learning curves help us understand why that number occurred.

---

 # Question 41: What would happen if we froze the convolutional layers?

 If we froze the pretrained feature layers, their parameters would not change during fine-tuning.

 The new classifier would learn to map the existing representations to the five target classes.

 This can be useful when the source and target domains are fairly similar.

 However, if the target domain differs significantly, allowing the feature layers to adapt can sometimes provide better performance.

 Our implementation allows the pretrained parameters to continue updating.

---

 # Question 42: Why do we normalize the images?

 Normalization puts the input values into a more suitable numerical range.

 For the normalization:

 mean equals 0.5,

 standard deviation equals 0.5,

 the transformation is approximately:

 input minus 0.5,

 divided by 0.5.

 We do this independently for each RGB channel.

 Normalization can make optimization more stable and consistent.

---

 # Question 43: What does the DataLoader do?

 The DataLoader provides the data to the model in batches.

 Instead of giving the entire dataset to the model at once, we process smaller groups of examples.

 For example, our batch size is 128.

 The DataLoader also handles things such as shuffling for the training set.

 So:

 **Dataset stores or provides examples.**

 **DataLoader organizes those examples into batches for training or evaluation.**

---

 # Question 44: What does collate\_fn do?

 The collate function controls how individual examples are combined into a batch.

 In our practical, it helps turn individual image-label pairs into tensors that the model can process as a batch.

 So if I ask:

 **What is the purpose of collate\_fn?**

 Think:

 **custom batch construction.**

---

 # Question 45: Why do we maintain separate metric objects?

 We have different stages:

 baseline,

 pre-training,

 and fine-tuning.

 We want to keep their metrics separate.

 For example:

 `baseline_metrics`

 tracks the baseline.

 `pt_metrics`

 tracks source-domain pre-training.

 `ft_metrics`

 tracks target-domain fine-tuning.

 This allows us to compare the experiments clearly.

---

 # Question 46: What is the most important code sequence in the entire practical?

 If I asked you to describe the practical without showing you the notebook, I would expect something like this:

 First, select the Vehicles superclass from CIFAR-100.

 Then select its five classes.

 Then remap their labels to zero through four.

 Then split the target data into training and validation sets.

 Then train the BasicCNN from scratch.

 That gives us the baseline.

 In our experiment, that baseline achieved **71.2 percent validation accuracy**.

 Next, create another BasicCNN with ten outputs.

 Train it on CIFAR-10.

 Save its learned parameters.

 Then create the fine-tuning model.

 Load the pretrained parameters.

 Replace the ten-output classifier with a five-output classifier.

 Then fine-tune on the Vehicles target data using the smaller learning rate.

 Finally, compare the target validation accuracy.

 That gave us **80.7 percent**.

 Therefore, transfer learning improved performance by **9.5 percentage points**.

---

 # Question 47: If I ask you to explain the entire practical in one answer, what should you say?

 Here is the answer I would want you to be able to give naturally:

 We first train a BasicCNN from scratch on a small target dataset consisting of the five Vehicles classes from CIFAR-100. This establishes a baseline, which in our experiment achieved 71.2 percent best validation accuracy.

 We then pre-train the same CNN architecture on the larger CIFAR-10 source dataset. The purpose of this stage is to learn useful visual representations such as edges, textures and shapes.

 After pre-training, we save the model parameters.

 We then load those parameters for the target task. Because CIFAR-10 has ten classes but the target Vehicles task has only five classes, we replace the final ten-output classifier with a new five-output classifier.

 We then fine-tune the model on the target dataset using a smaller learning rate so that the pretrained representations can be adapted without making excessively large updates.

 The fine-tuned model achieved 80.7 percent validation accuracy compared with 71.2 percent for the baseline.

 Therefore, transfer learning produced a 9.5 percentage-point improvement and demonstrated that representations learned from a larger source dataset can improve performance on a smaller target task even when the source and target classes are different.

---

 # Final Rapid-Fire Questions

 Before the exam, make sure you can answer these without hesitation.

 **What is the source?**

 CIFAR-10.

 **What is the target?**

 CIFAR-100 Vehicles.

 **How many target classes?**

 Five.

 **How many target training images?**

 2,000.

 **How many target validation images?**

 **How many CIFAR-10 classes?**

 Ten.

 **Why remap labels?**

 To convert the selected target classes into consecutive labels from zero to four.

 **What is the baseline?**

 Training the CNN from scratch on the target data.

 **Baseline result?**

 71.2 percent validation accuracy.

 **What is pre-training?**

 Training on the source dataset before the target task.

 **What is fine-tuning?**

 Adapting the pretrained model to the target task.

 **Why replace the classifier?**

 Because the source has ten classes and the target has five.

 **What does the new classifier output?**

 Five logits.

 **Why can different classes still transfer?**

 Because useful visual representations can be shared across image tasks.

 **Why use a smaller learning rate during fine-tuning?**

 To make smaller updates to already useful pretrained parameters.

 **What is overfitting?**

 Good training performance but poor generalization to validation data.

 **What does `train()` do?**

 Puts the model in training mode.

 **What does `eval()` do?**

 Puts the model in evaluation mode.

 **Why `torch.no_grad()`?**

 Because validation does not require gradients.

 **What does `backward()` do?**

 Computes gradients.

 **What does `optimizer.step()` do?**

 Updates parameters.

 **What does `zero_grad()` do?**

 Clears accumulated gradients.

 **What does `argmax()` do?**

 Selects the class with the largest logit.

 **What does `state_dict()` contain?**

 The model's learned parameter values.

 **What is domain shift?**

 A difference between the source and target data distributions or tasks.

 **What is negative transfer?**

 When transferred knowledge hurts target performance.

 **What was the final transfer-learning result?**

 80.7 percent.

 **How much better was it than the baseline?**

 9.5 percentage points.

---

 # Closing Message

 So, if you remember only one story from this practical, remember this:

 We have a **small target problem**.

 Training from scratch gives us a baseline of **71.2 percent**.

 We then use a **larger source dataset, CIFAR-10**, to learn useful visual representations.

 We save those learned parameters.

 We load them into a target model.

 Because the source has ten classes and the target has five, we **replace the classifier**.

 Then we **fine-tune** the model on the target data using a smaller learning rate.

 The result is **80.7 percent validation accuracy**.

 So the improvement is:

 **80.7 minus 71.2 equals 9.5 percentage points.**

 That is the central message of the practical:

 **Pre-training learns reusable representations from the source domain, while fine-tuning adapts those representations to the target task.**

 And the most important implementation detail is:

 **The pretrained CIFAR-10 classifier has ten outputs, so it must be replaced with a new five-output classifier before fine-tuning on the CIFAR-100 Vehicles task.**

 If you understand those two ideas, and you understand how the code implements them, you understand the core of Practical 1.

 This version is deliberately written so you can **read it aloud without relying on students seeing your screen**. It also gives you natural pauses where you can ask the class a question before explaining the answer.
