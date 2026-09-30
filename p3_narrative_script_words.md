Absolutely. Since the audience will **only hear you**, the script should sound like a guided explanation rather than a description of diagrams or notebook cells. I’ll make the flow conversational, explain what each stage is doing and why, and explicitly connect the code to the machine-learning concepts.

 # Advanced Machine Learning Practical 3 — Narrative Explanation Script

 ## Introduction

 In this practical, we are looking at **self-supervised learning**, and more specifically, how we can use a large collection of unlabelled images to learn useful visual features before training a model on a smaller labelled dataset.

 The central question of the practical is:

 **Can we improve performance on a target classification problem by first learning a visual representation from unlabelled data?**

 The important idea is that we are not simply trying to build a highly accurate classifier.

 Instead, we are studying a two-stage learning process.

 First, we use unlabelled images to train a model on an artificially constructed task. This is called a **pretext task**.

 The model learns features while solving that artificial task.

 Then we take the learned feature extractor, transfer it to our real classification problem, attach a new classification head, and fine-tune the model using a relatively small labelled dataset.

 Finally, we compare that model against a conventional model trained from scratch.

 So the practical is really an experiment about **representation learning and transfer learning**.

---

 # 1\. The overall idea of the practical

 Let me start with the overall workflow, because understanding this makes the rest of the notebook much easier.

 We have two datasets.

 The first is a relatively large **source dataset** for which we do not use the labels.

 The second is a smaller **target dataset** for which labels are available.

 The source dataset is used to learn general visual representations.

 The target dataset is used to solve the actual classification problem.

 The process therefore has three major stages.

 First, we construct a self-supervised or pretext task from the unlabelled source images.

 Second, we train a neural network to solve that pretext task. During this process, the network learns a feature representation.

 Third, we transfer that feature representation to the target classification problem and fine-tune it using the labelled target data.

 We then compare the result with a baseline model that was trained from scratch using only the labelled target data.

 So when thinking about the entire practical, keep this question in mind:

 **Does pre-training on unlabelled data produce a better starting representation for the target task than starting from random weights?**

 That is the main experimental question.

---

 # 2\. What is self-supervised learning?

 Before going through the notebook, it is important to understand what self-supervised learning actually means.

 In ordinary supervised learning, we have an input and a human-provided label.

 For example, an image might be labelled as a dog.

 The model receives the image, makes a prediction, compares that prediction with the human-provided label, and updates its parameters.

 Self-supervised learning changes the source of the target.

 We still create a learning problem, but instead of asking a human to provide the label, we generate the target automatically from the data itself.

 For example, suppose I take an image and rotate it by 90 degrees.

 Because I performed the rotation myself, I automatically know that the answer is 90 degrees.

 So I can train a neural network to predict which rotation was applied.

 There was no human annotation involved.

 The original image was unlabelled.

 The transformation that I applied created the training signal.

 This is why the approach is called self-supervised.

 The data supervises the learning process by providing a signal that we construct automatically.

---

 # 3\. Why do we do this?

 The obvious question is:

 Why bother doing all of this?

 Why not simply train the classifier directly?

 The reason is that labelled data can be expensive or difficult to obtain, whereas unlabelled images are often much easier to collect.

 Imagine that we have a relatively small collection of labelled images for our actual task, but a much larger collection of unlabelled images from a related visual domain.

 The unlabelled images cannot directly tell us which target class each image belongs to.

 However, they can still contain useful information about visual structure.

 For example, they contain edges, textures, shapes, object parts, spatial relationships, and other visual patterns.

 A neural network can potentially learn these general features before we ever ask it to solve the final classification problem.

 The hope is that when we eventually give the network the smaller labelled dataset, it does not have to learn everything from scratch.

 Instead, it starts with a useful visual representation and only needs to adapt that representation to the target task.

---

 # 4\. The notebook's dataset configuration

 The notebook provides a CIFAR-based configuration.

 The source dataset is **CIFAR-10**.

 The target dataset is constructed from **CIFAR-100**.

 This is an important design choice.

 The source and target datasets do not need to have exactly the same classes.

 In fact, CIFAR-10 and CIFAR-100 have different class structures.

 The important idea is that the images can still share useful visual properties.

 For example, a model learning from images of animals in the source dataset can potentially learn features such as edges, shapes, textures, body parts, and spatial arrangements.

 Those features may also be useful when the target task contains different animal categories.

 So this is not about transferring the source class labels.

 It is about transferring the **representation learned from the source images**.

---

 # 5\. Constructing the target dataset

 The notebook then creates a smaller target classification problem from CIFAR-100.

 CIFAR-100 contains many fine-grained classes, and the notebook groups these into larger categories.

 Each selected category contains five CIFAR-100 classes.

 For example, if we select a category containing large carnivores, the target classification problem might consist of bear, leopard, lion, tiger, and wolf.

 The important thing is that the original CIFAR-100 class identifiers are not necessarily convenient for the new five-class problem.

 Therefore the notebook creates a mapping from the original class indices to new consecutive target labels.

 Instead of having arbitrary original indices, the selected classes become class zero, class one, class two, class three, and class four.

 This is important because the classification head will eventually produce five outputs, one for each target class.

---

 # 6\. Creating the unlabelled source dataset

 Now we turn our attention to CIFAR-10.

 The notebook deliberately prevents us from using the CIFAR-10 class labels for the self-supervised experiment.

 Conceptually, we are changing the source data from:

 "image plus class label"

 to simply:

 "image".

 This is important because otherwise we would be doing ordinary supervised pre-training rather than the intended self-supervised experiment.

 The model is therefore not allowed to learn that a particular image is an airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.

 Instead, we have to create a new learning problem using only the images themselves.

---

 # 7\. The custom dataset classes

 The notebook defines custom dataset classes to control how the images and labels are returned.

 The unlabelled dataset returns an image without a human-provided target.

 The labelled dataset returns an image together with its target class.

 The `__len__` method tells PyTorch how many examples are available.

 The `__getitem__` method specifies what should be returned for a particular index.

 The notebook also checks whether an image is already represented as a PIL image and converts it when necessary.

 This matters because many of the torchvision transformations operate naturally on PIL images.

 The labelled dataset also checks that the number of images matches the number of targets.

 That is a simple but useful consistency check.

 If we had 5,000 images but only 4,900 labels, something would clearly be wrong.

---

 # 8\. Splitting the target dataset

 The target dataset needs to be divided into training and validation data.

 The training data is used to update the neural network.

 The validation data is kept separate so that we can measure how well the model generalises to examples it did not use for parameter updates.

 The notebook uses a validation fraction of approximately ten percent.

 The split is controlled using a random seed.

 The reason for the seed is reproducibility.

 If we run the notebook again with the same seed, we can reproduce the same random split, assuming the other sources of randomness are controlled appropriately.

 This is important for experimental work because otherwise changes in performance might simply be caused by different random splits.

---

 # 9\. DataLoaders and batches

 Once the datasets have been constructed, the notebook uses PyTorch `DataLoader` objects.

 A DataLoader is responsible for providing examples to the model in batches.

 Rather than processing one image at a time, we might process, for example, 128 images together.

 The training DataLoader uses shuffling.

 That means that the order of the training examples is changed between epochs.

 This helps prevent the model from relying on a particular ordering of the training examples.

 The validation DataLoader does not need this randomisation, so validation is performed with `shuffle=False`.

---

 # 10\. The collate functions

 The notebook also uses custom `collate_fn` functions.

 A collate function controls how individual examples are assembled into a batch.

 This is particularly useful here because we want to apply transformations to the images and then stack them into tensors.

 For example, the notebook transforms each image individually and then uses `torch.stack` to combine the resulting tensors into one batch tensor.

 For classification, the targets are converted to `torch.long`.

 This is important because PyTorch's `CrossEntropyLoss` expects class targets represented as integer class indices.

 The tensors are also moved to the selected computation device.

 That device may be a CUDA-enabled GPU if one is available, otherwise the notebook falls back to the CPU.

---

 # 11\. Preprocessing and augmentation

 The notebook then defines image preprocessing and augmentation.

 Some transformations are used simply to convert the images into tensors and normalise their values.

 Other transformations are random augmentations.

 Examples include random resized crops, horizontal flips, colour jitter, grayscale conversion, and random erasing.

 The purpose of augmentation is to expose the model to different versions of the same underlying visual content.

 This encourages the network to learn features that are more robust rather than simply memorising the exact pixels of the training images.

 However, there is an important warning for self-supervised learning.

 An augmentation must not destroy the information that defines the pretext task.

 Suppose our pretext task is to predict image rotation.

 If we deliberately rotate an image and call that rotation the target, but then apply another random rotation without accounting for it, we may change the very signal that the model is supposed to predict.

 The target could say one thing while the actual image presented to the model contains another transformation.

 Therefore, augmentation has to be designed around the pretext task.

---

 # 12\. The model architecture

 Now we reach the neural network itself.

 The notebook separates the model into two conceptual components.

 The first component is the **backbone**.

 The second component is the **task-specific head**.

 This separation is fundamental to the entire practical.

 The backbone is responsible for extracting features from the image.

 The head takes those features and turns them into predictions for a particular task.

 The same backbone can therefore potentially be used for several different tasks.

 For example, during self-supervised pre-training, the backbone could feed into a four-class rotation head.

 Later, that same backbone could feed into a five-class target classification head.

 The head changes because the task changes.

 The backbone is the component whose learned representation we want to reuse.

---

 # 13\. Understanding the convolutional backbone

 The `ConvBackbone` contains several convolutional layers.

 The first convolution takes the three RGB channels and produces 32 feature channels.

 Later convolutional layers progressively increase the number of feature channels.

 The network therefore moves from relatively low-level image information towards increasingly rich feature representations.

 Pooling operations reduce the spatial dimensions of the feature maps.

 The network eventually reaches 256 feature channels.

 Then adaptive average pooling reduces the spatial dimensions to one by one.

 This is useful because it gives us a fixed-size representation regardless of the exact spatial dimensions at that stage.

 The resulting representation is then passed through fully connected layers.

 The dimensions are reduced progressively until the final backbone representation has 64 values.

 So the important conceptual result is this:

 An image enters the backbone, and the backbone transforms it into a **64-dimensional feature representation**.

 Those 64 values are not human-readable labels.

 They are learned numerical features.

---

 # 14\. Why separate the backbone from the head?

 This is worth emphasising because it is the central architectural idea.

 Imagine that the backbone has learned useful information about shapes, textures, spatial structure, and object appearance.

 We do not want to throw that information away simply because the original task has finished.

 Instead, we remove the old task-specific head and attach a new head.

 During pretext training, the head might answer:

 "What rotation was applied?"

 During downstream training, the new head might answer:

 "Which of these five target classes is this image?"

 The backbone provides the features.

 The head interprets those features according to the current task.

 That is what makes transfer learning possible.

---

 # 15\. The classification head

 The target classification head receives the 64-dimensional representation produced by the backbone.

 It first projects those 64 features into a smaller 32-dimensional representation.

 A ReLU activation is then applied.

 Finally, a linear layer produces one output for each target class.

 If there are five target classes, the final output contains five logits.

 These are raw prediction scores.

 They are not explicitly converted into probabilities by the model.

 This is deliberate because the training loss is `CrossEntropyLoss`, which is designed to work directly with logits.

---

 # 16\. Understanding logits and CrossEntropyLoss

 This is an important PyTorch concept.

 Suppose the target task has five classes.

 For one image, the model might produce five values such as 1.2, 0.4, 3.8, 0.7, and 0.1.

 These are logits.

 The largest value is associated with the model's predicted class.

 So in this example, the third class would be selected because its logit is the largest.

 During training, `CrossEntropyLoss` compares these logits with the correct integer class index.

 The target therefore needs to be represented as a long integer.

 For a batch of 128 images and five classes, the prediction tensor would have a shape of 128 by 5.

 The target tensor would have a shape of 128.

 Each target value identifies the correct class for one image.

---

 # 17\. The baseline experiment

 Before we can determine whether self-supervised learning helps, we need a baseline.

 The baseline answers a simple question:

 **How well can this architecture perform if we train it directly on the labelled target dataset from random initialisation?**

 The notebook therefore creates a new backbone with randomly initialised parameters.

 It attaches the target classification head.

 Then it trains this complete model using the labelled target training data.

 After each epoch, the model is evaluated on the target validation set.

 The best validation accuracy gives us a reference point.

 For example, imagine that the baseline achieves 47 percent validation accuracy.

 That number is not the final answer.

 It is the reference against which the self-supervised approach will be compared.

---

 # 18\. Understanding the training loop

 The training loop is standard supervised deep-learning code.

 At the beginning of a training epoch, the model is put into training mode using `model.train()`.

 This matters because certain layers, such as dropout, behave differently during training.

 For each batch, the optimizer's existing gradients are cleared using `opt.zero_grad()`.

 The images are then passed through the model.

 That is the forward pass.

 The model produces predictions.

 The loss function compares those predictions with the correct targets.

 Then `loss.backward()` performs backpropagation.

 This calculates gradients telling us how the trainable parameters should change in order to reduce the loss.

 Finally, `opt.step()` updates the parameters using those gradients.

 So the essential training sequence is:

 clear the old gradients, perform the forward pass, calculate the loss, calculate gradients, and update the parameters.

---

 # 19\. AdamW and regularisation

 The notebook uses the AdamW optimiser.

 The optimiser controls how the model parameters are updated using the gradients.

 The notebook also includes an L2-style regularisation term through weight decay.

 The purpose is to discourage excessively large weights and potentially reduce overfitting.

 This is particularly relevant because the target dataset is relatively small compared with the model's capacity.

 A model can sometimes fit the training data very well while failing to generalise to unseen validation examples.

 Regularisation is one mechanism for controlling this behaviour.

---

 # 20\. Training mode versus evaluation mode

 When the model is being trained, we use:

 `model.train()`.

 When we evaluate the model, we use:

 `model.eval()`.

 Evaluation mode is important because layers such as dropout behave differently.

 During training, dropout randomly removes activations.

 During evaluation, dropout is disabled so that the model behaves deterministically according to its learned parameters.

 The notebook also uses `torch.no_grad()` during evaluation.

 This tells PyTorch that we do not need to calculate gradients.

 That reduces unnecessary computation and memory usage because we are only measuring performance rather than updating the model.

---

 # 21\. Measuring accuracy

 The notebook uses top-one accuracy.

 The model produces a set of logits for each image.

 We use `argmax` to identify the class with the largest logit.

 Then we compare that predicted class with the true class.

 The proportion of correct predictions gives us the accuracy.

 So if 80 out of 100 validation images are classified correctly, the validation accuracy is 80 percent.

 The notebook records both losses and accuracies during training so that we can examine how learning progresses.

---

 # 22\. Training metrics and visualisation

 The `TrainingMetrics` class records information from the training process.

 Training loss and training accuracy are recorded during training.

 Validation metrics are recorded during validation.

 The notebook can then plot these values.

 These plots help us understand whether the model is learning, overfitting, or converging.

 Training curves can be noisy because measurements are collected at the batch level.

 The notebook therefore applies smoothing to the training curves so that the overall trend is easier to see.

 Validation measurements provide a less noisy indication of how the model is performing on unseen data.

 The notebook also keeps track of the best validation accuracy and the best validation loss.

---

 # 23\. Early stopping

 The notebook includes an early-stopping mechanism.

 The basic idea is that we do not necessarily want to continue training indefinitely.

 If validation loss starts to deteriorate, continuing to train may make the model increasingly specialised to the training data.

 The supplied code examines recent validation losses and compares the current loss with an average of recent previous losses.

 If the current validation loss becomes worse according to that condition, training can be stopped.

 This is another mechanism intended to reduce unnecessary training and limit overfitting.

---

 # 24\. The pretext task

 Now we reach the most important creative part of the practical.

 We have an unlabelled source dataset.

 We need to invent a task that the network can learn without human-provided labels.

 This is the **pretext task**.

 One straightforward example is rotation prediction.

 We take an image and randomly select one of four rotations:

 zero degrees, ninety degrees, one hundred and eighty degrees, or two hundred and seventy degrees.

 Because we know which rotation we applied, we automatically know the correct target.

 The neural network is then trained to predict that rotation.

 This transforms the unlabelled source dataset into a self-supervised learning problem.

---

 # 25\. Why rotation prediction can be useful

 The purpose is not really to build a useful rotation classifier.

 The rotation classifier is only the mechanism through which the backbone learns.

 To predict rotation successfully, the model may need to learn information about object orientation, shape, spatial relationships, and visual structure.

 Those learned features may then be useful for another task.

 This is the central idea behind the practical.

 We are using an artificial task to encourage the network to learn a representation that may transfer to a real task.

---

 # 26\. The quality of the pretext task

 An important lesson is that a pretext task should not simply be easy.

 Suppose we create a task that can be solved using an extremely simple pixel-level statistic.

 The model might achieve very high pretext accuracy.

 But it may not learn useful semantic or structural features.

 In that situation, the pretext task has been solved successfully, but the representation may not transfer well.

 This is why pretext accuracy should not be treated as the ultimate objective.

 The real question is:

 **Does the learned representation improve performance on the downstream target task?**

 A pretext task with slightly lower accuracy can potentially produce a more useful representation than a task that reaches near-perfect accuracy for trivial reasons.

---

 # 27\. Other possible pretext tasks

 Rotation prediction is only one possibility.

 The notebook discusses other approaches.

 One possibility is relative-position prediction.

 For example, we could take two patches from an image and ask the model to determine their spatial relationship.

 Another possibility is a jigsaw-style task, where image patches are shuffled and the model has to determine how they should be arranged.

 We can also construct tasks involving different augmented views of an image.

 For example, we could ask whether two image crops originated from the same original image.

 These tasks all follow the same fundamental principle:

 We create a learning signal automatically from the image rather than relying on a human-provided class label.

---

 # 28\. Contrastive learning

 The notebook also introduces the broader idea of contrastive learning.

 Contrastive learning focuses on relationships between examples or different views of examples.

 The general intuition is that representations of related views should have an appropriate relationship to one another, while representations of unrelated examples should be distinguishable.

 The notebook mentions simpler approaches such as pairwise contrastive loss and triplet loss.

 A full SimCLR implementation is more complicated because it involves multiple augmented views, a projection head, contrastive objectives, and careful batch construction.

 For a practical focused on understanding the fundamental concepts, a simpler pretext task can therefore be more manageable.

---

 # 29\. Training the pretext model

 Once we have selected a pretext task, we construct a model consisting of the backbone and a pretext-specific head.

 For rotation prediction, the head has four outputs.

 The source images are transformed to generate the artificial rotation labels.

 The model is then trained using the source dataset.

 The backbone starts learning visual representations.

 The pretext head learns how to translate those representations into predictions for the artificial task.

 At this point, the model is not yet solving the final target classification problem.

 It is learning a representation that we hope will be useful later.

---

 # 30\. What happens after pre-training?

 Once pre-training is complete, we no longer need the pretext head.

 The pretext head was designed specifically to answer the artificial question.

 For example, it learned to distinguish zero, ninety, one hundred and eighty, and two hundred and seventy degree rotations.

 Those four outputs are not relevant to the target classification problem.

 What we want to preserve is the backbone.

 The backbone now contains parameters that were learned from the large source dataset.

 We can save those parameters using the model's `state_dict`.

 Later, we load those parameters into a new target backbone.

 This is the transfer-learning stage.

---

 # 31\. Fine-tuning on the target task

 We now return to the labelled target dataset.

 We take the pretrained backbone and attach a new target classification head.

 If our target problem contains five classes, the new head produces five outputs.

 The input to the head is still the 64-dimensional representation produced by the backbone.

 The important difference is that the backbone is no longer randomly initialised.

 It starts from the parameters learned during self-supervised pre-training.

 We then train the model using the labelled target dataset.

 This process is called **fine-tuning**.

 The model is adapting a previously learned representation to the specific target problem.

---

 # 32\. Why use a smaller learning rate?

 Fine-tuning can often benefit from a smaller learning rate than training from scratch.

 The reason is that the pretrained backbone already contains useful information.

 If we make very large parameter updates, we risk changing those learned features too aggressively.

 A smaller learning rate allows the model to adapt gradually.

 The exact learning rate is an experimental choice, however.

 The practical is not simply telling us that one particular learning rate is universally correct.

 We should evaluate the effect of our choices experimentally.

---

 # 33\. Freezing layers

 Another option is to freeze some of the pretrained layers.

 When a layer is frozen, its parameters are not updated during fine-tuning.

 This can preserve the pretrained representation while allowing later layers and the classification head to adapt to the target task.

 For example, we might freeze early feature-extraction layers while training the later layers and target head.

 However, freezing is another experimental choice.

 The practical is ultimately about measuring what happens when the pretrained representation is transferred to the target problem.

---

 # 34\. The importance of a fair baseline

 Now we have two experiments.

 The first is the baseline.

 The model starts from random parameters and learns directly from the labelled target dataset.

 The second is the self-supervised experiment.

 The backbone first learns from the unlabelled source dataset.

 Then that backbone is transferred to the target task and fine-tuned.

 To interpret the comparison properly, we want the experiments to be as comparable as reasonably possible.

 For example, we should consider the augmentation strategy, architecture, target data, evaluation procedure, and other training choices.

 Otherwise, an observed difference could be caused by something other than self-supervised pre-training.

 The objective is to isolate the effect of the pretrained representation as much as possible.

---

 # 35\. The final experiment

 At the end of the notebook, we compare the downstream validation performance.

 Suppose the baseline achieves 47 percent validation accuracy.

 Suppose the self-supervised model achieves 55 percent.

 The difference between those two numbers is the observed downstream improvement in that particular experiment.

 But the important point is not simply the numerical difference.

 We need to interpret what happened.

 Did the pretext task learn useful features?

 Did the model overfit?

 Did the augmentation help?

 Did the pretext task become too easy?

 Did the choice of learning rate matter?

 Did freezing layers help or hurt?

 These questions form part of the experimental analysis.

---

 # 36\. Why pretext accuracy is not the final objective

 This is probably the single most important conceptual point to remember.

 Imagine one pretext task reaches 95 percent accuracy.

 Another reaches only 70 percent.

 We cannot automatically conclude that the first task produced a better representation.

 The first task might simply have been easier.

 The second task might have forced the network to learn richer features that transfer better.

 Therefore there are two different measurements.

 The first is **pretext performance**.

 That tells us how well the model learned the artificial task.

 The second is **downstream performance**.

 That tells us whether the learned representation is actually useful for the target problem.

 For this practical, downstream validation performance is the key experimental outcome.

---

 # 37\. Understanding overfitting

 The notebook also gives us an opportunity to observe overfitting.

 Suppose training accuracy continues increasing while validation accuracy stops improving or begins to decline.

 That suggests that the model is becoming increasingly specialised to the training data without improving its ability to generalise.

 This can happen particularly easily when the labelled target dataset is small.

 The model has a substantial number of parameters, while the amount of labelled data is limited.

 Regularisation, augmentation, early stopping, and transfer learning are all techniques that can influence this behaviour.

---

 # 38\. Why self-supervised learning might fail

 It is also important to understand that self-supervised learning is not guaranteed to improve the target task.

 There are several possible reasons.

 The pretext task might be too easy.

 It might encourage the model to learn features that are unrelated to the target task.

 The augmentation might destroy important information.

 The source and target datasets might not share enough useful visual structure.

 The fine-tuning procedure might overwrite useful pretrained features.

 Or the baseline might already be strong enough that the additional pre-training provides little benefit.

 A good experiment therefore does not assume success.

 It measures what actually happens.

---

 # 39\. Interpreting the training curves

 When we look at the training plots, we should ask several questions.

 First, is the training loss decreasing?

 If it is not, the model may not be learning effectively.

 Second, is validation performance improving?

 If training performance improves while validation performance deteriorates, overfitting may be occurring.

 Third, does the pretext task converge extremely quickly?

 If so, perhaps the task is too easy.

 Fourth, after transfer, does the target model start from a better position than the randomly initialised baseline?

 Finally, does the final downstream validation accuracy justify the claim that the learned representation was useful?

 The plots therefore provide evidence for interpreting the experiment rather than simply serving as decoration.

---

 # 40\. A note about the notebook's implementation details

 When working through the notebook, there may also be small implementation issues that need attention.

 For example, the supplied example may refer to a variable such as `img_dim` in a transformation even though the earlier notebook code defines `img_size`.

 If that variable has not been defined elsewhere, running the cell will produce a `NameError`.

 The important lesson is that understanding the code is more useful than blindly executing it.

 When an error occurs, we should identify what the code expects, check the variables that have actually been defined, and correct the inconsistency.

 Similarly, placeholders such as the target category are intentionally provided for the student to complete.

---

 # 41\. What the notebook is teaching beyond the code

 Although there is a lot of PyTorch code in the notebook, the practical is teaching several broader machine-learning ideas.

 The first is **representation learning**.

 Instead of directly learning the final prediction, we learn a useful intermediate representation.

 The second is **self-supervised learning**.

 We generate training signals from unlabelled data.

 The third is **transfer learning**.

 We take knowledge learned on one problem and reuse it on another.

 The fourth is **fine-tuning**.

 We adapt pretrained parameters to the target task.

 The fifth is **experimental design**.

 We compare a baseline against an alternative approach and try to determine whether the difference is actually caused by the proposed method.

---

 # 42\. The practical in terms of information flow

 Let me describe the entire process again without referring to the notebook implementation.

 We begin with a large collection of unlabelled source images.

 We invent a task that can generate its own targets.

 For example, we rotate each image and ask the network to predict the rotation.

 We train a convolutional backbone together with a pretext head.

 While solving the pretext problem, the backbone learns visual features.

 Once pre-training is complete, we remove the pretext head.

 We keep the pretrained backbone.

 We then take our smaller labelled target dataset.

 We attach a new classification head designed specifically for the target classes.

 We fine-tune the network using the target labels.

 Finally, we evaluate it on the target validation set.

 Then we compare that result with a model that was trained directly on the target data from random initialisation.

 That comparison tells us whether the self-supervised pre-training was useful in this particular experiment.

---

 # 43\. What each major notebook component is doing

 It is useful to associate each major piece of code with its purpose.

 The dataset classes define how images and labels are represented.

 The dataset-splitting code creates training and validation subsets.

 The DataLoaders organise examples into batches.

 The transformations perform preprocessing and augmentation.

 The backbone extracts features.

 The classification head converts features into task-specific predictions.

 The training function performs optimisation.

 The evaluation function measures validation performance.

 The metrics class records what happened during training.

 The plotting functions help us understand the learning behaviour.

 The pretext dataset and pretext head create the self-supervised learning problem.

 The transfer-learning code moves the pretrained representation into the downstream task.

 And the final comparison tells us whether the entire idea helped.

---

 # 44\. The three most important words

 If I had to reduce the practical to three words, they would be:

 **Pretext. Transfer. Fine-tune.**

 Pretext means:

 We create an artificial task so that we can learn from unlabelled images.

 Transfer means:

 We keep the learned feature representation and move it to another task.

 Fine-tune means:

 We adapt that representation using labelled examples from the actual target problem.

 Remembering those three stages makes the entire notebook much easier to understand.

---

 # 45\. What should be discussed in the final report?

 The practical expects more than simply reporting a final accuracy.

 We should explain what pretext task we selected.

 We should explain why we selected it.

 We should explain how the artificial targets were generated.

 We should describe the architecture and the augmentations.

 We should report the training behaviour.

 We should report the downstream validation performance.

 And importantly, we should discuss what happened.

 If the self-supervised model improved performance, we should explain what evidence supports that conclusion.

 If it did not improve performance, that is also a useful experimental result.

 We can discuss possible reasons.

 Perhaps the pretext task was not sufficiently related to the target problem.

 Perhaps it was too easy.

 Perhaps the model overfit.

 Perhaps the source and target domains were not sufficiently aligned.

 The discussion should be based on evidence from the experiment rather than assuming that self-supervised learning must work.

---

 # 46\. The key distinction between source and target labels

 One final distinction is especially important.

 The source dataset does not provide the labels used for the final classification task.

 The artificial pretext labels are generated from transformations or relationships that we construct.

 The target labels are genuine labels for the actual downstream classification problem.

 Therefore there are effectively two different kinds of supervision involved.

 During pre-training, the supervision is automatically generated.

 During fine-tuning, the supervision comes from the actual labelled target dataset.

 That is why we can describe the first stage as self-supervised and the second stage as supervised fine-tuning.

---

 # 47\. The complete experiment in plain language

 If I had to explain the practical to someone who had never seen the notebook, I would describe it like this:

 We have many images without labels.

 Instead of throwing those images away, we create an artificial problem that the images themselves can provide the answers to.

 We train a convolutional neural network to solve that artificial problem.

 While doing so, the network learns a representation of visual information.

 We then remove the part of the network that was specific to the artificial problem.

 We keep the feature extractor.

 Next, we give the model a smaller labelled dataset for the real classification problem.

 We attach a new classifier and fine-tune the model.

 Finally, we compare its validation accuracy with a model that never received the self-supervised pre-training.

 If the pretrained model performs better, that provides evidence that the representation learned from the unlabelled source data was useful for the downstream task.

 That is the central experiment.

---

 # 48\. Final takeaway

 The most important thing to understand from this practical is that the goal is not simply to train another image classifier.

 The practical is investigating **how useful representations can be learned before the final task is known**.

 The source images provide the raw visual information.

 The pretext task provides an automatically generated learning signal.

 The backbone learns the representation.

 The pretext head solves the temporary artificial problem.

 The target head solves the actual classification problem.

 And the comparison against a baseline tells us whether the representation learned without human labels was useful.

 So the complete conceptual sequence is:

 We start with unlabelled images.

 We construct a pretext task.

 We pre-train the backbone.

 We discard the pretext head.

 We transfer the backbone.

 We attach a new target head.

 We fine-tune using labelled target data.

 And finally, we evaluate the target validation performance.

 That is the story of Practical 3.

 The code in the notebook exists to implement and measure each stage of that story.

 The most important experimental question is therefore not:

 **"Did the model solve the pretext task?"**

 It is:

 **"Did learning from the unlabelled source data produce a representation that helps the model solve the downstream target task?"**

 That is what the entire practical is designed to investigate.
