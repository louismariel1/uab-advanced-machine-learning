 ## Transfer Learning Practical — Presentation Speech

 Good morning. In this practical, I worked on a transfer-learning problem using convolutional neural networks, with the main objective of improving classification performance when the target dataset is relatively small.

 The central question behind the practical is quite simple: **if I do not have a lot of labelled data for my target problem, can I use knowledge learned from another, larger dataset to improve my model?**

 To investigate this, I compare two approaches. First, I train a CNN completely from scratch using only the target data. This gives me a baseline. Then, I pre-train the same CNN on the larger CIFAR-10 dataset and transfer the learned knowledge to the target task. Finally, I fine-tune the transferred model and compare its validation accuracy with the baseline.

---

 ## 1\. Understanding the target dataset

 The target domain comes from CIFAR-100.

 CIFAR-100 contains 100 fine-grained classes organized into 20 broader superclasses. Each superclass contains five classes.

 For this practical, I select one superclass. For example, if I select a vehicle-related superclass, it may contain five classes such as bicycle, bus, motorcycle, pickup truck, and train.

 Each CIFAR-100 class contains 500 images, so selecting one superclass gives me:

 - 5 target classes
- 500 images per class
- 2,500 images in total

 I then split these images into training and validation data. With an 80/20 split, I have 2,000 training images and 500 validation images.

 This is deliberately a relatively small target dataset.

 That is important because the practical is trying to simulate a common machine-learning situation: we may have a new classification problem, but we don't have enough labelled data to train a powerful model from scratch reliably.

---

 ## 2\. Selecting and preparing the target classes

 The first part of the code identifies the five classes belonging to my selected superclass.

 The original CIFAR-100 labels are not necessarily consecutive numbers. For example, my five classes could have original labels such as 4, 8, 19, 31, and 55.

 However, my target classifier only needs five output neurons.

 Therefore, I remap the original labels to:

```
0, 1, 2, 3, 4
```

 This is called **label remapping**.

 The reason for doing this is that the classifier operates over the five target classes, so it is much cleaner to represent those classes using consecutive indices.

 The code first finds the original CIFAR-100 indices belonging to my selected superclass. Then it creates a subset containing only those images.

 So at this stage, I have transformed the original CIFAR-100 dataset into a smaller five-class target dataset.

---

 ## 3\. Preparing the images

 The images are CIFAR images, so each image has a size of 32 by 32 pixels and three colour channels: red, green, and blue.

 Therefore, the input to the CNN has the shape:

```
3 × 32 × 32
```

 I use `ToTensor()` to convert the images into PyTorch tensors.

 I also normalize the RGB channels.

 Normalization helps put the input values into a more suitable numerical range for neural-network training and can make optimization more stable.

 The data are then provided to the network through a `DataLoader`.

 The `DataLoader` is responsible for dividing the dataset into batches. In this practical, the batch size is 128.

 For training, the data are shuffled so that the model does not repeatedly see the examples in exactly the same order.

---

 ## 4\. The CNN architecture

 The model used in the practical is called `BasicCNN`.

 The architecture starts with the input image:

```
3 × 32 × 32
```

 The first convolution changes the number of channels from 3 to 32.

 This convolution learns visual patterns from the input image.

 After the convolution, I apply ReLU.

 ReLU, or Rectified Linear Unit, is defined as:

```
ReLU(x) = max(0, x)
```

 Its purpose is to introduce non-linearity into the network. Without non-linear activation functions, stacking multiple linear operations would still essentially behave like one linear transformation.

 After ReLU, I apply max pooling.

 The first pooling operation reduces the spatial dimensions from:

```
32 × 32
```

 to:

```
16 × 16
```

 The second convolution increases the number of feature maps from 32 to 64.

 Again, I apply ReLU and max pooling.

 The second pooling operation reduces the spatial dimensions from:

```
16 × 16
```

 to:

```
8 × 8
```

 So immediately before flattening, I have:

```
64 channels × 8 × 8
```

 which gives:

```
64 × 8 × 8 = 4096
```

 Therefore, the first fully connected layer receives 4096 inputs.

 The network then reduces this representation to 256 neurons, followed by another fully connected layer producing 128 features.

 Finally, there is a classifier.

 For the target problem, the classifier is:

```
128 → 5
```

 because there are five target classes.

 The final five numbers are called **logits**. They represent the model's scores for the five possible classes.

---

 # 5\. Establishing the baseline

 Before doing any transfer learning, I need a baseline.

 The baseline answers the question:

 **How well can this CNN perform if I train it from scratch using only the target data?**

 So I create a new `BasicCNN` with five output classes.

 The model starts with randomly initialized parameters.

 I train it using:

 - Adam optimizer
- learning rate of `1e-3`
- weight decay of `1e-3`
- batch size of 128
- 20 epochs
- cross-entropy loss

 During each training iteration, the process is:

```
Input images
     ↓
Forward pass
     ↓
Predicted logits
     ↓
Cross-entropy loss
     ↓
Backpropagation
     ↓
Parameter update
```

 More specifically, `opt.zero_grad()` clears the gradients from the previous iteration.

 Then the model performs a forward pass.

 The predictions are compared with the true labels using cross-entropy loss.

 After that, `backward()` calculates the gradients using backpropagation.

 Finally, `optimizer.step()` uses those gradients to update the model parameters.

 This process is repeated for every batch.

 Once all batches have been processed, one epoch has been completed.

---

 # 6\. Understanding the loss and predictions

 The model produces logits rather than final class labels.

 For example, it might produce something like:

```
[1.2, -0.4, 3.8, 0.7, 0.2]
```

 The largest value is 3.8, which corresponds to class 2.

 Therefore:

```
pred.argmax(axis=1)
```

 selects the predicted class.

 The loss function is cross-entropy.

 Cross-entropy is appropriate for a multi-class classification problem because it compares the model's predicted class scores with the correct class labels.

 Importantly, I provide the raw logits to `CrossEntropyLoss`. I don't need to manually apply softmax first because PyTorch's cross-entropy implementation internally handles the appropriate log-softmax operation.

---

 # 7\. Training versus validation

 An important part of the practical is that I don't just look at training accuracy.

 After every epoch, I evaluate the model on the validation set.

 During validation, I use:

```
model.eval()
```

 This puts the model into evaluation mode.

 I also use:

```
torch.no_grad()
```

 because I am not updating the model during validation, so there is no need to calculate or store gradients.

 This makes validation more efficient.

 The validation accuracy tells me how well the model generalizes to images that were not used for parameter updates.

 I record the validation accuracy after every epoch and keep the best validation accuracy.

 This best validation accuracy becomes my baseline performance.

---

 # 8\. Why overfitting is a concern

 At this point, there is an important problem.

 I am using a CNN with a reasonable number of parameters, but I only have 2,000 target training images.

 Because the model has considerable capacity, it can start memorizing patterns specific to the training data.

 For example, I might see the training accuracy continue increasing while the validation accuracy stops improving or even decreases.

 That pattern indicates **overfitting**.

 The model is becoming very good at the training examples, but it is not necessarily learning representations that generalize well to new examples.

 This is where transfer learning becomes useful.

 Instead of asking the model to learn all visual features from only 2,000 target images, I can first expose it to a much larger dataset.

---

 # 9\. Moving to the source domain

 The source domain in this practical is CIFAR-10.

 CIFAR-10 contains 50,000 training images across ten classes.

 This is considerably more training data than the target task provides.

 I create another instance of `BasicCNN`, but this time it has ten output classes:

```
BasicCNN(num_classes=10)
```

 The reason for ten outputs is simple: CIFAR-10 has ten classes.

 I then train this model on CIFAR-10.

 This stage is called **pre-training**.

 The purpose of pre-training is not to solve my final target problem directly.

 Instead, I want the CNN to learn useful visual representations from the larger source dataset.

---

 # 10\. What does the model actually learn?

 This is one of the most important concepts in the practical.

 The model is not simply memorizing the names of CIFAR-10 classes.

 The convolutional layers learn visual patterns.

 For example, early layers can learn relatively general features such as:

 - edges
- corners
- colour transitions
- textures
- simple shapes

 Deeper layers can combine these into more complex structures and patterns.

 These representations can still be useful for a different classification problem.

 For example, even if a particular target class was never present in CIFAR-10, the model may already know how to recognize useful shapes, textures, edges, and structures that occur in images belonging to that target class.

 Therefore, the source and target classes do **not** need to be identical for transfer learning to provide a benefit.

---

 # 11\. Saving the pretrained model

 After pre-training on CIFAR-10, I save the learned parameters using:

```
torch.save(pt_model.state_dict(), 'pretrained_model.ckpt')
```

 The `state_dict` contains the model's learned parameter values.

 This gives me a checkpoint of the pretrained network.

 At this point, I have a CNN that has learned visual representations from the source domain.

 Now I want to transfer those representations to my target task.

---

 # 12\. Fine-tuning the target model

 The next stage is **fine-tuning**.

 I create a model with the same architecture used during pre-training and load the saved parameters.

 Initially, the pretrained model still has a CIFAR-10 classifier:

```
128 → 10
```

 But my target problem only has five classes.

 So I cannot simply keep the CIFAR-10 classifier.

 The output space is different.

 Instead, I replace the final classifier with:

```
128 → 5
```

 In code, this is essentially:

```
ft_model.classifier = nn.Linear(128, 5)
```

 This is one of the most important implementation steps in the entire practical.

 The convolutional layers contain the transferable visual knowledge.

 The final classifier is much more specific to the original task because it maps those features to CIFAR-10's ten classes.

 Therefore, I replace the source classifier with a new classifier designed for the five target classes.

 The new classifier starts with newly initialized parameters and must learn how the transferred features relate to the target labels.

---

 # 13\. Why use a smaller learning rate?

 For fine-tuning, the learning rate is reduced from:

```
1e-3
```

 to:

```
1e-4
```

 The reason is that the pretrained model already contains useful information.

 If I use very large parameter updates, I risk destroying some of the useful representations learned during pre-training.

 A smaller learning rate allows the network to make more gradual adjustments.

 In other words, I am not starting from zero anymore.

 I am starting from a model that already knows something useful, and I want to adapt that knowledge rather than completely overwrite it.

 This is sometimes described as avoiding **catastrophic forgetting** of useful pretrained representations.

---

 # 14\. Are the convolutional layers frozen?

 In this particular practical, the pretrained layers are not explicitly frozen.

 The optimizer receives the parameters of the entire fine-tuned model.

 Therefore, both the pretrained feature layers and the new classifier can be updated.

 This is genuine fine-tuning.

 An alternative strategy would be to freeze some of the convolutional layers and train only the new classifier.

 That can sometimes be useful, especially when the source and target domains are very similar or when the target dataset is extremely small.

 But this practical allows the whole network to adapt.

---

 # 15\. Comparing the two approaches

 Now I have two experiments.

 The first is:

```
Target data
    ↓
CNN initialized randomly
    ↓
Train from scratch
    ↓
Baseline validation accuracy
```

 The second is:

```
CIFAR-10
    ↓
Pre-train CNN
    ↓
Save pretrained parameters
    ↓
Replace 10-class classifier with 5-class classifier
    ↓
Fine-tune using target data
    ↓
Transfer-learning validation accuracy
```

 The important comparison is between the two validation accuracies.

 For example, suppose the baseline reaches 55% validation accuracy and the fine-tuned model reaches 63%.

 The improvement is:

```
63% - 55% = 8 percentage points
```

 So transfer learning has improved the target validation accuracy by eight percentage points.

 The practical expects an improvement of roughly five to ten percentage points, although the exact result can depend on which superclass is selected and on training conditions.

---

 # 16\. Why transfer learning can improve generalization

 The key reason is the amount of information available during representation learning.

 The baseline has only around 2,000 target training images.

 The pretrained model has first seen 50,000 CIFAR-10 training images.

 Therefore, the pretrained CNN has had a much larger opportunity to learn useful visual structures.

 When I fine-tune it, I am not asking it to discover everything from scratch.

 Instead, I am starting with an already useful feature representation and adapting it to the target task.

 This can make learning more data-efficient and can improve generalization when the target dataset is small.

---

 # 17\. But transfer learning is not guaranteed to work

 There is an important limitation.

 Transfer learning does not automatically guarantee better performance.

 The source and target domains can be different.

 This difference is called **domain shift**.

 If the source and target tasks are sufficiently related, the transferred features can be useful.

 However, if the source and target domains are very different, the transferred representations may provide little benefit or even make the target performance worse.

 This situation is known as **negative transfer**.

 In this practical, CIFAR-10 and CIFAR-100 are both small RGB natural-image datasets with the same image resolution, so there is a reasonable expectation that visual features learned from CIFAR-10 will be useful for CIFAR-100.

---

 # 18\. Low-level versus high-level features

 Another important transfer-learning concept is that not all layers are equally transferable.

 The earlier convolutional layers tend to learn relatively general features.

 For example:

```
edges
↓
textures
↓
simple shapes
```

 These can often transfer well between different image datasets.

 The deeper layers tend to become more task-specific.

 They may represent more complex structures associated with the source task.

 The final classifier is generally the most task-specific component because it explicitly maps the learned representation to the source classes.

 That is why replacing the final classifier is such a common transfer-learning strategy.

---

 # 19\. What I would look for in the learning curves

 The plots are also important because they tell me how the models behave during training.

 For the baseline, I might see training accuracy continue increasing while validation accuracy eventually plateaus.

 That would indicate overfitting.

 For the fine-tuned model, I would ideally see the validation accuracy improve more quickly and reach a higher value.

 The goal is not simply to achieve high training accuracy.

 The actual objective of the practical is to maximize **validation accuracy**, because validation performance gives us an estimate of how well the model generalizes to unseen target examples.

---

 # 20\. Important PyTorch concepts demonstrated

 This practical also demonstrates several fundamental PyTorch concepts.

 First, `DataLoader` handles batches of data.

 Second, `model.train()` puts the network into training mode, while `model.eval()` puts it into evaluation mode.

 Third, `torch.no_grad()` prevents unnecessary gradient computation during evaluation.

 Fourth, `backward()` performs backpropagation and calculates gradients.

 Fifth, `optimizer.step()` updates the model parameters.

 Sixth, `state_dict()` allows us to save and reload model parameters.

 Finally, the model and data must be placed on compatible devices, such as the CPU or GPU.

 For example, if CUDA is available, the model and tensors can be moved to the GPU using `.to(device)`.

---

 # 21\. What I learned from the practical

 The main lesson from this practical is that training a neural network is not always about starting from random initialization and giving it as much target data as possible.

 When target data are limited, we can exploit knowledge learned from another dataset.

 The source dataset acts as a way of providing prior visual knowledge.

 The model first learns general representations from the source domain.

 Then, the final classifier is adapted to the target classes, and the whole network can be fine-tuned using the smaller target dataset.

 So the practical demonstrates a very important machine-learning principle:

 **knowledge learned from one task can sometimes be reused to improve learning on another task.**

---

 # 22\. Final summary

 To summarize the complete practical, the workflow is:

```
                 CIFAR-100
              Target dataset
                    │
                    ↓
            Select one superclass
                    │
                    ↓
              Five classes
                    │
                    ↓
              Split 80/20
                    │
                    ↓
          ┌───────────────────┐
          │ Baseline CNN      │
          │ Train from scratch│
          └─────────┬─────────┘
                    │
                    ↓
          Baseline validation
               accuracy

                 CIFAR-10
              Source dataset
                    │
                    ↓
             Pre-train CNN
                    │
                    ↓
            Save checkpoint
                    │
                    ↓
          Load pretrained model
                    │
                    ↓
        Replace 10-class classifier
                    │
                    ↓
          New 5-class classifier
                    │
                    ↓
          Fine-tune on target
                    │
                    ↓
       Transfer-learning validation
               accuracy
                    │
                    ↓
          Compare both results
```

 So there are really two different learning strategies being compared.

 The baseline says:

 **"Learn everything from the small target dataset."**

 Transfer learning says:

 **"First learn useful visual representations from a larger source dataset, then adapt those representations to the small target task."**

 The key distinction I would remember for the exam is:

 > **Pre-training learns reusable representations from the source domain, while fine-tuning adapts those representations to the target task.**

 And the most important implementation detail is:

 > **The pretrained CIFAR-10 network has a ten-class classifier, so after loading the pretrained weights, the final classifier must be replaced with a five-class classifier for the CIFAR-100 target task.**

 Overall, the practical demonstrates that transfer learning can improve generalization when labelled target data are limited, because the model does not have to learn all useful visual representations from scratch.
