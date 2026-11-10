Absolutely\! I’ll use both Chunk 1 and Chunk 2 of Practical 6: Meta-learning as the basis for a question-and-answer-focused teaching narrative script, designed for students who can hear the instructor but cannot see them.

The script will be written as a spoken teaching session, rather than as lecture notes or a summary. It will guide students through the practical by asking questions, allowing them to think about the answers, and then explaining the concepts and code step by step.

I don't have the previous Practical 1 example visible in the current conversation context, so I’ll follow the intended Q&A teaching format.

# Practical 6: Meta-learning

## Question-and-answer-focused narrative teaching script

Instructor narration | Audio-first delivery | Based on Chunks 1 and 2

### Part 1. Introduction to meta-learning

Instructor:

Welcome to Practical 6. In this practical, we are going to explore an exciting area of machine learning called meta-learning.

Let's begin with a question.

Question 1: What is meta-learning, and how is it different from conventional machine learning?

In conventional machine learning, we train a model using a dataset so that it learns to perform a particular task. For example, we might train an image classifier to recognise different types of animals.

But what happens when we introduce new categories that the model has never encountered before?

Normally, we would need to collect additional labelled examples and train or fine-tune the model.

Meta-learning takes a different approach.

Its central idea is to teach a model how to learn new tasks quickly, using only a small number of examples.

This is why meta-learning is often described as learning to learn.

Question 2: What do we mean by few-shot learning?

Few-shot learning is a setting in which a model must learn to recognise new classes from only a few labelled examples.

For instance, imagine that we give a model five pictures of a new animal category. Can it learn to recognise other pictures of that animal using those five examples?

That is the type of challenge we are investigating.

Question 3: What is the difference between meta-training and meta-testing?

Meta-training involves repeatedly presenting the model with different small learning tasks so that it learns a useful way of adapting.

Meta-testing evaluates whether the model can apply what it has learned to completely new tasks, involving classes it did not encounter during meta-training.

This distinction is very important.

We are not simply trying to achieve high accuracy on the original training classes. We want the model to generalise to new classes.

### Part 2. Understanding few-shot learning tasks

Instructor:

Before we start implementing meta-learning algorithms, we need to understand how a few-shot task is constructed.

Question 4: What is an N-way K-shot classification task?

Let's break this down.

- N-way tells us how many different classes are included in a task.
- K-shot tells us how many labelled support examples we have for each class.

For example, a five-way one-shot task contains five classes and one labelled support example per class.

A five-way five-shot task contains five classes and five labelled support examples per class.

Question 5: What are the support set and query set?

Each few-shot task is divided into two parts.

The support set contains the labelled examples that the model uses to learn the task.

The query set contains additional examples used to evaluate how well the model has learned to classify the task's classes.

Think of the support set as the examples we learn from and the query set as the examples we must correctly classify.

Question 6: Why do we separate support and query sets?

Because we want to measure whether the model can generalise beyond the examples it was given.

If we evaluated the model on exactly the same examples it used for learning, we would not get a reliable indication of its few-shot learning ability.

### Part 3. Establishing a supervised baseline

Instructor:

Now let's look at the first major part of our practical.

Before implementing meta-learning, we need a baseline model.

Question 7: Why do we need a baseline?

A baseline gives us a reference point.

We can compare the performance of our meta-learning algorithms against a conventional supervised learning approach.

In this practical, we train a convolutional neural network on all the available meta-training classes.

The backbone extracts image features, while the classification head predicts the class labels.

Let's examine the code.

We initialise the number of training epochs to ten, the learning rate to one thousandth, and the L2 regularisation coefficient to one thousandth.

The model consists of two components:

1. `ConvBackbone()`, which extracts features from images.
2. `ClassifierHead()`, which converts those features into class predictions.

We combine these using `nn.Sequential()`.

Question 8: Why do we use top-five accuracy for baseline training?

Because the baseline is trained on a relatively large number of classes.

Top-five accuracy measures whether the correct class appears among the five highest-scoring predictions.

This is different from top-one accuracy, which only considers whether the highest-scoring prediction is correct.

We train the model and save its parameters so that we can reuse the trained backbone later.

Question 9: Does good baseline validation accuracy mean that we have solved the few-shot learning problem?

No.

This is an important distinction.

The baseline validation accuracy measures performance on the conventional classification problem.

Our actual objective is to determine how well the model adapts to new few-shot tasks.

That is why we subsequently perform meta-evaluation.

### Part 4. Fine-tuning on individual tasks

Instructor:

Now let's explore how we evaluate a pretrained model on a few-shot task.

We introduce a function called `finetune_backbone_on_task()`.

Question 10: What does this function do?

It takes a backbone and a particular few-shot task.

It creates a new classification head based on the number of classes in that task.

It then combines the backbone and classification head into a new model.

The model is trained using the support set and evaluated using the query set.

Notice that we have several hyperparameters:

- `FT_EPOCHS`, which controls the maximum number of fine-tuning epochs.
- `FT_LR`, which controls the learning rate.
- `FT_L2`, which controls weight decay.
- `FT_PATIENCE`, which controls the early-stopping patience.

Question 11: Why do we sometimes freeze the backbone?

Freezing the backbone means that its parameters are not updated during fine-tuning.

This is useful when we want to preserve previously learned image features and train only the new classification head.

For our baseline evaluation, we freeze the backbone because we want to measure how effectively its pretrained representations support new classification tasks.

Question 12: What does meta-evaluation measure?

The `meta_evaluate()` function evaluates the model across the collection of meta-test tasks.

For each task, it creates a fresh backbone and loads the evaluated model's parameters.

It then fine-tunes the model on that task's support set and evaluates it on the query set.

Finally, it calculates the average support and query losses and accuracies.

The most important measurement for our practical is the average meta-test query accuracy.

Why?

Because it tells us how successfully the model classifies previously unseen examples after adapting to a new task.

### Part 5. Reptile meta-learning

Instructor:

We now move to our first meta-learning algorithm: Reptile.

Question 13: What is Reptile trying to achieve?

Reptile is designed to find model parameters that can be adapted quickly to different tasks.

Instead of training on the entire dataset in the conventional way, we repeatedly sample small few-shot tasks.

For each task, the model is fine-tuned.

Then, the original model parameters are moved towards the parameters obtained after fine-tuning.

Let's examine the main steps.

First, we initialise the Reptile backbone.

We also define:

- A maximum of five thousand meta-training episodes.
- A meta-evaluation interval of two hundred episodes.
- A meta step size of 0.5.

Question 14: What is a meta-training episode?

One episode represents one sampled few-shot task.

During each episode, we:

1. Sample a task from the meta-training classes.
2. Fine-tune the backbone using the task's support set.
3. Compare the updated parameters with the original parameters.
4. Apply the Reptile update.
5. Record the task's query performance.

Question 15: How does the Reptile update work?

This is the central idea.

Suppose our original model parameters are represented by theta.

After fine-tuning on a particular task, we obtain updated parameters represented by theta prime.

Reptile moves the original parameters towards the updated parameters.

Mathematically:

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
+
\epsilon(\theta'-\theta_{\text{old}})
$$

Here, epsilon represents the meta step size.

In our implementation, it is set to 0.5.

This means that the model moves halfway towards the task-adapted parameters.

Notice that this is different from conventional gradient descent.

We are not simply updating the model using the gradient of one large training dataset. We are using the results of task-specific learning to guide the overall model.

Question 16: Why do we evaluate Reptile periodically?

Because we want to monitor whether meta-training is improving generalisation to unseen tasks.

Every two hundred episodes, the model is evaluated on the meta-test tasks.

If the query accuracy improves, we save a checkpoint.

At the end, we load the best checkpoint and evaluate it again.

Question 17: What did we learn from Reptile?

Reptile provides a useful demonstration of gradient-based meta-learning.

However, in this practical, we find that it may struggle to outperform the supervised baseline.

This motivates us to investigate another approach: Prototypical Networks.

### Part 6. Introducing Prototypical Networks

Instructor:

Now we arrive at the first major implementation task in Chunk 2.

We are going to implement a Prototypical Network, commonly called ProtoNet.

Question 18: What is the main idea behind ProtoNet?

ProtoNet uses the concept of prototypes.

A prototype is the average feature representation of all the support examples belonging to a particular class.

Instead of using a conventional linear classification head, ProtoNet classifies query examples according to their distances from these class prototypes.

Imagine that we have a five-way few-shot task.

We first pass the support images through the convolutional backbone.

This produces feature embeddings.

We then calculate one prototype for each of the five classes.

Finally, we pass the query images through the same backbone and compare their embeddings with the prototypes.

The nearest prototype determines the predicted class.

Question 19: What is an embedding?

An embedding is a numerical representation of an input image.

The convolutional backbone transforms the original image into a feature vector.

Ideally, images belonging to the same class have similar embeddings, while images belonging to different classes have more distinguishable embeddings.

This makes distance-based classification possible.

### Part 7. Implementing the ProtoNet class

Instructor:

Let's examine our implementation.

We begin by defining the `ProtoNet` class, which inherits from `nn.Module`.

Its constructor receives two arguments:

- A backbone feature extractor.
- A temperature parameter.

The temperature controls the scale of the classification logits.

Now let's look at the first method.

Question 20: What does `make_prototypes()` do?

It calculates the prototype for each class in the support set.

First, we extract the support embeddings:

python

```python
support_embeddings = self.backbone(support_x)
```

Next, we identify the unique class labels.

For each class, we select the corresponding embeddings and calculate their mean.

The result is one prototype per class.

This is the key operation:

python

```python
support_embeddings[support_y == class_label].mean(dim=0)
```

The condition selects embeddings belonging to the current class.

The mean operation averages those embeddings to create the prototype.

We then use `torch.stack()` to combine all the prototypes into one tensor.

Question 21: What does `predict_query()` do?

This method predicts the class of each query example.

First, the query images are passed through the backbone to generate embeddings.

Next, we calculate their distances from the class prototypes.

We use:

python

```python
torch.cdist(query_embeddings, prototypes, p=2)
```

This calculates Euclidean distances between the query embeddings and the prototypes.

We then square those distances.

Finally, we convert the distances into classification logits:

python

```python
logits = -distances / self.temperature
```

Why do we use negative distances?

Because the softmax function assigns higher probabilities to larger logits.

By taking the negative distance, a query example receives a higher score for a closer prototype.

Question 22: What is the role of temperature?

Temperature controls the scale of the logits.

A larger temperature reduces the differences between the logits, producing a softer probability distribution.

A smaller temperature produces a sharper distribution.

For this practical, the default temperature is one.

### Part 8. Understanding the forward method

Instructor:

We have implemented the two main operations.

Now we need to connect them through the `forward()` method.

Question 23: What happens when we call the ProtoNet model?

The model receives two inputs:

- The support set.
- The query set.

The support images and labels are passed to `make_prototypes()`.

This generates the class prototypes.

The query images are then passed to `predict_query()` together with those prototypes.

The method returns the classification logits.

Notice something important.

The ProtoNet does not need a separate learnable classification head for each task.

The prototypes are calculated directly from the support embeddings.

This is what allows the model to classify new classes without having to create and train a conventional classifier for every new task.

### Part 9. Evaluating ProtoNet

Instructor:

After implementing ProtoNet, we need to evaluate it on the meta-test tasks.

This is the purpose of `meta_evaluate_protonet()`.

Question 24: How is ProtoNet evaluation different from Reptile evaluation?

With Reptile, we fine-tune a model on the support set and evaluate it on the query set.

With ProtoNet, we calculate the prototypes directly from the support set.

We then classify the query set using distances to those prototypes.

There is no separate support-set fine-tuning process during evaluation.

We calculate cross-entropy loss and top-one accuracy for each task.

Finally, we average the query losses and accuracies across all meta-test tasks.

This gives us a measure of ProtoNet's few-shot generalisation performance.

### Part 10. Training ProtoNet

Instructor:

We now reach Task 2: implementing the ProtoNet training loop.

This is where we train the backbone to produce more useful embeddings for few-shot classification.

Question 25: What is different about ProtoNet meta-training compared with conventional supervised training?

In conventional supervised training, we usually sample batches directly from a dataset.

In ProtoNet meta-training, we sample entire few-shot tasks.

Each episode contains both a support set and a query set.

The support set is used to calculate prototypes.

The query set is used to calculate the classification loss.

This is important because we want to optimise the embedding space for few-shot classification.

Question 26: What optimiser do we use?

We use AdamW.

Our initial learning rate is:

python

```python
meta_lr = 1e-3
```

And the weight decay is:

python

```python
meta_l2 = 1e-3
```

We initialise the ProtoNet using a new ConvBackbone.

We then wrap this backbone inside the ProtoNet class.

Question 27: What are the steps in one ProtoNet meta-training episode?

Let's go through them carefully.

First, sample a new task:

python

```python
metatrain_task = create_task(
    num_support=SUPPORT_SIZE,
    num_query=QUERY_SIZE
)
```

Second, retrieve the support and query examples.

We move both sets to the appropriate computing device.

Third, switch the model into training mode:

python

```python
proto_model.train()
```

Fourth, clear the previous gradients:

python

```python
proto_opt.zero_grad()
```

Fifth, perform the forward pass:

python

```python
train_query_logits = proto_model(
    (support_x, support_y),
    (query_x, query_y)
)
```

The model calculates prototypes from the support set and predicts the query labels.

Sixth, calculate the cross-entropy loss:

python

```python
train_query_loss = F.cross_entropy(
    train_query_logits,
    query_y
)
```

Seventh, calculate query accuracy.

Eighth, backpropagate:

python

```python
train_query_loss.backward()
```

Finally, update the model parameters:

python

```python
proto_opt.step()
```

And that completes one meta-training episode.

Question 28: Why do we calculate the loss using the query set rather than the support set?

This is a fundamental question.

The support set is used to construct the class prototypes.

The query set measures whether those prototypes are useful for classifying additional examples.

By minimising the query classification loss, the backbone learns embeddings that make it easier to recognise new examples from a small support set.

This is the central learning mechanism of ProtoNet.

### Part 11. Monitoring training and saving checkpoints

Instructor:

During meta-training, we record the query loss and query accuracy.

We also periodically evaluate the model on the meta-test tasks.

In our implementation, evaluation takes place every five hundred episodes.

Question 29: Why do we save the best ProtoNet checkpoint?

Because the model's performance may fluctuate during training.

The model at the final episode is not necessarily the best model.

We therefore save the model whenever its meta-test query accuracy exceeds the previous best.

At the end of training, we reload the best checkpoint.

We then perform the final meta-evaluation.

Question 30: What should we compare in our final assessment?

We compare three approaches:

1. The conventional supervised baseline.
2. Reptile.
3. ProtoNet.

The most important metric is the average meta-test query accuracy.

We should also inspect the training plots to understand how the models improve over time.

### Part 12. Final assessment and reflection

Instructor:

We have now completed the main stages of Practical 6.

Let's finish with a few reflection questions.

Question 31: What is the most important difference between Reptile and ProtoNet?

Reptile learns an initialisation that can be adapted to different tasks through fine-tuning.

ProtoNet learns an embedding space in which new classes can be recognised using class prototypes calculated from a small support set.

Both are meta-learning approaches, but they use different mechanisms.

Question 32: Why might ProtoNet outperform the supervised baseline?

The supervised baseline learns to distinguish the original training classes.

ProtoNet explicitly optimises its feature representations for classification using small support sets.

This can make it more effective when encountering new classes with very few labelled examples.

However, better performance is not guaranteed. It depends on the dataset, architecture, training procedure and hyperparameters.

Question 33: What challenges might we encounter during this practical?

Some common challenges include:

- Ensuring that support and query labels use consistent class indices.
- Making sure that prototype tensors have the correct dimensions.
- Ensuring that the query logits match the dimensions expected by cross-entropy loss.
- Choosing suitable learning rates and numbers of meta-training episodes.
- Avoiding overfitting to the sampled meta-training tasks.

Question 34: What is the key lesson from this practical?

The most important lesson is that machine learning does not always have to involve learning one fixed task.

Instead, we can train models to learn how to adapt to new tasks.

Meta-learning provides a framework for this, and few-shot learning is one of its important applications.

ProtoNet demonstrates how a carefully learned feature space can allow a model to recognise previously unseen classes using only a handful of labelled examples.

Instructor – closing remarks:

Well done. You have now explored the fundamentals of meta-learning, established a supervised baseline, implemented Reptile, and built and trained a Prototypical Network.

For your final submission, make sure you include your training plots, report the actual meta-test query accuracies for all three approaches, and briefly discuss the challenges you encountered.

Remember: the objective is not simply to achieve high classification accuracy. It is to build a model that can learn new tasks quickly and generalise from very few examples.

Thank you, and well done on completing Practical 6\!

Note for the instructor: The questions above are written as spoken prompts. When delivering the practical, pause briefly after each question to give students time to think before revealing the explanation. Since students cannot see the instructor, describe important code operations, tensor transformations, and comparisons verbally rather than relying on pointing to cells, highlighting code, or referring to visual positions on the screen.
