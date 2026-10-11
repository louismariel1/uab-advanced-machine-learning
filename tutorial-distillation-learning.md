# Advanced Machine Learning Tutorial: Knowledge Distillation

We can extend the existing cat-versus-dog example to explain distillation learning, also called knowledge distillation.

Knowledge distillation is closely related to student–teacher learning, which we covered in the previous tutorial. In fact, knowledge distillation is one of the most common forms of student–teacher learning. It focuses on transferring knowledge from a teacher model to a student model, often so that the student can be smaller, faster, or cheaper to deploy.

We will keep the same practical example throughout: a neural network that recognizes whether a photograph contains a cat or a dog.

## 1. The main idea

Imagine that we have trained a large neural network to recognize cats and dogs.

The model performs very well, but it has millions of parameters and requires considerable computing power. We want to deploy it on a mobile phone or a small embedded device.

Instead of training a smaller model from scratch, we use the large model to teach the smaller one.

Meet the cats of House & Garden: 14 of the finest felines featured in our pages | House & Garden

Input: Photograph of an animal

The same image is passed to both models during training.

Teacher model

Large backbone and classification head

Cat: 80%

Dog: 20%

Provides learning targets

Student model

Smaller backbone and classification head

Cat: 78%

Dog: 22%

Learns to imitate the teacher

The teacher predicts that the image is a cat with 80% probability and a dog with 20% probability. The student learns to produce a similar probability distribution.

The goal is not necessarily to make the student identical to the teacher. It is to transfer useful behavior while allowing the student to use a more efficient architecture.

- Teacher: Provides knowledge.
- Student: Learns from the teacher.
- Distillation loss: Measures how different the student's predictions are from the teacher's predictions.
- Student deployment: The student can be used independently after training; the teacher is often no longer needed for inference.

## 2. A small numerical example

Let's follow the same cat-versus-dog classification example and see exactly how distillation works.

### Step 1: The teacher makes a prediction

Suppose a large pretrained neural network receives a photograph and produces two scores, called logits:

$$
\mathbf{z}_T=
\begin{bmatrix}
2.0\\
0.5
\end{bmatrix}
$$

The first score corresponds to cat, and the second to dog.

Applying the softmax function converts these logits into probabilities:

$$
P(c)=\frac{e^{z_c}}{\sum_j e^{z_j}}
$$

For the cat:

$$
P_T(\text{cat})=
\frac{e^{2.0}}{e^{2.0}+e^{0.5}}
\approx0.818
$$

For the dog:

$$
P_T(\text{dog})\approx0.182
$$

The teacher's prediction is therefore approximately:

- Cat: 81.8%
- Dog: 18.2%

### Step 2: The student makes its own prediction

Now suppose the student has a smaller backbone and a new classification head. Its initial logits are:

$$
\mathbf{z}_S=
\begin{bmatrix}
1.0\\
0.8
\end{bmatrix}
$$

After applying softmax, the student's probabilities are approximately:

- Cat: 55.0%
- Dog: 45.0%

The student predicts cat, just like the teacher, but its probability distribution is different.

A conventional classification loss could teach the student using the correct label, cat. Distillation provides an additional signal: the student should learn to match the teacher's distribution.

### Step 3: Compare the teacher and student

| Class | Teacher | Student before training |
| --- | --- | --- |
| Cat | 81.8% | 55.0% |
| Dog | 18.2% | 45.0% |

The student assigns too little probability to cat and too much to dog compared with the teacher.

During training, we adjust the student's weights to reduce this difference.

For example, after training, the student might produce:

| Class | Teacher | Student after training |
| --- | --- | --- |
| Cat | 81.8% | 79.0% |
| Dog | 18.2% | 21.0% |

These final student values are illustrative rather than the result of an actual training run.

The student has learned to imitate the teacher more closely, while retaining its own parameters and architecture.

### Step 4: Calculate the distillation loss

A common way to measure the difference between teacher and student predictions is the Kullback–Leibler (KL) divergence:

$$
\mathcal{L}_{KD}
=
\sum_c P_T(c)
\log\frac{P_T(c)}{P_S(c)}
$$

Here:

- $P_T(c)$ is the teacher's probability for class $c$.
- $P_S(c)$ is the student's probability for class $c$.
- $c$ represents each class, in this example cat and dog.

Using the teacher's probabilities and the student's initial predictions:

$$
\begin{aligned}
\mathcal{L}_{KD}
={}&0.818\log\frac{0.818}{0.550}\\
&+0.182\log\frac{0.182}{0.450}
\end{aligned}
$$

The result is approximately:

$$
\mathcal{L}_{KD}\approx0.169
$$

This positive value measures the difference between the distributions. Training the student to minimize this loss encourages its predictions to approach the teacher's predictions.

Notice that the loss compares entire probability distributions, not just the final predicted classes.

### Step 5: Update the student's weights

During training:

1. The teacher processes the training image and produces target predictions.
2. The student processes the image and produces its own predictions.
3. The distillation loss measures the difference.
4. Backpropagation updates the student's trainable weights to reduce the loss.
5. The process repeats over many training examples.

In ordinary knowledge distillation, the teacher's weights remain fixed while the student learns.

The student can eventually reproduce much of the teacher's behavior without needing the same number of parameters.

## 3. Why is distillation more informative than simply copying the correct label?

Consider two animal photographs.

For the first photograph, the teacher predicts:

- Cat: 99%
- Dog: 1%

For the second photograph, the teacher predicts:

- Cat: 60%
- Dog: 40%

Suppose both images are correctly labeled as cats.

A conventional classification target says only that both images belong to the cat class. The teacher's predictions provide additional information: the first image appears much more clearly cat-like to the teacher, while the second is more ambiguous.

This information is called soft-target information.

A one-hot target for a cat is:

$$
\mathbf{y}=[1,0]
$$

The teacher's prediction might instead be:

$$
\mathbf{p}_T=[0.818,0.182]
$$

The teacher's distribution contains information about its relative preference for the two classes. With more classes, the pattern of probabilities assigned to alternative classes can provide even richer information.

However, these probabilities are not guaranteed to represent true uncertainty perfectly. A teacher can be overconfident or incorrect, so distillation quality depends on the teacher and training setup.

## 4. What is the temperature parameter?

In practice, knowledge distillation often uses a temperature $T$ in the softmax function:

$$
P(c)=
\frac{\exp(z_c/T)}
{\sum_j\exp(z_j/T)}
$$

When $T=1$, this is the ordinary softmax function.

When $T>1$, the probability distribution generally becomes softer, meaning the differences between the class probabilities become less pronounced.

Let's examine the teacher's original logits:

$$
\mathbf{z}_T=[2.0,0.5]
$$

## Explore the temperature

Adjust the temperature to see how the teacher's probability distribution changes. These calculations use the same illustrative logits.

Temperature (T)

## 1.0

0.5

1

2

4

Cat

### 81.8%

Dog

### 18.2%

At T = 1, the probabilities are the ordinary softmax probabilities.

A higher temperature can reveal more about the teacher's relative preferences between classes. It is particularly useful in multiclass problems, where the teacher's relative scores for alternative classes may contain useful information.

A common distillation objective is:

$$
\mathcal{L}_{KD}
=
T^2 D_{KL}
\left(
P_T^{(T)}\|P_S^{(T)}
\right)
$$

Here, $P_T^{(T)}$ and $P_S^{(T)}$ are the teacher and student distributions computed using the same temperature $T$. The $T^2$ factor is a common scaling convention that helps account for the effect of temperature on gradients.

In practice, the temperature is a hyperparameter chosen during validation. A higher temperature is not automatically better.

# 5. How does distillation relate to the backbone and head?

The original tutorial separated a neural network into a backbone and a head. We can now extend that picture to include the teacher and student.

## Teacher network

Input image

Large backbone

Extracts rich visual features

Teacher classification head

Teacher probabilities

Cat: 81.8% · Dog: 18.2%

## Student network

Same input image

Smaller backbone

Learns a more efficient representation

Student classification head

Student probabilities

Cat: 79% · Dog: 21%

There are several ways to perform distillation:

- Output distillation: The student learns to match the teacher's final predictions.
- Feature distillation: The student learns to match selected intermediate representations from the teacher's backbone.
- Attention distillation: The student learns to reproduce selected attention patterns or attention maps.
- Combined distillation: The student learns from teacher outputs and intermediate representations, potentially alongside ground-truth labels.

The simplest approach is output distillation, which is what we used in the numerical example.

## 5.1 Does the student need the same backbone?

No. This is one of the most useful aspects of distillation.

For example:

| Component | Teacher | Student |
| --- | --- | --- |
| Architecture | ResNet-101 | MobileNet |
| Backbone | Large | Lightweight |
| Classification head | Cat/dog head | Cat/dog head |
| Training guidance | Ground-truth labels | Teacher outputs, optionally labels |
| Deployment goal | High predictive capability | Lower computational cost |

The student may have fewer layers, fewer parameters, or a more efficient architecture.

If feature distillation is used, the teacher's and student's feature dimensions may differ. An additional projection layer or another alignment mechanism can help make their representations comparable.

An important distinction: the student does not necessarily copy the teacher's backbone weights. In ordinary distillation, it learns its own parameters by optimizing an objective derived from the teacher's behavior.

## 5.2 What happens after training?

Once training is complete, the student can be deployed on its own.

The teacher is generally needed during distillation training but not during student inference.

This is why distillation can reduce the cost of deployment: each new prediction requires only the student to run, rather than both models.

However, a smaller model does not automatically achieve the teacher's accuracy. The trade-off between model size, latency, and predictive quality must be measured experimentally.

# 6. Compare distillation with the other learning methods

Now we can distinguish all five approaches from our extended tutorial.

| Method | What is learned or combined? | Main training idea |
| --- | --- | --- |
| Contrastive learning | Representations | Learn relationships between examples |
| Student–teacher learning | Knowledge from another model | A student learns from a teacher |
| Knowledge distillation | A teacher's behavior or representations | Optimize a student using distillation targets |
| Ensemble learning | Predictions from multiple models | Aggregate model outputs |
| Boosting | A sequence of predictive learners | Add learners to improve the current model |

Student–teacher learning and distillation overlap substantially: distillation is a specific and widely used way to transfer knowledge from teacher to student. Not every student–teacher method is conventional knowledge distillation, however; some self-supervised methods use evolving teachers and different representation-learning objectives.

The methods can also be combined. For example, we might contrastively pretrain a large teacher, train it on labeled cat-versus-dog images, distill it into a smaller student, and then evaluate whether combining several students offers any additional benefit.

# 7. Check your understanding

## Mini-quiz

0 of 4 answered

1. A large model predicts Cat: 80% and Dog: 20%. A smaller model is trained to reproduce those probabilities. What is the primary technique?

Knowledge distillation

Hard voting

Boosting

2. During ordinary knowledge distillation, which model's weights are typically updated by the distillation loss?

Only the teacher

Only the student

Both teacher and student equally

3. Why can teacher probabilities contain more information than the label 'cat'?

They always guarantee a correct prediction

They describe the teacher's relative preferences among classes

They eliminate the need for training data

4. After distillation training, which model would typically be deployed on a resource-constrained mobile phone?

The teacher and student together

The teacher only

The student only

 Check answers

# Final takeaway

Knowledge distillation is best understood as training a smaller or different model to learn from a more established model.

Using our cat-versus-dog example:

- The teacher uses its backbone and head to make predictions.
- The student receives the same image and makes its own predictions.
- A distillation loss compares the teacher's and student's outputs.
- Backpropagation updates the student to reproduce useful aspects of the teacher's behavior.
- After training, the student can make predictions independently.

The central insight is that the teacher provides more than an answer. Its output distribution can provide a richer learning signal than a class label alone.

This makes distillation particularly valuable when we want to retain much of a powerful model's predictive capability while reducing the computational cost of using it.
