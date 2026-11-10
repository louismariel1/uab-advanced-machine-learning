# Advanced Machine Learning – Practical 5: Knowledge Distillation

## Question-and-Answer Focused Narrative Script

Narrator's script for audio delivery

Teaching approach: This script is designed for students who can hear the narrator but cannot see the screen. Therefore, all explanations are self-contained, questions are spoken clearly, pauses are provided for students to think, and references to notebook sections, code, plots, and outputs are explained verbally.

The script follows the practical notebook in sequence, guiding students through the concepts, implementation tasks, expected observations, and assessment.

# Part 1: Introduction to Knowledge Distillation

### Question 1: What is knowledge distillation, and why do we need it?

Welcome to Practical 5 of Advanced Machine Learning.

In this practical, we are going to explore an important technique in deep learning called Knowledge Distillation.

Let's begin with a question.

Suppose we have two neural networks. The first is a large, powerful neural network with many parameters. The second is a much smaller neural network with fewer parameters.

Which model would you expect to perform better?

Pause for a moment and think about your answer.

The likely answer is the larger model because it has greater capacity to learn complex patterns from data.

However, larger models also have disadvantages. They require more memory, more computational resources, and potentially more time to train and deploy.

Now, imagine that we want the smaller model to perform almost as well as the larger model.

Is there a way for the smaller model to benefit from what the larger model has already learned?

The answer is yes.

This is the main idea behind knowledge distillation.

Knowledge distillation is a model compression and knowledge transfer technique in which a larger neural network, called the teacher, transfers useful information to a smaller neural network, called the student.

The objective is to improve the performance of the smaller student model without increasing its architecture or computational requirements significantly.

In simple terms, we want the student to learn not only from the correct answers but also from the knowledge and predictions of the teacher.

### Question 2: What are the three stages of this practical?

There are three important stages in this practical.

First, we train a small neural network independently, without any teacher supervision.

This is our independent student model. It establishes the baseline performance of a small model.

Second, we train a larger neural network using the same training data.

This is our teacher model. We expect it to achieve better accuracy because it has a larger architecture and greater learning capacity.

Third, we train another small neural network using knowledge transferred from the teacher.

This is our distilled student model.

Our objective is to determine whether the distilled student performs better than the independently trained student.

Ideally, the distilled student should approach the performance of the teacher while retaining the smaller architecture.

Let's remember these three models:

- Independent student: the small model trained normally.
- Teacher: the larger model trained normally.
- Distilled student: the small model trained using information from the teacher.

Keep these three models in mind throughout the practical because we will compare their performance at the end.

# Part 2: Setting Up the Environment and Dataset

### Question 3: Which dataset are we using, and why?

In this practical, we are using the CIFAR-10 dataset.

CIFAR-10 is a popular image classification dataset containing ten different image categories.

Can you name some of these categories?

Pause and think.

The ten categories are:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Each image is a small colour image with a resolution of thirty-two by thirty-two pixels.

Why are we using CIFAR-10?

The dataset is relatively small and manageable, but it is sufficiently complex to demonstrate the differences between a teacher and a student model.

### Question 4: Why are we reducing the size of the dataset?

An interesting decision in this practical is that we do not use the entire CIFAR-10 dataset.

Instead, we use approximately thirty per cent of the original dataset for training, twenty per cent for validation, and discard the remaining fifty per cent for this experiment.

Why do you think we are doing this?

Pause for a moment.

The main reason is to make the experiment faster and to make the effects of knowledge distillation more apparent.

Knowledge distillation can be particularly useful when training data is limited.

By reducing the training dataset, we create a situation in which transferring additional information from the teacher may help the student learn more effectively.

The training data is used to update the model parameters.

The validation data is used to evaluate how well the models generalise to examples they have not used for training.

Remember that validation accuracy is especially important because it allows us to compare the three models fairly.

### Question 5: Why do we inspect the training images?

The notebook includes a function called `inspect_data`.

This function selects a number of images from the training dataset and displays them alongside their class labels.

Why is this useful?

It allows us to verify that the dataset has been downloaded correctly and that the images and their corresponding labels are sensible.

For example, an image of a horse should have the label horse, rather than airplane or automobile.

Understanding the dataset is an important first step before training any machine learning model.

# Part 3: Understanding the Teacher and Student Architectures

### Question 6: What neural network architecture are we using?

In this practical, both the teacher and student are based on a convolutional neural network architecture called ResNet.

ResNet stands for Residual Network.

ResNet uses residual connections, which allow information to bypass certain layers through shortcut connections.

These connections help train deeper neural networks by improving the flow of information and gradients through the network.

The notebook defines a model called `BasicResnet`.

This model consists of three main components.

First, a convolutional backbone extracts features from the input images.

Second, an adaptive average pooling layer reduces the spatial dimensions of the extracted feature maps.

Third, a classification layer produces predictions for the ten CIFAR-10 classes.

### Question 7: What is the difference between the teacher and student architectures?

Let's compare their architectures.

The independent student uses the following convolutional layer sizes:

Thirty-two, sixty-four, and one hundred and twenty-eight channels.

The teacher uses:

Sixty-four, one hundred and twenty-eight, one hundred and twenty-eight, two hundred and fifty-six, and two hundred and fifty-six channels.

What does this tell us?

The teacher has a deeper and wider architecture.

It therefore has significantly more parameters than the student.

In fact, the notebook indicates that the teacher has nearly ten times as many parameters as the student.

Why is this important?

The teacher has greater representational capacity and can potentially learn more complex features and relationships in the training data.

The student is deliberately smaller because our objective is to achieve good performance with a computationally efficient model.

### Question 8: Why does the teacher use dropout?

The teacher also uses dropout with a probability of zero point two.

What is the purpose of dropout?

Dropout is a regularisation technique that randomly deactivates some activations during training.

This discourages the network from relying too heavily on particular neurons and can help reduce overfitting.

Remember that our teacher is a relatively large model trained on a restricted dataset.

Regularisation is therefore particularly useful.

# Part 4: Training the Independent Student and Teacher

### Question 9: Why do we train the independent student first?

Before we train the teacher, we need to establish a baseline.

The independent student is trained using the normal supervised learning approach.

It receives an image, produces predictions, and compares those predictions with the correct class labels.

The notebook uses cross-entropy loss.

Can you recall what cross-entropy loss measures?

Pause and think.

Cross-entropy loss measures the difference between the predicted class probabilities and the correct target labels.

When the model assigns a high probability to the correct class, the loss is generally lower.

When it assigns a low probability to the correct class, the loss is higher.

The model uses this loss to update its parameters through backpropagation and an optimisation algorithm called AdamW.

### Question 10: What training settings are used?

The independent student is trained using twenty epochs.

The learning rate is zero point zero zero one.

The weight decay parameter is zero point zero zero zero one.

The learning rate scheduler gradually reduces the learning rate during training.

Why do we use a learning rate scheduler?

A scheduler can help the model make larger improvements earlier in training and smaller, more controlled updates later.

After every epoch, the model is evaluated using the validation dataset.

The notebook records training loss, validation loss, training accuracy, and validation accuracy.

These metrics help us understand how the model learns and whether it generalises well.

### Question 11: What performance do we expect from the independent student?

According to the practical instructions, the independent student should achieve approximately sixty-five to seventy per cent validation accuracy.

This is not a guaranteed result. Actual performance may vary depending on the runtime environment, random initialisation, and training conditions.

However, this gives us an approximate baseline.

Why is this baseline important?

Because we need to determine whether knowledge distillation genuinely improves the performance of the smaller model.

Without a baseline, we cannot reliably measure the benefit of distillation.

### Question 12: Why do we train the teacher separately?

Now that we have trained the independent student, we train the larger teacher model.

The teacher uses the same CIFAR-10 training and validation data.

However, it has a deeper and wider architecture.

The teacher is also trained for twenty epochs.

Its learning rate is zero point zero zero one, and its weight decay is zero point one.

We expect the teacher to achieve approximately ten percentage points higher validation accuracy than the independent student.

Why do we need a stronger teacher?

Because the teacher is intended to provide useful information to the smaller student.

If the teacher performs poorly, there may be little benefit in transferring its predictions.

Once training is complete, the teacher's model parameters are saved in a checkpoint file called `teacher.ckpt`.

This allows us to restore the trained teacher later.

# Part 5: Understanding Knowledge Transfer

### Question 13: What does the teacher actually transfer to the student?

We have now trained our teacher and independent student.

But how exactly can the teacher transfer knowledge?

This is one of the most important questions in the practical.

Let's consider an image containing a horse.

The correct label is horse.

In conventional supervised learning, the model receives the correct answer and learns to predict horse.

This is called learning from hard labels.

A hard label tells us which class is correct, but it does not tell us much about the relationships between the different classes.

For example, it does not tell us whether the image resembles a horse more than a dog, a bird, or a truck.

The teacher model can provide additional information through its predictions.

Suppose the teacher predicts:

- Horse: eighty per cent.
- Deer: ten per cent.
- Dog: five per cent.
- Bird: three per cent.
- Other classes: two per cent combined.

These numbers are illustrative rather than actual experimental results.

What can we learn from these predictions?

The teacher is telling us that horse is the most likely category, but it also considers deer and dog more plausible alternatives than truck or ship.

This provides information about relationships between classes.

These additional predictions are often called soft labels or soft targets.

The student can learn from this richer information rather than relying exclusively on the correct class label.

### Question 14: Why are soft labels more informative than hard labels?

Let's compare the two approaches.

A hard label might simply say:

This image is a horse.

A soft label might say:

This image is most likely a horse, but it also shares some characteristics with a deer and a dog.

Which one provides more information?

The soft label.

It communicates the teacher's relative confidence across all the possible classes.

These relationships are sometimes called dark knowledge.

The student can use this additional information to learn more meaningful class boundaries.

This is the central idea behind response-based knowledge distillation.

# Part 6: Task 1 – Temperature-Scaled Softmax

### Question 15: What is softmax, and why do we need temperature scaling?

Before implementing the distillation loss, we need to understand the teacher's outputs.

The final classification layer of a neural network produces values called logits.

Are logits the same as probabilities?

No.

Logits are raw, unscaled prediction scores.

They can be positive or negative, and they do not necessarily add up to one.

To convert logits into probabilities, we apply the softmax function.

The softmax function converts the logits into a probability distribution.

Each probability lies between zero and one, and all probabilities add up to one.

However, there is a problem.

The teacher may be very confident in its predictions.

For example, it might assign ninety-nine per cent probability to horse and very small probabilities to every other class.

This means that most of the useful information about relationships between alternative classes becomes difficult to observe.

How can we make the teacher's output distribution softer?

We introduce a temperature parameter.

### Question 16: What is temperature in knowledge distillation?

Temperature is a parameter used to control the softness of the probability distribution produced by softmax.

The temperature-scaled softmax formula is:

$$
P_i =
\frac{\exp(z_i/T)}
{\sum_j \exp(z_j/T)}
$$

Here:

- $z_i$ represents the logit for class $i$.
- $T$ represents the temperature.
- The denominator sums the exponentials across all classes.

What happens when the temperature is equal to one?

We obtain the standard softmax function.

What happens when the temperature increases?

The probability distribution becomes softer.

The probabilities become more evenly distributed across the different classes.

This makes it easier to observe the teacher's relative preferences among alternative classes.

For example, instead of assigning nearly all the probability to horse, the teacher might distribute more probability across horse, deer, and dog.

Again, these are illustrative values.

### Question 17: What happens when temperature is too low or too high?

Let's think about this carefully.

If the temperature is too low, the distribution becomes very sharp.

The teacher's strongest prediction dominates, and the smaller probabilities become less informative.

If the temperature is increased, the distribution becomes smoother.

However, if the temperature becomes excessively high, the distribution can become almost uniform.

What is the problem with a uniform distribution?

It contains very little information about which classes the teacher considers more likely.

Therefore, temperature must be chosen carefully.

The practical suggests experimenting with relatively high temperatures, such as twenty.

### Question 18: What do we need to implement in Task 1?

Our first programming task is to complete the function called `temperature_softmax`.

The function receives two inputs.

The first is a tensor containing the model's logits.

The second is the temperature parameter.

What must the function do?

First, divide the logits by the temperature.

Second, apply the softmax function across the class dimension.

Third, return the resulting probability distribution.

In PyTorch, the softmax operation can be implemented using `torch.softmax`, specifying the class dimension.

For a tensor with shape batch size by number of classes, the class dimension is dimension one.

After implementing the function, the notebook uses a temperature of twenty to generate the teacher's softened predictions.

The results are then plotted so that we can compare the standard softmax distribution with the temperature-scaled distribution.

What should we observe?

The temperature-scaled distribution should be smoother, with more probability assigned to alternative classes.

This completes Task 1.

# Part 7: Task 2 – Distillation Loss

### Question 19: How do we measure the difference between teacher and student predictions?

We now understand how to generate soft probability distributions.

The next question is:

How do we measure whether the student is learning to reproduce the teacher's predictions?

We need a loss function.

In this practical, we use a measure called Kullback-Leibler divergence, commonly abbreviated as KL divergence.

KL divergence measures the difference between two probability distributions.

In our case, the two distributions are:

- The teacher's temperature-scaled probability distribution.
- The student's temperature-scaled probability distribution.

If the student produces predictions similar to the teacher, the KL divergence should be small.

If the two distributions are very different, the divergence will be larger.

### Question 20: What is the KL divergence formula?

The mathematical definition is:

$$
D_{KL}(P\parallel Q)
=
\sum_i P(i)
\log\left(\frac{P(i)}{Q(i)}\right)
$$

Here, $P$ represents the reference probability distribution, and $Q$ represents the distribution being compared with it.

For our implementation, we use the convention in PyTorch's KL divergence function:

- The teacher distribution is the target.
- The student distribution is the input.

Therefore, the calculation corresponds to:

$$
D_{KL}(P_{\text{teacher}}
\parallel P_{\text{student}})
$$

This distinction is important because KL divergence is not symmetric.

In other words, exchanging the teacher and student distributions generally produces a different result.

### Question 21: Why do we multiply the KL divergence by temperature squared?

This is a particularly important detail.

When we increase the temperature, the softmax distributions become smoother.

However, the gradients produced by the soft-target loss become smaller.

The practical explains that this scaling is approximately proportional to one divided by temperature squared.

To compensate for this effect, we multiply the KL divergence by $T^2$.

Our final distillation loss is therefore:

$$
L_{\text{distillation}}
=
T^2
D_{KL}
(P_{\text{teacher}}\parallel P_{\text{student}})
$$

Why is this multiplication important?

It helps maintain an appropriate scale for the gradients during training.

Without this adjustment, the distillation loss could become too small when using a high temperature.

### Question 22: What do we implement in Task 2?

We now need to complete the function called `distillation_output_loss`.

The function receives three inputs:

- Student logits.
- Teacher logits.
- Temperature.

What are the steps?

First, apply temperature-scaled softmax to the student logits.

Second, apply temperature-scaled softmax to the teacher logits.

Third, calculate the KL divergence between the two distributions.

Fourth, multiply the divergence by temperature squared.

Finally, return a single scalar loss averaged across the batch.

There is one important implementation detail.

PyTorch's KL divergence function expects log probabilities for its input and probabilities for its target.

Therefore, we need to calculate the logarithm of the student's temperature-scaled probabilities.

We can use:

`torch.log_softmax(student_pred / temperature, dim=1)`

for the student log probabilities.

For the teacher, we use:

`torch.softmax(teacher_pred / temperature, dim=1)`

The teacher is not being trained during this calculation, so its predictions should not require gradients.

In PyTorch, we can use `torch.no_grad()` when generating teacher predictions.

We then use `torch.nn.functional.kl_div` with the reduction set to `batchmean`.

Why is `batchmean` important?

It ensures that the KL divergence is averaged across the batch rather than being dependent on the number of examples.

Finally, we multiply the result by temperature squared.

### Question 23: How do we check whether our distillation loss works?

The notebook provides several sanity checks.

We first compare the student's ordinary softmax predictions with the teacher's predictions.

Then we compare their temperature-scaled probability distributions.

We also calculate the conventional cross-entropy loss between the student's predictions and the correct labels.

This is our hard loss.

Next, we calculate the distillation loss between the student and teacher predictions.

This is our soft loss.

The notebook checks that the soft loss is a scalar value and that it is positive for the example being tested.

Why do we perform these checks?

Because they help us identify implementation errors before we move on to training.

Remember that the hard loss measures disagreement with the correct labels, whereas the soft loss measures disagreement with the teacher's predictions.

They provide two different learning signals.

This completes Task 2.

# Part 8: Task 3 – Teacher-Student Distillation Training

### Question 24: How do we train the final distilled student?

We have reached the most important practical task.

We now need to train a new student model using both the correct class labels and the teacher's predictions.

Remember that this student has the same architecture as our independent student.

We are not making the student larger.

Instead, we are giving it additional supervision.

The independent student learned from the hard labels only.

The distilled student learns from both hard labels and soft targets.

This is the fundamental difference between the two training procedures.

### Question 25: What is the combined loss function?

The distilled student uses two loss components.

The first is the hard loss, calculated using cross-entropy between the student's logits and the correct class labels.

We can write this as:

$$
L_{\text{hard}}
=
CE(y_{\text{true}},z_{\text{student}})
$$

The second is the soft loss, calculated using KL divergence between the teacher's and student's softened predictions.

We can write this as:

$$
L_{\text{soft}}
=
T^2D_{KL}
(P_{\text{teacher}}\parallel P_{\text{student}})
$$

We combine the two losses using a weighting parameter called alpha.

A common formulation is:

$$
L_{\text{total}}
=
(1-\alpha)L_{\text{hard}}
+
\alpha L_{\text{soft}}
$$

What does alpha control?

Alpha determines how much importance we give to the teacher's knowledge compared with the original training labels.

If alpha is zero, the model relies entirely on hard-label supervision.

This is effectively ordinary supervised training.

If alpha is one, the model relies entirely on the teacher's soft predictions.

For values between zero and one, the model combines both sources of information.

Why is this combination useful?

The hard labels provide direct supervision about the correct answers.

The teacher provides additional information about relationships between classes.

Together, these signals can help the student learn a better representation of the classification problem.

### Question 26: Why must we freeze the teacher during student training?

This is another important implementation detail.

During distillation, the teacher provides predictions, but we do not want to update its parameters.

Why?

Because the teacher is already trained and should remain a fixed source of knowledge.

We only want to update the student.

In PyTorch, we can prevent teacher gradients from being calculated by using `torch.no_grad()` when obtaining its predictions.

We also set the teacher to evaluation mode using:

`teacher.eval()`

This is particularly important because our network contains batch normalisation layers.

Batch normalisation behaves differently during training and evaluation.

If the teacher is accidentally left in training mode, its predictions may change because of updates to its internal running statistics.

This could make the teacher's supervision unstable.

Therefore, the teacher should remain in evaluation mode throughout student distillation training.

### Question 27: Why must we set the student to training mode?

At the beginning of every training epoch, we set:

`student.train()`

Why?

Because the student needs to update its parameters through backpropagation.

Training mode also ensures that layers such as batch normalisation and dropout behave appropriately during training.

Remember the distinction:

- Teacher: evaluation mode.
- Student: training mode.

During validation, the student should also be placed in evaluation mode.

### Question 28: How does the distillation training loop work?

Let's walk through the process.

First, initialise a new student model using the same architecture as the independent student.

Second, load the saved teacher checkpoint to ensure that we are using the trained teacher.

Third, select a temperature value.

The notebook encourages experimentation with different positive temperatures, such as twenty.

Fourth, select an alpha value to control the contribution of the teacher's predictions.

Now we can begin training.

For every batch of training images, we perform the following steps.

Step one: send the images through the student to obtain the student logits.

Step two: send the same images through the teacher to obtain the teacher logits.

Step three: calculate the hard loss using the correct labels.

Step four: calculate the soft loss using the teacher and student predictions.

Step five: combine the two losses using the alpha weighting parameter.

Step six: calculate gradients from the total loss.

Step seven: update the student parameters using the optimiser.

Notice something important.

Which model is being updated?

Only the student.

The teacher remains fixed.

After processing all batches in an epoch, we evaluate the student on the validation dataset.

We record the validation loss and accuracy.

We repeat this process for the specified number of epochs.

### Question 29: Why should we be careful when combining the two losses?

The practical provides an important warning.

The hard loss and soft loss can have similar numerical scales.

If we simply add them together, we may increase the overall magnitude of the gradients.

This could make the student training behave differently from the independent student training.

How can we control this?

We use the alpha weighting parameter to balance the two loss components.

We should also keep the learning rate, number of epochs, regularisation, and other training settings consistent with the independent student experiment.

Why?

Because we want to make a fair comparison.

If we change several training settings simultaneously, it becomes difficult to determine whether any improvement is actually due to knowledge distillation.

# Part 9: Comparing the Three Models

### Question 30: How do we determine whether knowledge distillation was successful?

We have now trained all three models.

Let's remind ourselves what they represent.

The first model is the independent student.

It is our baseline.

The second model is the teacher.

It is our upper-performance reference.

The third model is the distilled student.

It is the model we want to evaluate.

The notebook produces training curves comparing their performance.

In particular, we focus on validation accuracy.

Why validation accuracy?

Because it measures the percentage of validation images classified correctly.

It gives us an indication of how well each model generalises to unseen examples.

The practical provides approximate expectations:

- Independent student: around sixty-five to seventy per cent.
- Teacher: approximately ten percentage points higher than the independent student.
- Distilled student: ideally, at least five percentage points higher than the independent student.

These are targets, not guaranteed outcomes.

### Question 31: What does it mean if the distilled student performs better than the independent student?

Suppose the independent student achieves sixty-eight per cent validation accuracy.

Suppose the teacher achieves seventy-eight per cent.

Finally, suppose the distilled student achieves seventy-four per cent.

These are illustrative values.

What can we conclude?

The distilled student has improved by six percentage points compared with the independent student.

It has also recovered six out of the ten percentage points separating the independent student from the teacher.

This means it has recovered sixty per cent of the performance gap.

The distilled student has therefore benefited from the teacher's knowledge while retaining the smaller architecture.

This is exactly the type of improvement we are trying to achieve.

### Question 32: What if the distilled student does not outperform the independent student?

Does this automatically mean that knowledge distillation is ineffective?

No.

There are several possible reasons.

First, the temperature may not be appropriate.

If the temperature is too low, the teacher's predictions may be too sharp.

If it is too high, the predictions may become almost uniform.

Second, the alpha weighting parameter may not be appropriate.

If the soft loss receives too little weight, the student may not benefit sufficiently from the teacher.

If it receives too much weight, the student may rely excessively on the teacher's predictions.

Third, the teacher itself may not be sufficiently accurate.

A weak teacher cannot provide consistently useful information.

Fourth, implementation errors may affect training.

For example, the teacher might accidentally be updated, or the student might be placed in the wrong training mode.

Therefore, a disappointing result should encourage us to investigate the implementation and experiment with the hyperparameters.

### Question 33: How can we measure how much of the teacher-student performance gap has been recovered?

The notebook calculates a value called `gap_closed`.

Let's understand what it means.

Suppose:

- Independent student accuracy is sixty-eight per cent.
- Teacher accuracy is seventy-eight per cent.
- Distilled student accuracy is seventy-four per cent.

The improvement over the independent student is six percentage points.

The original performance gap is ten percentage points.

Therefore:

$$
\text{Gap closed}
=
\frac{\text{Distilled improvement}}
{\text{Teacher-student baseline gap}}
$$

In our example:

$$
\text{Gap closed}
=
\frac{74-68}{78-68}
=
0.6
$$

This is sixty per cent.

What does this tell us?

The distilled student has recovered sixty per cent of the performance difference between the original student and the teacher.

This is a useful way to quantify the effectiveness of knowledge transfer.

# Part 10: Assessment and Reflection

### Question 34: What do you need to submit for this practical?

We have now completed the three main tasks.

Let's review the assessment requirements.

First, you need to report your experimental results.

You should include one or two plots comparing the validation accuracy of the independent student, distilled student, and teacher.

Your plots should clearly identify the three models.

Second, you should write a few sentences explaining any problems you encountered during implementation.

For example:

- Did you experience problems implementing temperature-scaled softmax?
- Did you encounter difficulties calculating KL divergence?
- Did you have problems balancing the hard and soft losses?
- Did you need to experiment with temperature or alpha?
- Did you encounter problems with teacher and student training modes?

Third, you should explain how you addressed those problems.

Finally, reflect on what you learned about knowledge distillation.

### Question 35: What are the main lessons from this practical?

Let's summarise the most important concepts.

First, knowledge distillation allows a smaller neural network to learn from a larger, more powerful neural network.

Second, the larger network is called the teacher, and the smaller network is called the student.

Third, hard labels provide the correct class for each training example.

Fourth, soft labels provide additional information about the teacher's confidence across different classes.

Fifth, temperature scaling makes the teacher's probability distribution smoother and reveals more information about alternative classes.

Sixth, KL divergence measures the difference between the teacher's and student's probability distributions.

Seventh, multiplying the distillation loss by temperature squared helps compensate for the gradient scaling associated with temperature.

Finally, the student is trained using a combination of hard-label loss and soft distillation loss.

The overall objective is to improve the performance of a smaller model without increasing its size.

# Part 11: Final Knowledge Check

Before we finish, let's test your understanding with a few questions.

Take a moment to think about each question before listening to the answer.

Question 1: What is the main purpose of knowledge distillation?

Pause.

The answer is to transfer useful information from a larger teacher model to a smaller student model to improve the student's performance.

Question 2: What are the three models used in this practical?

Pause.

The answer is the independent student, the teacher, and the distilled student.

Question 3: What is the difference between hard labels and soft labels?

Pause.

Hard labels indicate the correct class. Soft labels provide a probability distribution over all classes, including information about alternative predictions.

Question 4: What is the purpose of temperature scaling?

Pause.

Temperature scaling makes the softmax probability distribution smoother, allowing the student to learn more information from the teacher's predictions.

Question 5: What happens when the temperature equals one?

Pause.

We obtain the standard softmax function.

Question 6: Which loss function is used to measure the difference between the teacher's and student's soft predictions?

Pause.

The answer is Kullback-Leibler divergence, or KL divergence.

Question 7: Why do we multiply the KL divergence by temperature squared?

Pause.

To compensate for the reduction in gradient magnitude associated with temperature-scaled soft targets.

Question 8: What are the two components of the total distillation loss?

Pause.

The hard cross-entropy loss and the soft KL divergence loss.

Question 9: Which model's parameters are updated during distillation training?

Pause.

Only the student model's parameters are updated. The teacher remains fixed.

Question 10: How do we determine whether distillation has been successful?

Pause.

We compare the validation accuracy of the distilled student with that of the independent student. Ideally, the distilled student should achieve higher accuracy and move closer to the teacher's performance.

# Conclusion

Congratulations\! You have reached the end of Practical 5 on Knowledge Distillation.

The most important idea to remember is that a powerful neural network can teach a smaller neural network through its predictions.

Instead of learning exclusively from the correct answers, the student also learns from the teacher's understanding of relationships between different classes.

This allows us to transfer useful knowledge into a smaller model that may be more efficient to deploy.

In this practical, you have explored the teacher-student framework, implemented temperature-scaled softmax, calculated the KL divergence distillation loss, and designed a training procedure combining hard and soft supervision.

You have also learned how to evaluate the effectiveness of knowledge distillation by comparing the independent student, distilled student, and teacher.

As a final reflection, consider this question:

Can we build smaller and more efficient neural networks without sacrificing too much predictive performance by transferring knowledge from larger models?

That is the central question behind knowledge distillation.

Thank you for listening, and good luck with your experiments and assessment\!

End of Practical 5 – Knowledge Distillation.
