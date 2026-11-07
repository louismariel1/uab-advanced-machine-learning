# Advanced Machine Learning — Practical 6

## Part 9: Advanced Understanding and Exam Preparation

MCQ Practice Questions with Answers and Explanations

Part 9 focuses on the deeper conceptual understanding of few-shot learning, meta-learning, Reptile, and Prototypical Networks (ProtoNet).

This section is particularly important for oral examinations, written assessments, and questions that require you to explain why the algorithms work rather than simply describe their implementation.

25 MCQsAnswers includedDetailed explanations

## Section A — Fundamental concepts of meta-learning

### Question 1. What is the fundamental difference between ordinary supervised learning and meta-learning?

A. Supervised learning uses neural networks, whereas meta-learning does not.

B. Supervised learning learns to solve a particular prediction problem, whereas meta-learning learns how to adapt across different tasks.

C. Meta-learning does not require training data.

D. Meta-learning always uses fewer parameters than supervised learning.

Correct answer: BExplanation: Ordinary supervised learning typically learns a mapping from inputs to labels for a fixed collection of classes. Meta-learning aims to learn knowledge that can transfer across tasks, allowing a model to adapt to new tasks using limited examples.

The central idea is:

$$
\boxed{\text{Meta-learning = Learning to learn}}
$$

### Question 2. What is the main objective of few-shot learning?

A. To classify images using millions of labelled examples.

B. To recognise new categories using only a small number of labelled examples per category.

C. To eliminate the need for labelled data.

D. To increase the number of output neurons in a fixed classifier.

Correct answer: BExplanation: Few-shot learning aims to recognise unfamiliar categories using only a few labelled examples.

For example, a five-way, five-shot task involves:

- Five different classes.
- Five labelled support examples per class.
- A total of 25 support examples.

The model must then classify additional query examples.

### Question 3. What does an episode represent in meta-learning?

A. One complete pass through the entire training dataset.

B. One sampled few-shot classification task containing support and query examples.

C. One update of the learning-rate scheduler.

D. One individual convolutional layer.

Correct answer: BExplanation: An episode represents one small classification task constructed by sampling classes and dividing their examples into support and query sets.

Meta-learning models repeatedly practise solving different episodes during meta-training.

### Question 4. Why are meta-training and meta-testing performed using separate classes?

A. To make the training dataset larger.

B. To test whether the model can generalise to previously unseen categories.

C. To ensure that training accuracy reaches 100%.

D. To prevent the model from using a backbone.

Correct answer: BExplanation: Meta-training uses one collection of classes, while meta-testing uses held-out classes.

This separation measures whether the model has learned transferable knowledge rather than simply memorising the original training categories.

### Question 5. What is the main purpose of the support and query sets?

A. Both sets are used only to train the original classification head.

B. The support set defines or adapts the task-specific classifier, while the query set evaluates its performance.

C. The support set calculates accuracy, while the query set creates the backbone.

D. Both sets must contain identical images.

Correct answer: BExplanation: The support set provides labelled examples that help the model recognise the task's classes. The query set contains additional examples used to measure how well the model generalises within that task.

This distinction is central to episodic learning.

## Section B — Advanced understanding of Reptile

### Question 6. What does Reptile attempt to learn?

A. A fixed classifier for the original training classes.

B. An initial parameter configuration that can adapt quickly to different tasks.

C. A collection of class prototypes.

D. A model that never requires gradient updates.

Correct answer: BExplanation: Reptile is a gradient-based meta-learning algorithm. Its goal is to find a useful initialisation from which task-specific learning can happen quickly.

Rather than optimising only for one classification problem, it seeks parameters that are useful across many sampled tasks.

### Question 7. Which equation represents the Reptile meta-update?

A.

$$
\theta\leftarrow\theta-\epsilon\theta_i'
$$

B.

$$
\theta\leftarrow\theta+\epsilon(\theta_i'-\theta)
$$

C.

$$
\theta\leftarrow\theta_i'-\epsilon
$$

D.

$$
\theta\leftarrow\theta+\theta_i'
$$

Correct answer: BExplanation: The Reptile update moves the current meta-parameters towards the parameters obtained after task-specific fine-tuning.

Where:

- $\theta$: Original meta-parameters.
- $\theta_i'$: Task-adapted parameters.
- $\epsilon$: Meta-step size.

The update is:

$$
\boxed{\theta_{\text{new}}=\theta+\epsilon(\theta_i'-\theta)}
$$

### Question 8. Suppose a Reptile parameter starts at 4 and task fine-tuning changes it to 10. If the meta-step size is 0.5, what is the updated parameter?

A. 4.

B. 6.

C. 7.

D. 10.

Correct answer: CExplanation:

Using the Reptile equation:

$$
\theta_{\text{new}}
=4+0.5(10-4)
$$

$$
=4+3=7
$$

The meta-model moves halfway towards the task-adapted parameter.

### Question 9. What happens if the Reptile meta-step size is equal to 1?

A. The meta-model does not change.

B. The meta-model moves completely to the task-adapted parameters for that update.

C. The meta-model moves in the opposite direction.

D. The task-specific fine-tuning is cancelled.

Correct answer: BExplanation: When $\epsilon=1$:

$$
\theta_{\text{new}}
=
\theta+(\theta_i'-\theta)
=\theta_i'
$$

The meta-model becomes equal to the task-adapted parameters.

### Question 10. Why does Reptile fine-tune its model on the support set before performing the meta-update?

A. To calculate class prototypes.

B. To discover which parameter changes help the model solve the sampled task.

C. To calculate the original baseline accuracy.

D. To remove the backbone.

Correct answer: BExplanation: Fine-tuning produces task-adapted parameters. Reptile then uses the difference between the original and adapted parameters to update the meta-initialisation.

The model learns from the adaptation process across multiple tasks.

## Section C — Advanced understanding of Prototypical Networks

### Question 11. What is the fundamental idea behind ProtoNet?

A. Learn a separate fixed classification head for every possible class.

B. Classify query examples by comparing their embeddings with class prototypes.

C. Fine-tune the backbone separately for every query image.

D. Predict classes using only raw pixel values.

Correct answer: BExplanation: ProtoNet learns an embedding space where examples from the same class should be close together.

It calculates a prototype for each class by averaging its support embeddings and classifies query examples according to their distances from these prototypes.

### Question 12. How is a class prototype calculated?

A. By selecting the query image with the highest confidence.

B. By calculating the mean embedding of the labelled support examples belonging to that class.

C. By calculating the largest Euclidean distance.

D. By randomly selecting a feature vector.

Correct answer: BExplanation: A prototype is the mean of the support embeddings for a particular class:

$$
c_k=
\frac{1}{|S_k|}
\sum_{(x_i,y_i)\in S_k}f_\theta(x_i)
$$

Here, $S_k$ is the support subset belonging to class $k$.

The prototype acts as a representative feature vector for that class.

### Question 13. Why does ProtoNet use an embedding space?

A. To convert images into numerical representations that capture useful visual similarities.

B. To increase the number of labelled images.

C. To eliminate the need for a neural network.

D. To convert query labels into pixels.

Correct answer: AExplanation: The backbone transforms each image into a numerical feature vector.

A good embedding space places images from the same category close together and images from different categories farther apart.

This makes prototype-based classification possible.

### Question 14. Why does ProtoNet use Euclidean distance?

A. To measure how close a query embedding is to each class prototype.

B. To calculate the learning rate.

C. To determine the number of support examples.

D. To calculate the number of convolutional layers.

Correct answer: AExplanation: Euclidean distance measures the geometric distance between two feature vectors:

$$
d(z,c_k)=
\sqrt{\sum_j(z_j-c_{k,j})^2}
$$

A smaller distance indicates that the query embedding is more similar to the class prototype in the learned embedding space.

### Question 15. Why does ProtoNet use negative distances as classification logits?

A. Because larger distances should indicate stronger class predictions.

B. Because smaller distances should correspond to larger logits and therefore higher class probabilities.

C. Because negative numbers cannot be used in cross-entropy.

D. Because it removes the need for query labels.

Correct answer: BExplanation: A distance has the opposite interpretation to an ordinary classification logit.

A smaller distance indicates a better match. Negating the distance reverses the ordering:

$$
z_k=-d(z,c_k)
$$

The closest prototype receives the largest logit.

### Question 16. What is the purpose of temperature in ProtoNet?

A. To change the number of classes.

B. To control the scale of the negative-distance logits and the sharpness of the resulting probability distribution.

C. To change the support-set size.

D. To calculate the average prototype.

Correct answer: BExplanation: ProtoNet uses:

$$
z_k=-\frac{d(z,c_k)}{T}
$$

A larger positive temperature generally produces softer probability distributions, while a smaller positive temperature produces sharper distributions.

Temperature does not change which prototype is closest.

### Question 17. Why can ProtoNet operate without a conventional learned linear classification head?

A. Because the support-set prototypes act as task-specific class representatives.

B. Because it does not need to classify query images.

C. Because its backbone has no trainable parameters.

D. Because the model ignores the class labels.

Correct answer: AExplanation: A conventional classifier learns fixed output weights for predefined classes.

ProtoNet calculates new prototypes from the support examples of each task. Therefore, it can classify classes that were not present during meta-training without requiring a new learned linear classification layer.

## Section D — Comparing Reptile and ProtoNet

### Question 18. What is the most important difference between Reptile and ProtoNet?

A. Reptile uses images, whereas ProtoNet uses text.

B. Reptile learns an adaptable parameter initialisation, whereas ProtoNet learns an embedding space for distance-based classification.

C. Both use identical task-adaptation mechanisms.

D. ProtoNet does not use meta-training.

Correct answer: BExplanation: The two methods learn to generalise across tasks in different ways.

- Reptile: Learns parameters that can be quickly adapted using gradient updates.
- ProtoNet: Learns embeddings that allow new classes to be recognised through support-set prototypes.

This is one of the most important comparison questions in Practical 6.

### Question 19. What happens when a new task is presented to Reptile?

A. It calculates prototypes and performs no adaptation.

B. It fine-tunes the model parameters on the support set.

C. It ignores the support set.

D. It replaces the backbone with a linear classifier.

Correct answer: BExplanation: Reptile adapts the model using gradient-based fine-tuning on task-specific support examples.

This is how its learned initialisation helps solve new tasks.

### Question 20. What happens when a new task is presented to ProtoNet during meta-testing?

A. It performs several gradient updates to fine-tune the backbone.

B. It calculates new class prototypes from the support embeddings and classifies query embeddings using distances.

C. It creates a completely new neural network.

D. It uses the original meta-training classification head.

Correct answer: BExplanation: ProtoNet uses the learned backbone to extract support and query embeddings.

It constructs prototypes from the support embeddings and calculates negative distances for query classification.

No gradient-based backbone fine-tuning is required during meta-testing.

### Question 21. Which statement about the ProtoNet training process is correct?

A. ProtoNet cannot be trained because it has no linear classification head.

B. ProtoNet trains its backbone using query-set classification loss.

C. ProtoNet only uses support-set loss.

D. ProtoNet does not require backpropagation.

Correct answer: BExplanation: Although ProtoNet has no conventional learned classification head, its backbone contains trainable parameters.

The query classification loss is calculated using prototype-based logits. Backpropagation updates the backbone so that its embeddings become more useful for few-shot classification.

### Question 22. Why is query-set loss particularly important in ProtoNet meta-training?

A. It measures whether the embedding space allows additional examples to be classified correctly using support-set prototypes.

B. It calculates the number of classes.

C. It replaces the backbone.

D. It trains only the original supervised classification head.

Correct answer: AExplanation: The support set defines the prototypes, while the query set tests whether those prototypes generalise to additional examples.

The query loss therefore encourages the backbone to learn representations that support effective few-shot classification.

## Section E — Advanced reasoning and oral examination

### Question 23. Why can a model achieve high meta-training accuracy but low meta-test accuracy?

A. Because meta-training and meta-testing always use identical classes.

B. Because the model may not have learned transferable representations or may have overfitted to the meta-training task distribution.

C. Because meta-test tasks do not contain labels.

D. Because high training accuracy automatically means high generalisation.

Correct answer: BExplanation: Strong performance on training tasks does not guarantee that the model will generalise to held-out classes.

Possible causes include overfitting, an unsuitable embedding space, insufficient task diversity, and poorly selected hyperparameters.

### Question 24. Which statement best explains why ProtoNet is particularly suitable for recognising new classes?

A. It requires a fixed output neuron for every possible future class.

B. It constructs class prototypes dynamically from the support examples of each new task.

C. It memorises every possible image category.

D. It does not require a learned feature representation.

Correct answer: BExplanation: ProtoNet does not depend on a fixed classification head containing all possible classes.

Its prototypes are calculated dynamically from the support examples, allowing it to classify previously unseen categories.

### Question 25. What is the most important overall lesson from Practical 6?

A. Ordinary supervised learning is always superior to meta-learning.

B. Reptile and ProtoNet are identical algorithms.

C. Learning to classify a fixed collection of classes is different from learning to recognise new classes using only a few labelled examples.

D. Few-shot learning eliminates the need for neural networks.

Correct answer: CExplanation: Practical 6 demonstrates three approaches to few-shot classification:

- Baseline: Learns visual representations through ordinary supervised training.
- Reptile: Learns an initialisation that can adapt quickly to new tasks.
- ProtoNet: Learns an embedding space in which new classes can be represented by support-set prototypes.

The central lesson is that successful few-shot learning requires generalisation across tasks and previously unseen categories.

# Final revision summary — Part 9

| Concept | Essential knowledge |
| --- | --- |
| Meta-learning | Learning how to learn across different tasks |
| Few-shot learning | Recognising new classes with limited labelled examples |
| Episode | One sampled classification task |
| Support set | Defines or adapts the task-specific classifier |
| Query set | Measures task generalisation |
| Reptile | Learns an adaptable initialisation |
| Reptile update | Moves parameters towards task-adapted parameters |
| ProtoNet | Learns an embedding space |
| Prototype | Mean support embedding for a class |
| Euclidean distance | Measures distance between query embeddings and prototypes |
| Negative distances | Convert distances into classification logits |
| Temperature | Controls the scale of logits |
| Query loss | Trains ProtoNet's embedding backbone |
| Meta-test accuracy | Measures generalisation to unseen tasks |

## Five particularly important oral-exam questions

Make sure you can explain these without looking at your notes:

1. What does it mean to learn how to learn?
2. How does Reptile update its initial parameters?
3. Why can ProtoNet classify new classes without a conventional classification head?
4. What is the fundamental difference between Reptile's adaptation mechanism and ProtoNet's?
5. Why does ProtoNet use query-set loss to train its embedding backbone?

## Final takeaway

The most important distinction to remember for your examination is:

$$
\boxed{\text{Reptile: Learn an adaptable initialisation}}
$$

$$
\boxed{\text{ProtoNet: Learn an embedding space for comparison}}
$$

Both methods aim to generalise to unseen classification tasks, but they achieve this through fundamentally different mechanisms.

You have now covered Parts 1–9 of Practical 6. You can use these MCQs as a question bank for your final revision and oral examination preparation.
