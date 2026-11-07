# Advanced Machine Learning — Practical 6

## Part 1: Understanding the Assignment — Multiple-Choice Questions

Exam preparation 20 MCQs Beginner → Advanced

These questions cover Part 1: Understanding the Assignment (Q1–Q5). They test definitions, conceptual understanding, and the ability to distinguish similar meta-learning concepts.

Each question has four options, followed by the correct answer and an explanation.

## Question 1 — What is few-shot learning?

Which statement best describes few-shot learning?

A. Training a model using millions of labelled examples from known classes.

B. Teaching a model to recognise new categories using only a small number of labelled examples per category.

C. Training a model without using any labelled examples.

D. Using a model to classify images faster by reducing their resolution.

Correct answer: B

Explanation:

Few-shot learning aims to recognise new classes using very few labelled examples.

For example, a model might receive only five labelled images of dogs and five labelled images of ships. It must learn to distinguish these categories using these limited examples.

The central challenge is generalisation to unseen classes with limited labelled data.

- A describes large-scale supervised learning.
- C describes unsupervised or self-supervised learning more closely.
- D concerns computational efficiency rather than few-shot learning.

## Question 2 — What does 5-way, 5-shot classification mean?

A few-shot classification task is described as 5-way, 5-shot. What does this mean?

A. Five tasks, each containing five query images.

B. Five classes, with five labelled support examples per class.

C. Five classes, with five total training images.

D. Five models, each trained for five epochs.

Correct answer: B

Explanation:

- 5-way means the task contains five different classes.
- 5-shot means there are five labelled support examples per class.

Therefore:

$$
5\times5=25
$$

The support set contains 25 labelled images.

Remember:

- Way = number of classes.
- Shot = number of labelled support examples per class.

This distinction is frequently tested in exams.

## Question 3 — What is the main difference between supervised learning and few-shot learning?

Which statement is correct?

A. Supervised learning uses labels, whereas few-shot learning never uses labels.

B. Few-shot learning always requires a larger dataset than supervised learning.

C. Supervised learning generally trains on many examples of known classes, whereas few-shot learning aims to recognise new classes using limited labelled examples.

D. There is no difference between the two approaches.

Correct answer: C

Explanation:

Both approaches can use labelled data.

The difference is their learning objective:

- Supervised learning learns to classify examples from a known class vocabulary.
- Few-shot learning aims to generalise to new categories using only a few labelled examples.

In this practical, the baseline is trained on meta-training classes and evaluated on separate meta-test tasks containing held-out classes.

The important concept is generalisation to unseen classes.

## Question 4 — What is meta-training?

What happens during meta-training?

A. The model is evaluated on the final test dataset without updating its parameters.

B. The model learns from many sampled tasks constructed from meta-training classes.

C. The model memorises the labels of the query set.

D. The model is deployed to classify real-world images.

Correct answer: B

Explanation:

Meta-training involves repeatedly sampling different few-shot tasks.

For each task, the model practises learning from a support set and being evaluated on a query set.

The objective is not simply to learn one fixed classification problem. It is to learn information that transfers across different tasks.

Exam tip: Meta-learning means learning how to learn across tasks.

## Question 5 — What is meta-validation used for?

What is the main purpose of meta-validation?

A. To train the model from scratch.

B. To replace the support set.

C. To monitor performance on held-out tasks and help select hyperparameters or models.

D. To calculate the number of classes in the original dataset.

Correct answer: C

Explanation:

Meta-validation provides an independent way to monitor generalisation during development.

It can be used to:

- Compare different model configurations.
- Select hyperparameters.
- Monitor training progress.
- Choose a suitable checkpoint.

Meta-testing should ideally be reserved for final evaluation.

The distinction is:

| Stage | Purpose |
| --- | --- |
| Meta-training | Learn across sampled tasks |
| Meta-validation | Monitor and select models |
| Meta-testing | Measure final generalisation |

## Question 6 — What is a task in few-shot learning?

Which statement best defines a task?

A. The complete CIFAR dataset.

B. One training epoch.

C. A small classification problem constructed by selecting classes and dividing their images into support and query sets.

D. A single image passed through the backbone.

Correct answer: C

Explanation:

A dataset contains images and labels.

A task is a smaller classification problem generated from that dataset.

A typical task involves:

1. Selecting a number of classes.
2. Sampling labelled support examples.
3. Sampling query examples.
4. Evaluating the model on the query examples.

The task is the fundamental unit of episodic meta-learning.

## Question 7 — What is the support set?

What is the primary purpose of the support set?

A. To measure the final test accuracy.

B. To provide labelled examples used to define or adapt the task-specific classifier.

C. To calculate the learning rate.

D. To update the original dataset labels.

Correct answer: B

Explanation:

The support set contains the few labelled examples available for each class.

Different methods use it differently:

- ProtoNet uses support embeddings to calculate class prototypes.
- Reptile fine-tunes the model using support examples.
- The baseline trains a task-specific classification head using support examples.

The support set provides the information needed to recognise the classes in the current task.

## Question 8 — What is the query set?

Which statement correctly describes the query set?

A. It contains examples used to calculate the class prototypes in ProtoNet.

B. It contains additional examples used to evaluate how well the model classifies the task.

C. It contains only examples from the meta-training classes.

D. It is used to initialise the model's parameters.

Correct answer: B

Explanation:

The query set tests whether the model can generalise beyond the support examples.

In ProtoNet:

- Support examples construct the prototypes.
- Query examples are classified using distances to those prototypes.

During meta-training, query labels also provide the loss used to improve the model.

Remember: Support teaches; query tests.

## Question 9 — What is an episode?

In episodic meta-learning, what does an episode represent?

A. One complete pass through the training dataset.

B. One sampled few-shot classification task.

C. One convolutional layer.

D. One update to the learning-rate scheduler.

Correct answer: B

Explanation:

An episode is one sampled task.

It contains its own support and query sets.

For example:

- Sample five classes.
- Select five support images per class.
- Select fifteen query images per class.
- Perform one task-specific learning and evaluation process.

This is one episode.

Do not confuse:

- Epoch = typically one pass through a training dataset.
- Episode = one sampled meta-learning task.

## Question 10 — How many support and query images are in a 5-way, 5-shot task with 15 query images per class?

A. 5 support and 15 query images.

B. 25 support and 15 query images.

C. 25 support and 75 query images.

D. 75 support and 25 query images.

Correct answer: C

Explanation:

Support images:

$$
5\text{ classes}\times5=25
$$

Query images:

$$
5\text{ classes}\times15=75
$$

Therefore:

- Support set = 25 images.
- Query set = 75 images.
- Total = 100 images.

This is an important numerical question because it tests whether you understand way, shot, and query examples per class.

## Question 11 — Why are meta-training and meta-testing classes separated?

A. To reduce the number of convolutional layers.

B. To ensure that the model is evaluated on its ability to generalise to unseen classes.

C. To make the support set unnecessary.

D. To avoid calculating cross-entropy loss.

Correct answer: B

Explanation:

If the model is evaluated on exactly the same classes it learned during training, we cannot properly establish whether it can generalise to new categories.

Separating the classes tests whether the model has learned transferable representations or learning strategies.

This is called generalisation to unseen classes.

## Question 12 — What is the main objective of meta-learning?

Which option best describes the objective?

A. Learn one fixed classifier that can only recognise training classes.

B. Learn how to learn new tasks efficiently using experience from previous tasks.

C. Eliminate the need for neural network parameters.

D. Increase the number of training examples artificially.

Correct answer: B

Explanation:

Ordinary learning focuses on solving one classification problem.

Meta-learning focuses on learning from multiple tasks so that the model can solve new tasks more effectively.

The fundamental distinction is:

$$
\text{Traditional learning: Learn a task}
$$

$$
\text{Meta-learning: Learn how to learn tasks}
$$

This is arguably the most important concept in Part 1.

## Question 13 — What happens during meta-testing?

Which statement is correct?

A. The model is trained on the meta-test classes using thousands of labelled examples.

B. The model is evaluated on separate tasks to measure its generalisation to unseen classes.

C. The model's architecture is changed for every query image.

D. The model ignores the support set.

Correct answer: B

Explanation:

Meta-testing measures the model's ability to solve new few-shot tasks.

The tasks use held-out classes that were not used during meta-training.

For ProtoNet, the support set is used to construct prototypes, and the query set is used to calculate classification performance.

Meta-testing is about measuring generalisation, not training on the final test classes.

## Question 14 — Which statement about support and query sets is TRUE?

A. Both sets must contain exactly the same images.

B. The support set is used to define the task-specific classifier, while the query set evaluates it.

C. The query set always contains fewer images than the support set.

D. The support set is only used during meta-testing.

Correct answer: B

Explanation:

The support and query sets have different roles.

- Support: provides the labelled information needed to solve the task.
- Query: measures whether the model can classify additional examples.

The number of examples in each set depends on the task configuration.

They should not be confused or treated as interchangeable.

## Question 15 — Which of the following is the correct sequence for a typical ProtoNet episode?

A. Query images → prototypes → support labels → classification.

B. Support images → embeddings → prototypes → query embeddings → distance-based classification.

C. Support images → cross-entropy → query images → backbone initialisation.

D. Query labels → prototypes → support images → classification.

Correct answer: B

Explanation:

ProtoNet follows this sequence:

1. Pass support images through the backbone.
2. Generate support embeddings.
3. Calculate class prototypes by averaging embeddings.
4. Pass query images through the same backbone.
5. Calculate distances between query embeddings and prototypes.
6. Use negative distances as classification logits.

The support set must establish the prototypes before the query examples can be classified.

## Question 16 — What is the difference between a dataset and an episode?

A. A dataset is one image, whereas an episode contains the entire training dataset.

B. A dataset contains images and labels, whereas an episode is a sampled classification task constructed from those images.

C. They mean exactly the same thing.

D. An episode contains only unlabelled images.

Correct answer: B

Explanation:

This is a conceptual distinction that often appears in oral exams.

A dataset is the underlying collection of data.

An episode is a temporary task created from that data.

The same dataset can generate many different episodes by sampling different classes and examples.

This allows the model to practise solving a variety of tasks.

## Question 17 — Why is evaluating on unseen classes important?

A. It guarantees 100% accuracy.

B. It tests whether the model can transfer what it learned to categories that were excluded from meta-training.

C. It eliminates the need for labelled support examples.

D. It guarantees that the model cannot overfit.

Correct answer: B

Explanation:

Few-shot learning is specifically concerned with learning new categories from limited examples.

If the evaluation uses only familiar training classes, it does not adequately test this capability.

Held-out classes allow us to assess whether the learned representations or adaptation strategies transfer to new categories.

## Question 18 — A model achieves 95% accuracy on its original training classes but only 30% on meta-test tasks. What does this suggest?

A. The model necessarily has an excellent few-shot learning ability.

B. The model performs well on familiar classes but struggles to generalise to unseen classes.

C. The model has no trainable parameters.

D. The query set must be empty.

Correct answer: B

Explanation:

High training accuracy does not guarantee successful few-shot generalisation.

The model may have learned features that work well for its original classes but are not sufficiently transferable to new categories.

This is precisely why meta-test query accuracy is important.

## Question 19 — Which statement best describes episodic training?

A. Training the model using only one fixed classification task.

B. Repeatedly sampling different tasks and training the model across those tasks.

C. Randomly deleting labels from the dataset.

D. Evaluating the model without using any images.

Correct answer: B

Explanation:

Episodic training repeatedly exposes the model to different classification tasks.

Each episode provides a new combination of classes and support/query examples.

This encourages the model to learn transferable knowledge rather than memorising one fixed classification problem.

## Question 20 — Which statement summarises Part 1 most accurately?

A. Few-shot learning is primarily about increasing the number of labelled examples.

B. Meta-learning trains a model to solve only one fixed classification task.

C. Few-shot learning uses limited labelled examples, while meta-learning aims to develop knowledge that helps a model learn new tasks efficiently.

D. Support and query sets are interchangeable.

Correct answer: C

Explanation:

This answer combines the two central ideas:

- Few-shot learning: recognising new categories from a small number of labelled examples.
- Meta-learning: learning across tasks so that new tasks can be solved efficiently.

Support examples provide the information needed to solve a task, while query examples evaluate generalisation.

This is the foundation for understanding the baseline, Reptile, and ProtoNet in the following parts.

# Final revision sheet — Part 1

These are the essential facts to memorise before an assessment.

| Concept | What you must remember |
| --- | --- |
| Few-shot learning | Learn new classes using few labelled examples. |
| 5-way | Five classes in one task. |
| 5-shot | Five support examples per class. |
| Dataset | Collection of images and labels. |
| Task | A small classification problem sampled from a dataset. |
| Episode | One sampled few-shot task. |
| Support set | Labelled examples used to define/adapt the classifier. |
| Query set | Additional examples used to evaluate classification. |
| Meta-training | Learning across sampled tasks. |
| Meta-validation | Monitoring performance and selecting hyperparameters. |
| Meta-testing | Evaluating generalisation to unseen tasks. |
| Generalisation | Applying learned knowledge to new classes/tasks. |
| Episodic training | Repeatedly training across different tasks. |

## Test yourself: The 5 most important exam questions

If you can answer these five without looking at the explanations, you have a strong understanding of Part 1.

1. What is the difference between few-shot learning and ordinary supervised learning?
2. Explain the difference between meta-training, meta-validation, and meta-testing.
3. What are the support set and query set, and why do we need both?
4. What is an episode, and how does it differ from an epoch?
5. Why must meta-test classes be separate from meta-training classes?

My recommendation: Make sure you can explain Questions 1, 3, 4, 7, 8, and 11 in your own words. These cover the conceptual foundations that you will need to understand Reptile and Prototypical Networks in the later parts.
