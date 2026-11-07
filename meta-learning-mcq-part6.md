 # Advanced Machine Learning – Practical 6

## Part 1 of 8: Understanding Reptile and the fundamentals of meta-learning

This MCQ is designed to test whether you genuinely understand the concepts rather than simply memorising the notebook code.

We'll focus on:

- What meta-learning is.
- What Reptile is trying to achieve.
- The difference between ordinary supervised learning and meta-learning.
- Support sets and query sets.
- The intuition behind learning an initialisation that adapts to unseen tasks.

10 multiple-choice questions Answers and explanations included

## Section A – Test your understanding

Your assessment

0/10 answered

## Question 1

What is the primary objective of Reptile?A. To train a neural network to classify only the original training classes.

B. To learn a neural network initialisation that can adapt quickly to new classification tasks.

C. To eliminate the need for a neural network backbone.

D. To maximise accuracy on the support set of every individual task.

## Question 2

What type of meta-learning algorithm is Reptile?A. Metric-based meta-learning.

B. Unsupervised clustering.

C. Gradient-based meta-learning.

D. Reinforcement learning.

## Question 3

In the practical, what is the purpose of generating many small classification tasks from the 80 meta-training classes?A. To ensure that the model memorises all 80 classes.

B. To train a separate backbone for every class.

C. To learn representations and an initialisation that generalise across different tasks.

D. To avoid using support and query sets.

## Question 4

Which statement correctly describes ordinary supervised learning compared with Reptile?A. Both methods necessarily train using exactly the same task structure.

B. Ordinary supervised learning learns to classify fixed classes, whereas Reptile learns an initialisation that adapts across tasks.

C. Reptile does not use neural networks.

D. Ordinary supervised learning always performs better than Reptile.

## Question 5

Suppose a few-shot task contains butterflies, mice and couches. What happens during Reptile's task-specific training?A. The backbone is fine-tuned using the support examples of the task.

B. The backbone is trained using the meta-test query labels.

C. The model immediately calculates class prototypes without training.

D. The model permanently adds these three classes to its original classifier.

## Question 6

What is the role of the query set during a Reptile training episode?A. It replaces the support set during fine-tuning.

B. It is used to initialise the backbone.

C. It evaluates how well the task-adapted model performs on examples not used for its inner-loop updates.

D. It determines the number of convolutional layers.

## Question 7

Which statement best explains why Reptile is useful for few-shot learning?A. It requires thousands of labelled examples for every new class.

B. It learns an initialisation that can be adapted using relatively few labelled examples.

C. It eliminates the need to learn visual features.

D. It uses the query set to memorise every unseen class.

## Question 8

What happens after Reptile fine-tunes its backbone on one task?A. It permanently discards the original model and keeps the adapted model unchanged.

B. It compares the original and adapted parameters and moves the original model towards the adapted parameters.

C. It freezes every parameter permanently.

D. It evaluates only the training accuracy and stops.

## Question 9

Why does Reptile repeatedly sample different tasks during meta-training?A. To ensure that the model learns from a variety of classification problems rather than specialising in one task.

B. To make the support set identical in every episode.

C. To prevent the model from updating its parameters.

D. To replace the backbone after every episode.

## Question 10

Which statement best describes the intended result of successful Reptile meta-training?A. A model that only recognises the 80 meta-training classes without adaptation.

B. A model that has learned a useful starting point for adapting to previously unseen classes.

C. A model that no longer requires feature extraction.

D. A model that achieves 100% accuracy on every possible classification task.

 Submit answers and see explanations Answer all 10 questions to reveal your score and explanations.

## Section B – Essential knowledge to retain

Before moving to Part 2, make sure you can explain these five concepts without looking at the notes:

1. Meta-learning: Learning how to learn across multiple tasks.
2. Reptile: A gradient-based meta-learning algorithm.
3. Support set: The labelled examples used for task-specific adaptation.
4. Query set: The examples used to evaluate generalisation within a task.
5. Meta-training objective: Learning an initialisation that adapts effectively to previously unseen tasks.

My recommendation: Aim for at least 8/10 before moving on. Understanding these foundations will make the Reptile update equation and the notebook implementation considerably easier to master.
