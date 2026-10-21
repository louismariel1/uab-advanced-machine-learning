# Advanced Machine Learning — MCQ: Knowledge Distillation and Self-Distillation

This comprehensive multiple-choice question is designed to clarify the fundamental concepts, objectives, mechanisms, and differences between knowledge distillation, self-distillation, and related teacher–student learning approaches.

## Question: Distillation in Deep Learning

Advanced

12 points · 12 questions · Single best answer per question

A research team wants to improve the efficiency and generalization of deep neural networks using knowledge distillation and self-distillation. They consider teacher–student training, soft targets, temperature scaling, intermediate representations, and teacher-free alternatives.

Select the single most accurate answer for each question.

## Part A — Fundamental concepts

### Question 1. What is the primary objective of knowledge distillation? (1 point)

A. To train a neural network exclusively on unlabeled data without any supervisory signal.

B. To transfer useful information from a teacher model to a student model, often allowing the student to achieve competitive performance with lower computational cost.

C. To guarantee that a smaller student network performs better than its teacher on every test example.

D. To reduce the number of training samples required to zero by memorizing the teacher's predictions.

## Part B — Teacher, student, and soft targets

Choose an answer for each question.

Question 2. In conventional knowledge distillation, what are the roles of the teacher and student? (1 point)

A. The student generates labels, and the teacher learns to reproduce them.

B. The teacher and student must have identical parameters and architecture.

C. The teacher provides supervisory signals, and the student learns to approximate useful aspects of the teacher's behavior.

D. The teacher is used only after training to correct the student's predictions.

Question 3. Why can a teacher's soft probability targets provide more information than one-hot labels? (1 point)

A. They express relative similarity between classes through the probability assigned to each class.

B. They always contain the correct label for every example.

C. They remove the need for a loss function.

D. They guarantee that the student will have lower test error.

Question 4. What is the primary effect of increasing the softmax temperature \\(T\\) above 1? (1 point)

A. It makes the distribution more peaked around the most likely class.

B. It increases the number of classes in the dataset.

C. It guarantees a more accurate teacher.

D. It produces a softer probability distribution, exposing more information about non-top classes.

## Part C — Distillation losses and temperature

### Question 5. A common knowledge distillation objective combines a hard-label loss with a teacher–student loss. Which formulation best represents this approach? (1 point)

A. Minimize only the teacher's training loss, without updating the student.

B. Combine supervised cross-entropy on ground-truth labels with a divergence loss between the teacher's and student's softened output distributions.

C. Maximize the distance between the teacher's and student's probability distributions.

D. Replace the student model's parameters with the teacher's parameters after every minibatch.

## Part D — Understanding self-distillation

### Question 6. Which statement best describes self-distillation? (1 point)

A. Training a model from scratch without labels, augmentations, or other learning signals.

B. Compressing every teacher into a student with fewer parameters.

C. Training a model using signals produced by the same model, an earlier or updated version of it, or a related branch of the same network.

D. Using a teacher model exclusively for data augmentation, without a learning objective.

### Question 7. Which statement most accurately distinguishes conventional knowledge distillation from self-distillation? (1 point)

A. Conventional distillation often transfers knowledge from a separate teacher, whereas self-distillation obtains supervisory signals from the same model family, training process, or related model instance.

B. Conventional distillation uses only hard labels, whereas self-distillation always uses soft labels.

C. Self-distillation requires two different datasets, whereas conventional distillation requires one.

D. Conventional distillation is only possible for convolutional networks, whereas self-distillation is restricted to transformers.

## Part E — Advanced mechanisms

### Question 8. A student network is trained to match the intermediate feature maps of a teacher network. What type of distillation is being used? (1 point)

A. Label smoothing only.

B. Adversarial training.

C. Parameter pruning.

D. Feature-based or intermediate-representation distillation.

### Question 9. Which statement about the benefits of self-distillation is most accurate? (1 point)

A. It always improves generalization because the model's own predictions are necessarily correct.

B. It can improve generalization through additional training signals or regularization, but its success depends on the method, targets, and training setup.

C. It guarantees that the student will converge to the global optimum.

D. It eliminates the need for validation data and hyperparameter selection.

### Question 10. During conventional knowledge distillation, what typically happens to the teacher's parameters? (1 point)

A. They are randomly reinitialized before each student update.

B. They are replaced by the student's parameters after every gradient step in all distillation methods.

C. They are usually kept fixed during student training, although some variants update the teacher or use an exponential moving average.

D. They are always jointly optimized using exactly the same loss and gradients as the student.

## Part F — Synthesis and practical understanding

### Question 11. A researcher trains a network using the predictions of an earlier checkpoint of the same architecture. The student has the same number of parameters as the earlier checkpoint. Which conclusion is correct? (1 point)

A. This can qualify as self-distillation; a reduction in parameter count is not required.

B. This cannot be distillation because the teacher and student must have different architectures.

C. This is necessarily pruning because the two models have equal capacity.

D. This is conventional knowledge distillation only if the teacher is larger.

### Question 12. After completing standard offline knowledge distillation, what is generally required to make predictions with the trained student? (1 point)

A. Both teacher and student must run simultaneously for every prediction.

B. The teacher must generate fresh soft targets for every inference input.

C. The teacher must be loaded to initialize each student forward pass.

D. The student can generally make predictions independently, without running the teacher.

## Submit your answers

### Your progress

0 of 12 answered

# 0/12

Submit and reveal answers

## Key concepts to remember

| Concept                | Essential idea                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------- |
| Knowledge distillation | A student learns from a teacher's outputs, features, or other supervisory signals.                |
| Soft targets           | Probability distributions convey information about relationships between classes.                 |
| Temperature \\(T\\)    | Higher temperatures soften output distributions; lower temperatures sharpen them.                 |
| Self-distillation      | A model learns from its own predictions, earlier versions, updated versions, or related branches. |
| Feature distillation   | Intermediate representations are matched, not just final predictions.                             |
| Model compression      | A common motivation for distillation, but not a defining requirement.                             |
| Inference              | In standard offline distillation, the student usually runs independently of the teacher.          |

The most important distinction: knowledge distillation describes the transfer of learning signals from a teacher to a student. Self-distillation is a related approach in which those signals originate from the same model, a related branch, or another version of the model. Self-distillation does not necessarily involve a smaller student or a separate pretrained teacher.
